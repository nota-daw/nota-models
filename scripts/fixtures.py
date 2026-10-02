# Usage: fixtures.py <song.wav> [out-dir]. Run from the folder that holds htdemucs.onnx.
# Golden fixtures for the C# ports: htdemucs STFT + apply_model, basic-pitch inference + decoding.
import sys, os, json, numpy as np, soundfile as sf, librosa, torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
out = (sys.argv[2] if len(sys.argv) > 2 else "fx") + "/"; os.makedirs(out, exist_ok=True)
wav, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
wav = wav[10 * sr: 30 * sr].T                                   # 20 s, (2, n)
w44 = np.stack([librosa.resample(c, orig_sr=sr, target_sr=44100, res_type="soxr_vhq") for c in wav]).astype(np.float32)
w44.tofile(out + "mix44.f32")                                   # planar L then R

m = get_model("htdemucs").models[0].eval()
L = 343980
x = torch.from_numpy(w44[:, :L].copy())[None]
z = m._spec(x); mag = m._magnitude(z)
mag[0].numpy().astype(np.float32).tofile(out + "mag.f32")      # (4, 2048, 336)
m._ispec(z, L)[0].numpy().astype(np.float32).tofile(out + "ispec.f32")

ref = torch.from_numpy(w44).mean(0)
mix = (torch.from_numpy(w44) - ref.mean()) / ref.std()
with torch.no_grad():
    stems = apply_model(m, mix[None], shifts=0, split=True, overlap=0.25, progress=False)[0]
stems = stems * ref.std() + ref.mean()
stems.numpy().astype(np.float32).tofile(out + "stems.f32")      # (4, 2, n)
print("stems", tuple(stems.shape))

import basic_pitch.inference as bi
from basic_pitch import ICASSP_2022_MODEL_PATH, note_creation as infer
mono = librosa.resample(wav.mean(0), orig_sr=sr, target_sr=22050, res_type="soxr_vhq").astype(np.float32)
mono.tofile(out + "mono22.f32")
bi.librosa.load = lambda *a, **k: (mono, 22050)
onnx = [p for p in __import__("glob").glob(str(ICASSP_2022_MODEL_PATH.parent) + "/**/*.onnx", recursive=True)][0]
mo = bi.run_inference("x.wav", bi.Model(onnx))
mo["note"].astype(np.float32).tofile(out + "bp_note.f32"); mo["onset"].astype(np.float32).tofile(out + "bp_onset.f32")
_, events = infer.model_output_to_notes(mo, onset_thresh=0.5, frame_thresh=0.3, min_note_len=11, include_pitch_bends=False)
json.dump({"frames": int(mo["note"].shape[0]), "notes": [[float(s), float(e), int(p), float(a)] for s, e, p, a, _ in events]}, open(out + "bp_notes.json", "w"))
print("bp frames", mo["note"].shape, "notes", len(events))
import shutil; shutil.copy(onnx, "nmp.onnx")
