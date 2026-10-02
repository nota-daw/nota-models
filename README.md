# Nota models

AI models that [Nota](https://github.com/nota-daw/nota) downloads on request
(Settings → Downloads → AI Models) and the scripts that produced them. Nota pins every file
here to its URL, size and sha256 (`src/managed/Nota.Infrastructure/ModelStore/ModelCatalog.cs`),
so a release asset is never replaced — a new export is a new release.

| Release | Asset | What | License |
|---|---|---|---|
| `htdemucs-v1` | `htdemucs.onnx` (174 264 717 bytes, sha256 `04c2b1a1…c78`) | Demucs v4 `htdemucs` for Separate Stems | MIT (Meta) |

basic-pitch (Convert to MIDI) and ONNX Runtime are not re-hosted: Nota fetches them from
[spotify/basic-pitch](https://github.com/spotify/basic-pitch) (v0.4.0) and
[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2).

## htdemucs.onnx

`scripts/export_htdemucs.py` exports the network of Meta's pretrained `htdemucs` (Demucs
4.1.0) with its STFT and iSTFT left out — ONNX has no complex STFT. The model takes

- `mix`  `[1, 2, 343980]` — one 7.8 s stereo segment at 44.1 kHz, normalised
- `mag`  `[1, 4, 2048, 336]` — its spectrogram, real/imaginary parts as channels

and returns

- `spec` `[1, 4, 4, 2048, 336]` — a complex spectrogram per stem (drums, bass, other, vocals)
- `wave` `[1, 4, 2, 343980]` — a waveform per stem

A stem is iSTFT(`spec`) + `wave`. Nota's C# reproduces the rest of Demucs (spectrogram,
normalisation, overlapping segments): `src/managed/Nota.Infrastructure/Ai/`. The script
checks the wrapper against the full PyTorch model (max error 0.0) before exporting.

```sh
uv venv -p 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/export_htdemucs.py htdemucs.onnx
shasum -a 256 htdemucs.onnx
```

## Golden fixtures

`scripts/fixtures.py <song.wav> [out-dir]` writes 20 s of reference output from the Python
originals (htdemucs spectrogram, iSTFT and `apply_model` stems; basic-pitch activations and
notes) for Nota's golden test:

```sh
.venv/bin/python scripts/fixtures.py song.wav fx      # also copies basic-pitch's nmp.onnx here
dotnet run --project <nota>/tests/Nota.SmokeTest -- --ai-golden <libonnxruntime> .
```
