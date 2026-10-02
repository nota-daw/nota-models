# Usage: export_htdemucs.py [out.onnx]. See README.md.
# Export htdemucs to ONNX with the STFT/iSTFT moved out of the graph: the host computes the
# mixture spectrogram (cac layout) and turns the frequency-branch output back into audio.
import math, sys, torch, torch.nn.functional as F
from demucs.pretrained import get_model
from demucs.htdemucs import HTDemucs

bag = get_model("htdemucs")
m: HTDemucs = bag.models[0]
m.eval()
L = int(m.segment * m.samplerate)
print("segment", m.segment, "samples", L, "sources", m.sources, "nfft", m.nfft, "hop", m.hop_length)

class Core(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, mix, mag):
        m = s.m
        x = mag
        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True); std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True); stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)
        saved, saved_t, lengths, lengths_t = [], [], [], []
        for idx, encode in enumerate(m.encoder):
            lengths.append(x.shape[-1]); inject = None
            if idx < len(m.tencoder):
                lengths_t.append(xt.shape[-1]); tenc = m.tencoder[idx]; xt = tenc(xt)
                if not tenc.empty: saved_t.append(xt)
                else: inject = xt
            x = encode(x, inject)
            if idx == 0 and m.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = m.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + m.freq_emb_scale * emb
            saved.append(x)
        if m.crosstransformer:
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = m.channel_upsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = m.channel_upsampler_t(xt)
            x, xt = m.crosstransformer(x, xt)
            if m.bottom_channels:
                b, c, f, t = x.shape
                x = m.channel_downsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = m.channel_downsampler_t(xt)
        for idx, decode in enumerate(m.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            offset = m.depth - len(m.tdecoder)
            if idx >= offset:
                tdec = m.tdecoder[idx - offset]; length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]; xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1); xt, _ = tdec(xt, skip, length_t)
        S = len(m.sources)
        x = x.view(B, S, -1, Fq, T) * std[:, None] + mean[:, None]
        xt = xt.view(B, S, -1, L) * stdt[:, None] + meant[:, None]
        return x, xt

core = Core(m).eval()
mix = torch.randn(1, 2, L) * 0.1
z = m._spec(mix); mag = m._magnitude(z)
print("mag", tuple(mag.shape))
with torch.no_grad():
    ref = m(mix)
    xs, xt = core(mix, mag)
    zout = m._mask(z, xs); y = m._ispec(zout, L) + xt
    print("wrapper vs model max err", (y - ref).abs().max().item())
out = sys.argv[1] if len(sys.argv) > 1 else "htdemucs.onnx"
torch.onnx.export(core, (mix, mag), out, input_names=["mix", "mag"], output_names=["spec", "wave"],
                  opset_version=17, dynamo=False)
print("exported", out)
