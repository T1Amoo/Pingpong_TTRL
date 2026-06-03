"""Rebuild the 64-64 ball predictor from a training checkpoint and export to ONNX."""
import argparse, torch, torch.nn as nn

class MLPPredictor(nn.Module):  # mirrors rsl_rl ... _MLPPredictor
    def __init__(self, input_dim=15, hidden=(64, 64), output_dim=3):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, output_dim))
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--history_len", type=int, default=5)
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    assert "pred_state_dict" in sd, f"no pred_state_dict in {a.ckpt} (was it trained with --predictor?)"
    m = MLPPredictor(input_dim=3 * a.history_len)
    m.load_state_dict(sd["pred_state_dict"]); m.eval()
    dummy = torch.zeros(1, 3 * a.history_len)
    torch.onnx.export(m, dummy, a.out, input_names=["ball_history"], output_names=["pred"],
                      opset_version=17, dynamic_axes=None)
    import numpy as np, onnxruntime as ort
    x = torch.randn(1, 3 * a.history_len)
    t = m(x).detach().numpy()
    o = ort.InferenceSession(a.out).run(None, {"ball_history": x.numpy()})[0]
    err = float(np.max(np.abs(t - o)))
    print(f"[export_predictor] wrote {a.out}; onnx-vs-torch max|Δ|={err:.2e}")
    assert err < 1e-5, "predictor onnx mismatch"
    print("[export_predictor] OK")
