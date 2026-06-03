"""Equivalence checks for the deploy pipeline.
- --policy_onnx : Tier-1 (python) — feed recorded actor_obs through policy.onnx, compare action.
- --cpp_obs     : Tier-2 — compare a C++-assembled obs dump (.npy, shape (N,435)) vs recorded actor_obs.
- --cpp_action  : Tier-1 — compare a C++ action dump (.npy, shape (N,23)) vs recorded action.
"""
import argparse, numpy as np

def maxdiff(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a - b)))

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--policy_onnx")
ap.add_argument("--cpp_obs")
ap.add_argument("--cpp_action")
ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args()
ref = np.load(a.ref, allow_pickle=True)

if a.policy_onnx:
    import onnxruntime as ort
    sess = ort.InferenceSession(a.policy_onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    obs = ref["actor_obs"].astype(np.float32)
    acts = np.stack([sess.run(None, {iname: obs[i:i+1]})[0][0] for i in range(obs.shape[0])])
    d = maxdiff(acts, ref["action"])
    print(f"[Tier-1/py] policy.onnx({iname}) action vs recorded: max|Δ|={d:.3e}  {'PASS' if d < a.tol else 'FAIL'}")

if a.cpp_obs:
    d = maxdiff(np.load(a.cpp_obs), ref["actor_obs"])
    print(f"[Tier-2] C++ obs vs recorded actor_obs: max|Δ|={d:.3e}  {'PASS' if d < a.tol else 'FAIL'}")

if a.cpp_action:
    d = maxdiff(np.load(a.cpp_action), ref["action"])
    print(f"[Tier-1] C++ action vs recorded action: max|Δ|={d:.3e}  {'PASS' if d < a.tol else 'FAIL'}")
