"""Local environment validation; no datasets or checkpoints are downloaded."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("WANDB_MODE", "disabled")
rows = subprocess.check_output([
    "nvidia-smi", "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)
gpus = [dict(zip(("index", "uuid", "name", "free_mib", "utilization"),
                 map(str.strip, row.split(",")))) for row in rows.strip().splitlines()]
candidates = [g for g in gpus if int(g["utilization"]) == 0 and int(g["free_mib"]) > 8192]
assert candidates, "No idle GPU with at least 8 GiB free for the smoke test"
selected = max(candidates, key=lambda g: int(g["free_mib"]))
os.environ["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
os.environ["MUJOCO_EGL_DEVICE_ID"] = selected["index"]
print("GPU:", selected, flush=True)

import cv2
import decord
import gymnasium as gym
import numpy as np
import torch
import stable_worldmodel
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

assert torch.__version__ == "2.7.1+cu128"
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
torch.set_num_threads(4)
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")
sys.path.insert(0, str(ROOT / "h_jepa"))
from main_hjepa import _normalize_uint8_images_on_device
from hjepa_utils import create_world_model, hjepa_forward

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)
with initialize_config_dir(config_dir=str(ROOT / "h_jepa/config/train"), version_base=None):
    cfg = compose(config_name="ant_hjepa_l3")
model, losses = create_world_model(cfg)
model = model.cuda().train()
holder = SimpleNamespace(model=model, log_dict=lambda *args, **kwargs: None,
                         **{k: v.cuda() for k, v in losses.items()})
batch = {
    "pixels_level1": torch.randint(0, 256, (2, 13, 3, 64, 64), dtype=torch.uint8, device="cuda"),
    "proprio_level1": torch.randn(2, 13, 27, device="cuda"),
    "action_level1": torch.randn(2, 13, 40, device="cuda"),
}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    result = hjepa_forward(holder, batch, "train", cfg,
                           normalize_batch=_normalize_uint8_images_on_device)
loss = result["loss"]
assert torch.isfinite(loss)
loss.backward()
for level in range(1, 4):
    grads = [p.grad for p in model.get_level(level).parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), f"Bad gradients at level {level}"
optimizer.step()
torch.cuda.synchronize()
report = {
    "python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
    "visible_physical_gpus": gpus, "test_gpu": selected, "bf16": True,
    "config": "ant_hjepa_l3", "synthetic_batch_size": 2,
    "loss": float(loss.detach()),
    "model_parameters": sum(p.numel() for p in model.parameters()),
    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    "three_level_forward_backward_optimizer": "passed", "environments": {},
}
print("Three-level BF16 forward/backward/AdamW passed", flush=True)

for env_id, kwargs in [
    ("swm/PushT-v1", {"resolution": 64}),
    ("swm/FourRoomDistractors-v0", {}),
    ("swm/OGBAntMaze-v0", {"maze_type": "medium", "width": 64, "height": 64}),
    ("swm/OGBCube-v0", {"env_type": "single", "ob_type": "pixels", "width": 64, "height": 64}),
]:
    env = gym.make(env_id, render_mode="rgb_array", **kwargs)
    try:
        env.reset(seed=42)
        env.step(env.action_space.sample())
        frame = np.asarray(env.render())
        assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.size > 0, (env_id, frame.shape)
        report["environments"][env_id] = {"reset_step_render": "passed", "shape": list(frame.shape)}
        print(env_id, "reset/step/render passed", frame.shape, flush=True)
    finally:
        env.close()

video = ROOT / "local_env/smoke-video.avi"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
assert writer.isOpened()
for value in (0, 64, 128, 192):
    writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
writer.release()
reader = decord.VideoReader(str(video), ctx=decord.cpu(0), num_threads=1)
decoded = reader.get_batch([0, 1, 2, 3]).asnumpy()
assert decoded.shape == (4, 64, 64, 3)
assert np.abs(decoded.mean(axis=(1, 2, 3)) - np.array([0, 64, 128, 192])).max() < 5
report["decord_video_decode"] = "passed"
(ROOT / "local_env/validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
