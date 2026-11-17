#!/usr/bin/env python3
"""Fixed-projection SIGReg sharding/value/input-gradient check using CPU Gloo."""

from __future__ import annotations

import json
import csv
import hashlib
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path("/home/songxw/H-JEPA")
OUT = ROOT / "local_env/analysis_20261027/sigreg_sharding.json"
N, D, M, K = 256, 32, 64, 17
SEED_DATA = 20261027
SEED_PROJ = 261027


def fixed_loss(z: torch.Tensor, raw_projection: torch.Tensor) -> torch.Tensor:
    """Call the repository SIGReg with a deterministic Gaussian projection matrix."""
    from loss import SIGReg

    original_randn = torch.randn

    def deterministic_randn(*size, **kwargs):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
        if shape == (D, M):
            return raw_projection.to(device=kwargs.get("device", z.device), dtype=torch.get_default_dtype()).clone()
        return original_randn(*size, **kwargs)

    torch.randn = deterministic_randn
    try:
        return SIGReg(knots=K, num_proj=M).to(dtype=z.dtype)(z)
    finally:
        torch.randn = original_randn


def generate_fixed_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.set_default_dtype(torch.float64)
    data_gen = torch.Generator(device="cpu").manual_seed(SEED_DATA)
    proj_gen = torch.Generator(device="cpu").manual_seed(SEED_PROJ)
    z = torch.randn((1, N, D), generator=data_gen, dtype=torch.float64)
    raw_projection = torch.randn((D, M), generator=proj_gen, dtype=torch.float64)
    return z, raw_projection


def make_encoder() -> torch.nn.Linear:
    encoder = torch.nn.Linear(D, D, bias=False, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(D, dtype=torch.float64))
    return encoder


def single_reference(z: torch.Tensor, raw_projection: torch.Tensor) -> dict:
    encoder = make_encoder()
    inputs = z.detach().clone().requires_grad_(True)
    loss = fixed_loss(encoder(inputs), raw_projection)
    loss.backward()
    return {
        "tep": float(loss.detach()),
        "weight_grad_norm": float(encoder.weight.grad.norm()),
        "weight_grad": encoder.weight.grad.detach().clone(),
        "input_grad": inputs.grad.detach().clone(),
        "input_grad_norm": float(inputs.grad.norm()),
    }


def local_shard_reference(z: torch.Tensor, raw_projection: torch.Tensor, world_size: int) -> dict:
    shard_size = N // world_size
    losses = []
    gradients = []
    for rank in range(world_size):
        encoder = make_encoder()
        local = z[:, rank * shard_size : (rank + 1) * shard_size]
        loss = fixed_loss(encoder(local), raw_projection)
        loss.backward()
        losses.append(float(loss.detach()))
        gradients.append(encoder.weight.grad.detach().clone())
    mean_gradient = torch.stack(gradients).mean(0)
    return {
        "mean_local_tep": sum(losses) / len(losses),
        "ddp_averaged_weight_grad_norm": float(mean_gradient.norm()),
        "local_teps": losses,
        "weight_grad": mean_gradient,
    }


def worker(rank: int, world_size: int, init_method: str, z: torch.Tensor, raw_projection: torch.Tensor, output_dir: str) -> None:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    try:
        shard_size = N // world_size
        local = z[:, rank * shard_size : (rank + 1) * shard_size].detach().clone().requires_grad_(True)
        encoder = make_encoder()
        loss = fixed_loss(encoder(local), raw_projection)
        loss.backward()
        local_grad_norm = float(encoder.weight.grad.norm())
        # Match DDP's averaged parameter-gradient reduction after the local backward.
        dist.all_reduce(encoder.weight.grad, op=dist.ReduceOp.SUM)
        encoder.weight.grad.div_(world_size)
        # The local mean's backward scales input gradients by W. DDP averages
        # shared parameter gradients; for the input-gradient comparison undo W
        # explicitly, then concatenate the disjoint shards.
        input_shards = [torch.zeros_like(local.grad) for _ in range(world_size)]
        dist.all_gather(input_shards, local.grad / world_size)
        result = {
            "rank": rank,
            "world_size": world_size,
            "distributed_tep": float(loss.detach()),
            "aux_local_identity_encoder_parameter_gradient_norm_before_ddp_average": local_grad_norm,
            "aux_ddp_averaged_identity_encoder_parameter_gradient_norm": float(encoder.weight.grad.norm()),
            "aux_ddp_averaged_identity_encoder_parameter_gradient": encoder.weight.grad.detach().cpu().tolist(),
            "distributed_input_gradient": torch.cat(input_shards, dim=1).detach().cpu().tolist(),
        }
        if rank == 0:
            (Path(output_dir) / f"world_{world_size}.json").write_text(json.dumps(result))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_distributed(world_size: int, z: torch.Tensor, raw_projection: torch.Tensor, tmpdir: Path) -> dict:
    init_file = tmpdir / f"gloo_{world_size}.init"
    output_dir = tmpdir / "worker_results"
    output_dir.mkdir(exist_ok=True)
    mp.spawn(
        worker,
        args=(world_size, f"file://{init_file}", z, raw_projection, str(output_dir)),
        nprocs=world_size,
        join=True,
    )
    return json.loads((output_dir / f"world_{world_size}.json").read_text())


def main() -> None:
    torch.set_num_threads(1)
    z, raw_projection = generate_fixed_inputs()
    global_ref = single_reference(z, raw_projection)
    records = []
    with tempfile.TemporaryDirectory(prefix="hjepa_sigreg_gloo_") as temp:
        tmpdir = Path(temp)
        for world_size in (1, 2, 4):
            local_ref = local_shard_reference(z, raw_projection, world_size)
            distributed = run_distributed(world_size, z, raw_projection, tmpdir)
            grad = torch.tensor(
                distributed["aux_ddp_averaged_identity_encoder_parameter_gradient"],
                dtype=torch.float64,
            )
            grad_error = grad - global_ref["weight_grad"]
            input_grad = torch.tensor(distributed["distributed_input_gradient"], dtype=torch.float64)
            input_error = input_grad - global_ref["input_grad"]
            record = {
                "shards": f"{world_size} x {N // world_size}",
                "world_size": world_size,
                "local_batch": N // world_size,
                "dtype": "float64",
                "projection_directions": M,
                "quadrature_knots": K,
                "data_seed": SEED_DATA,
                "projection_seed": SEED_PROJ,
                "fixed_projection_sha256": __import__("hashlib").sha256(raw_projection.numpy().tobytes()).hexdigest(),
                "distributed_tep": distributed["distributed_tep"],
                # Table 11's gradient is with respect to the fixed Gaussian input Z.
                "distributed_input_gradient_norm": float(input_grad.norm()),
                "distributed_vs_global_tep_abs_error": abs(distributed["distributed_tep"] - global_ref["tep"]),
                "distributed_vs_global_input_gradient_max_abs_error": float(input_error.abs().max()),
                "distributed_vs_global_input_gradient_l2_error": float(input_error.norm()),
                "aux_identity_encoder_parameter_gradient_norm": distributed[
                    "aux_ddp_averaged_identity_encoder_parameter_gradient_norm"
                ],
                "aux_identity_encoder_parameter_gradient_max_abs_error": float(grad_error.abs().max()),
                "aux_identity_encoder_parameter_gradient_l2_error": float(grad_error.norm()),
                "local_reference_mean_tep_without_ecf_collective": local_ref["mean_local_tep"],
                "local_reference_identity_encoder_ddp_gradient_norm_without_ecf_collective": local_ref[
                    "ddp_averaged_weight_grad_norm"
                ],
                "local_reference_shard_teps": local_ref["local_teps"],
                "global_reference_tep": global_ref["tep"],
                "global_reference_input_gradient_norm": global_ref["input_grad_norm"],
                "global_reference_aux_identity_encoder_parameter_gradient_norm": global_ref["weight_grad_norm"],
                "invariance_pass": (
                    abs(distributed["distributed_tep"] - global_ref["tep"]) < 1e-12
                    and float(grad_error.abs().max()) < 1e-11
                    and float(input_error.abs().max()) < 1e-11
                ),
            }
            records.append(record)

    result = {
        "created_at_local": __import__("datetime").datetime.now(__import__("zoneinfo").ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "test": "repository SIGReg with fixed projections, float64 data, CPU Gloo ECF synchronization, and gradients with respect to the Gaussian input Z",
        "shape": {"n": N, "d": D, "directions": M, "time_steps": 1, "quadrature_knots": K},
        "random_seeds": {"data": SEED_DATA, "projection": SEED_PROJ},
        "torch_version": torch.__version__,
        "python_executable": sys.executable,
        "backend": "Gloo; all W=1,2,4 runs initialize actual process groups",
        "input_sha256": hashlib.sha256(z.numpy().tobytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repository_loss_sha256": hashlib.sha256((ROOT / "h_jepa/loss.py").read_bytes()).hexdigest(),
        "gradient_protocol": "Table 11 is matched with the gradient with respect to input Z. Because the repository all_reduce is not tracked by autograd, local input gradients are divided by W before gathering. The identity-encoder parameter gradient is retained only as an auxiliary DDP check.",
        "quadrature_note": "Production SIGReg creates float32 quadrature buffers; this check casts those same buffers to float64 for all shardings.",
        "autograd_warning": "PyTorch warns c10d::allreduce_ has no autograd kernel. This checks the repository's current local backward plus explicit DDP gradient averaging; future PyTorch versions must rerun this numerical verification.",
        "reference": {
            "tep": global_ref["tep"],
            "input_gradient_l2_norm": global_ref["input_grad_norm"],
            "aux_identity_encoder_parameter_gradient_l2_norm": global_ref["weight_grad_norm"],
            "note": "The paper's Table 11 gradient is compared with the input-Z gradient. Its fixed Gaussian matrix, projection directions, and seeds are unpublished, so matching absolute values is not expected.",
        },
        "paper_table11_reference_for_context": {
            "shards": ["1 x 256", "2 x 128", "4 x 64"],
            "reported_ours_tep": [1.372619178543, 1.372619178543, 1.372619178543],
            "reported_gradient_norm": [0.349154178012, 0.349154178012, 0.349154178012],
            "reported_single_gpu_tep": [1.372619, 1.180770, 1.186233],
        },
        "sharding_results": records,
        "all_distributed_shards_match_single_batch": all(record["invariance_pass"] for record in records),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with OUT.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"output": str(OUT), "pass": result["all_distributed_shards_match_single_batch"], "reference_tep": global_ref["tep"], "reference_input_grad_norm": global_ref["input_grad_norm"], "aux_identity_encoder_parameter_grad_norm": global_ref["weight_grad_norm"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "h_jepa"))
    main()
