"""Training protocol safeguards and optimizer-step accounting."""
from __future__ import annotations

import math
from pathlib import Path

from omegaconf import OmegaConf


def accumulation_factor(trainer):
    # SPT keeps the real factor here after resetting Lightning's attribute to 1.
    return max(1, int(getattr(trainer, "accumulate_grad_batches_",
                              getattr(trainer, "accumulate_grad_batches", 1))))


def optimizer_step_budget(trainer):
    """Count manual optimizer updates, honoring Trainer limits.

    The upstream SPT loop does not flush a partial accumulation window at an
    epoch boundary. Reject that ambiguous case instead of miscounting updates.
    """
    estimated = trainer.estimated_stepping_batches  # initializes batch counts
    batches, epochs, max_steps = trainer.num_training_batches, trainer.max_epochs, trainer.max_steps
    accum = accumulation_factor(trainer)
    if math.isfinite(batches) and epochs is not None and epochs > 0:
        batches = int(batches)
        if batches % accum:
            raise ValueError(
                f"{batches} train batches is not divisible by accumulation={accum}; "
                "use accumulation=1 with a real batch, or an exactly divisible "
                "limit_train_batches. Partial SPT accumulation is unsupported."
            )
        steps = (batches // accum) * int(epochs)
        if max_steps is not None and max_steps > 0:
            steps = min(steps, int(max_steps))
    elif max_steps is not None and max_steps > 0:
        steps = int(max_steps)
    else:
        raise ValueError(f"A finite optimizer-step budget is required; got {estimated}.")
    if steps < 1:
        raise ValueError("The training configuration has no optimizer updates.")
    return steps


def create_training_scheduler(optimizer, module, warmup_ratio=0.01):
    from stable_pretraining.optim import create_scheduler

    total_steps = optimizer_step_budget(module.trainer)
    if not 0 <= warmup_ratio < 1:
        raise ValueError("optimizer.warmup_ratio must be in [0, 1)")
    warmup_steps = min(total_steps - 1, max(1, int(warmup_ratio * total_steps)))
    return create_scheduler(optimizer, {
        "type": "LinearWarmupCosineAnnealingLR",
        "warmup_steps": warmup_steps,
        "max_steps": total_steps,
    }, module)


def training_protocol(cfg):
    return {
        "version": int(cfg.get("training_protocol_version", 2)),
        "batch_size_per_device": int(cfg.loader.batch_size),
        "accumulate_grad_batches": int(cfg.trainer.get("accumulate_grad_batches", 1)),
        "scheduler_units": "optimizer_updates",
    }


def validate_run_directory(run_dir, cfg):
    """Never relabel an existing legacy run by overwriting its config."""
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        return
    previous = OmegaConf.load(config_path)
    old = training_protocol(previous)
    old["version"] = int(previous.get("training_protocol_version", 1))
    new = training_protocol(cfg)
    if old != new:
        raise ValueError(
            f"Refusing to overwrite/resume a different training protocol in {run_dir}. "
            f"Existing={old}; requested={new}. Use a new subdir, e.g. "
            "subdir=<env>/<model>/seed42_paper_v2. Existing weights are unchanged."
        )


def configure_encoder_memory(model, cfg):
    """Checkpoint only per-image backbones; BN heads stay outside recomputation."""
    from models.jepa import ProjectedEncoder
    from torch import nn

    memory = cfg.get("memory", {})
    enabled = bool(memory.get("encoder_checkpointing", False))
    chunk = int(memory.get("encoder_frame_chunk_size", 128))
    if chunk < 1:
        raise ValueError("encoder_frame_chunk_size must be positive")
    count = 0
    for module in model.modules():
        if not isinstance(module, ProjectedEncoder):
            continue
        if not hasattr(module.encoder, "config") or getattr(module.encoder.config, "model_type", None) != "vit":
            continue
        if enabled and any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                           for m in module.encoder.modules()):
            raise ValueError("Cannot checkpoint/chunk an image backbone containing BatchNorm")
        module.checkpoint_encoder = enabled
        module.encoder_frame_chunk_size = chunk
        count += 1
    return count
