"""Regression coverage for the three reproduction-protocol repairs."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_runtime import (accumulation_factor, optimizer_step_budget,
                              create_training_scheduler, validate_run_directory)
from utils import load_training_config_for_checkpoint
from models.jepa import ProjectedEncoder
from loss import SIGReg


def trainer(batches=100, epochs=4, accum=4, max_steps=-1):
    return SimpleNamespace(estimated_stepping_batches=batches * epochs,
        num_training_batches=batches, max_epochs=epochs, max_steps=max_steps,
        accumulate_grad_batches=1, accumulate_grad_batches_=accum)


@pytest.mark.parametrize('accum,steps', [(1,400),(4,100)])
def test_schedule_uses_updates_and_reaches_zero(accum, steps):
    t = trainer(accum=accum)
    p = nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD([p], lr=5e-5)
    sched = create_training_scheduler(opt, SimpleNamespace(trainer=t))
    assert sched.max_steps == steps
    for _ in range(steps):
        opt.step()
        sched.step()
    assert sched.last_epoch == steps
    assert opt.param_groups[0]['lr'] == pytest.approx(0., abs=1e-15)


def test_step_limits_and_partial_window():
    assert optimizer_step_budget(trainer(max_steps=17)) == 17
    with pytest.raises(ValueError, match='not divisible'):
        optimizer_step_budget(trainer(batches=101))


def test_one_update_schedule():
    p = nn.Parameter(torch.ones(1))
    opt = torch.optim.SGD([p], lr=.1)
    sched = create_training_scheduler(opt, SimpleNamespace(trainer=trainer(batches=1,epochs=1,accum=1)))
    opt.step(); sched.step()
    assert opt.param_groups[0]['lr'] == 0


class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3*4*4, 12)

    def forward(self, x, **kwargs):
        return self.fc(x.flatten(1)).tanh()


def test_chunk_checkpoint_preserves_full_batch_bn_loss_and_gradients():
    torch.manual_seed(11)
    plain = ProjectedEncoder(ImageBackbone(), nn.Sequential(nn.Linear(12,16), nn.BatchNorm1d(16), nn.Linear(16,8)))
    fixed = copy.deepcopy(plain)
    fixed.checkpoint_encoder = True
    fixed.encoder_frame_chunk_size = 7
    x = torch.randn(128,3,4,4)
    reg = SIGReg(num_proj=16)
    results = []
    for model in [plain, fixed]:
        y = model(x)
        torch.manual_seed(5)
        loss = y.square().mean() + reg(y.unsqueeze(0))
        loss.backward()
        results.append((y.detach(),loss.detach()))
    torch.testing.assert_close(results[0][0], results[1][0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(results[0][1], results[1][1], atol=2e-6, rtol=2e-5)
    for a,b in zip(plain.parameters(),fixed.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)
    assert plain.projector[1].num_batches_tracked == fixed.projector[1].num_batches_tracked == 1
    torch.testing.assert_close(plain.projector[1].running_mean,fixed.projector[1].running_mean)


def test_posthoc_learning_rates():
    from final_probing_decoding_eval import _optimizer
    heads = SimpleNamespace(probers=nn.Linear(2,2),decoders=nn.Linear(2,2))
    for env in ['ant','fourroom','pusht','cube']:
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1]/f'config/probing/{env}.yaml')
        opt = _optimizer(heads,cfg)
        rates = {g['name']:g['lr'] for g in opt.param_groups}
        assert rates['probes']==.001
        assert rates['decoders']==(.001 if env=='ant' else .0001)


def test_existing_legacy_directory_is_not_overwritten(tmp_path):
    old = OmegaConf.create({'loader':{'batch_size':32},'trainer':{'accumulate_grad_batches':4}})
    OmegaConf.save(old,tmp_path/'config.yaml')
    before = (tmp_path/'config.yaml').read_bytes()
    new = OmegaConf.create({'training_protocol_version':2,'loader':{'batch_size':128},'trainer':{'accumulate_grad_batches':1}})
    with pytest.raises(ValueError,match='different training protocol'):
        validate_run_directory(tmp_path,new)
    assert (tmp_path/'config.yaml').read_bytes()==before


def test_accumulation_loss_is_scaled_once(monkeypatch):
    import main_hjepa
    value=nn.Parameter(torch.tensor(3.))
    monkeypatch.setattr(main_hjepa,'hjepa_forward',lambda *a,**k:{'loss':value.square()})
    module=SimpleNamespace(trainer=trainer(accum=4))
    result=main_hjepa._hjepa_training_forward(module,{},'fit',None)
    result['loss'].backward()
    assert value.grad.item()==pytest.approx(1.5)
    assert main_hjepa._hjepa_training_forward(module,{},'validate',None)['loss'].item()==9.


def test_release_checkpoint_infers_training_config(tmp_path):
    checkpoint = tmp_path / "ant" / "ant_hjepa_l2" / "seed43" / "ant_hjepa_l2_object.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    cfg = load_training_config_for_checkpoint(checkpoint)
    assert cfg.output_model_name == "ant_hjepa_l2"
    assert cfg.num_levels == 2
    assert cfg.seed == 43


def test_release_config_inference_inside_initialized_hydra(tmp_path):
    from hydra import initialize_config_dir

    checkpoint = tmp_path / "cube" / "cube_hjepa_l3" / "seed44" / "cube_hjepa_l3_object.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    probing_dir = Path(__file__).resolve().parents[1] / "config/probing"
    with initialize_config_dir(config_dir=str(probing_dir), version_base=None):
        cfg = load_training_config_for_checkpoint(checkpoint)
    assert cfg.output_model_name == "cube_hjepa_l3"
    assert cfg.num_levels == 3
    assert cfg.seed == 44


def test_droid_planner_matches_paper_table_8():
    eval_dir = Path(__file__).resolve().parents[1] / "config/eval"
    flat = OmegaConf.load(eval_dir / "droid_flat.yaml")
    hierarchical = OmegaConf.load(eval_dir / "droid_l2.yaml")

    assert flat.eval_budget == 36
    assert flat.plan_config.horizon == 36
    assert flat.plan_config.receding_horizon == 36
    assert flat.solver.num_samples == 32
    assert flat.solver.optimizer_kwargs.lr == pytest.approx(0.01)

    assert hierarchical.eval_budget == 36
    assert hierarchical.hierarchical_plan_config.receding_horizon == 36
    assert hierarchical.hierarchical_plan_config.level1.horizon == 36
    assert hierarchical.hierarchical_plan_config.level2.horizon == 12
    assert hierarchical.hierarchical_plan_config.level2.num_subgoals == 12
    assert hierarchical.hierarchical_plan_config.level1.intermediate_cost_weight == pytest.approx(0.5)
    for level, learning_rate in (("level1", 0.01), ("level2", 0.3)):
        solver = hierarchical.solver.solvers[level]
        assert solver.num_samples == 16
        assert solver.n_steps == 90
        assert solver.optimizer_kwargs.lr == pytest.approx(learning_rate)


def test_simulation_planners_match_paper_table_7():
    eval_dir = Path(__file__).resolve().parents[1] / "config/eval"
    expected = {
        "fourroom": {
            "budget": 210, "flat_lr": 0.1, "flat_h": 23, "flat_s": 200, "l1_lr": 0.08, "upper_lr": 0.1,
            "h": {2: [2, 12], 3: [2, 2, 6], 4: [2, 2, 2, 3]},
            "s": {2: [100, 800], 3: [50, 200, 200], 4: [50, 200, 200, 400]},
        },
        "ant": {
            "budget": 350, "flat_lr": 0.3, "flat_h": 23, "flat_s": 100, "l1_lr": 0.06, "upper_lr": 0.1,
            "h": {2: [4, 9], 3: [4, 2, 5], 4: [3, 2, 2, 3]},
            "s": {2: [25, 200], 3: [100, 100, 400], 4: [50, 200, 25, 50]},
        },
        "pusht": {
            "budget": 90, "flat_lr": 0.05, "flat_h": 15, "flat_s": 800, "l1_lr": 0.1, "upper_lr": 0.075,
            "h": {2: [2, 8], 3: [2, 2, 4], 4: [2, 2, 2, 2]},
            "s": {2: [50, 400], 3: [25, 50, 200], 4: [200, 800, 800, 800]},
        },
        "cube": {
            "budget": 30, "flat_lr": 0.05, "flat_h": 4, "flat_s": 800, "l1_lr": 0.1, "upper_lr": 0.075,
            "h": {2: [2, 2], 3: [2, 2, 1], 4: [2, 2, 1, 1]},
            "s": {2: [400, 100], 3: [25, 800, 800], 4: [200, 50, 800, 800]},
        },
    }
    for env, values in expected.items():
        flat = OmegaConf.load(eval_dir / f"{env}_flat.yaml")
        assert flat.eval.num_eval == 50
        assert flat.eval.eval_budget == values["budget"]
        assert flat.plan_config.horizon == values["flat_h"]
        assert flat.solver.num_samples == values["flat_s"]
        assert flat.solver.n_steps == 90
        assert flat.solver.optimizer_kwargs.lr == pytest.approx(values["flat_lr"])
        for depth in (2, 3, 4):
            cfg = OmegaConf.load(eval_dir / f"{env}_l{depth}.yaml")
            assert cfg.eval.num_eval == 50
            assert cfg.eval.eval_budget == values["budget"]
            for level in range(1, depth + 1):
                solver = cfg.solver.solvers[f"level{level}"]
                plan = cfg.hierarchical_plan_config[f"level{level}"]
                assert solver.n_steps == 90
                assert solver.num_samples == values["s"][depth][level - 1]
                assert plan.horizon == values["h"][depth][level - 1]
                expected_lr = values["l1_lr"] if level == 1 else values["upper_lr"]
                assert solver.optimizer_kwargs.lr == pytest.approx(expected_lr)
