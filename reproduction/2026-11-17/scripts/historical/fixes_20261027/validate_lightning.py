"""Exercise the installed Lightning/SPT manual loop with real accumulation."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/home/songxw/H-JEPA/h_jepa')
import torch
import lightning as pl
import stable_pretraining as spt
from torch.utils.data import DataLoader,Dataset
from omegaconf import OmegaConf
from main_hjepa import GradClipModule
from training_runtime import accumulation_factor,create_training_scheduler

ROOT=Path('/home/songxw/H-JEPA/local_env/fixes_20261027')
torch.set_num_threads(2)

class Data(Dataset):
    def __len__(self):return 32
    def __getitem__(self,i):return {'x':torch.tensor([float(i%4)/4])}

def forward(self,batch,stage):
    return {'loss':self.model(batch['x']).square().mean()/accumulation_factor(self.trainer)}

results=[]
for accum in [1,4]:
    spt.set(cache_dir=str(ROOT/f'spt_loop_{accum}'))
    pl.seed_everything(42)
    def schedule(opt,module):return create_training_scheduler(opt,module,.1)
    module=GradClipModule(model=torch.nn.Linear(1,1),forward=forward,
        optim={'model_opt':{'modules':'model','optimizer':{'type':'SGD','lr':.1},'scheduler':schedule,'interval':'step'}})
    module._train_cfg=OmegaConf.create({'training_protocol_version':2,'loader':{'batch_size':1},'trainer':{'accumulate_grad_batches':accum}})
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=2,accumulate_grad_batches=accum,
                       logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False)
    trainer.fit(module,DataLoader(Data(),batch_size=1))
    sched=module.lr_schedulers()
    expected=64//accum
    assert trainer.global_step==sched.last_epoch==sched.max_steps==expected
    assert sched.get_last_lr()==[0.]
    checkpoint=ROOT/f'loop_accum{accum}.ckpt'
    trainer.save_checkpoint(checkpoint)
    d=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert d['hjepa_training_protocol']['accumulate_grad_batches']==accum
    results.append({'accumulation':accum,'optimizer_updates':trainer.global_step,
                    'scheduler_steps':sched.last_epoch,'scheduler_budget':sched.max_steps,'last_lr':sched.get_last_lr()})
print(json.dumps(results,indent=2))
(ROOT/'lightning_results.json').write_text(json.dumps(results,indent=2))
