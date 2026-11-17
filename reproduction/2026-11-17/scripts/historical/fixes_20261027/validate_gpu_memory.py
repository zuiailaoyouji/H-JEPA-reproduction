"""Two real full-batch optimizer steps per architecture; no production run writes."""
import gc
import json
from pathlib import Path
import sys
import time
import argparse

ROOT=Path('/home/songxw/H-JEPA')
sys.path.insert(0,str(ROOT/'h_jepa'))
import torch
from torch import nn
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from hjepa_utils import create_world_model,hjepa_forward
from main_hjepa import _normalize_uint8_images_on_device,_configure_runtime_performance
from training_runtime import configure_encoder_memory


class Context(nn.Module):
    def __init__(self,model,losses):
        super().__init__();self.model=model
        for key,value in losses.items():self.add_module(key,value)
    def log_dict(self,*args,**kwargs):pass


def run(names):
    _configure_runtime_performance()
    torch.set_num_threads(4)
    if not OmegaConf.has_resolver('eval'):
        OmegaConf.register_new_resolver('eval',eval)
    results=[]
    for name in names:
        with initialize_config_dir(config_dir=str(ROOT/'h_jepa/config/train'),version_base=None):
            cfg=compose(config_name=name)
        model,losses=create_world_model(cfg)
        configure_encoder_memory(model,cfg)
        context=Context(model,losses).cuda().train()
        opt=torch.optim.AdamW(context.model.parameters(),lr=5e-5)
        frames=4
        for level in range(int(cfg.num_levels),1,-1):
            frames=(frames-1)*int(cfg[f'level{level}'].stride)+int(cfg[f'level{level}'].window_size)
        b=int(cfg.loader.batch_size)
        adim=int(cfg.level1.wm.action_dim)*int(cfg.data.dataset.level1.frameskip)
        sigreg_batches=[]
        hooks=[]
        for key,loss in losses.items():
            if key.startswith('sigreg'):
                hooks.append(loss.register_forward_pre_hook(lambda m,args:sigreg_batches.append(int(args[0].shape[-2]))))
        torch.cuda.reset_peak_memory_stats()
        samples=[]
        for step in range(2):
            batch={'pixels_level1':torch.randint(256,(b,frames,3,int(cfg.img_size),int(cfg.img_size)),dtype=torch.uint8,device='cuda'),
                   'action_level1':torch.randn(b,frames,adim,device='cuda')}
            if bool(cfg.level1.wm.use_proprio):
                batch['proprio_level1']=torch.randn(b,frames,27,device='cuda')
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize();t=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=hjepa_forward(context,batch,'fit',cfg,normalize_batch=_normalize_uint8_images_on_device)
                loss=out['loss']
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in context.model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(context.model.parameters(),1.)
            opt.step();torch.cuda.synchronize()
            samples.append({'loss':float(loss),'seconds':time.monotonic()-t})
            del batch,out,loss
        assert sigreg_batches and set(sigreg_batches)=={128},sigreg_batches
        row={'config':name,'batch':b,'frames':frames,'sigreg_batch_sizes':sorted(set(sigreg_batches)),
             'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
             'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'steps':samples}
        print(json.dumps(row),flush=True);results.append(row)
        (ROOT/'local_env/fixes_20261027/gpu_memory_results.json').write_text(json.dumps(results,indent=2))
        for hook in hooks:hook.remove()
        del context,model,losses,opt
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('configs',nargs='*',default=['fourroom_hjepa_l2','pusht_hjepa_l3','cube_hjepa_l3','ant_hjepa_l2','cube_hjepa_l4'])
    run(p.parse_args().configs)
