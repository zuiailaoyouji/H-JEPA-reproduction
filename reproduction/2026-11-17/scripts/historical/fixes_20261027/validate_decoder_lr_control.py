"""Held-out FourRoom decoder check on frozen legacy world-model features.

This is a small repair-validation experiment, not the paper's 60-epoch result.
All new output is isolated; no old decoder/model is overwritten.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time

ROOT=Path('/home/songxw/H-JEPA');OUT=ROOT/'local_env/fixes_20261027/decoder_lr_control';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'h_jepa'))
import h5py
import hdf5plugin
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from models.module import CLSDecoder
from final_probing_decoding_eval import _dense_encode_no_stride,_optimizer
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

torch.set_num_threads(4)
torch.set_float32_matmul_precision('high')
torch.manual_seed(42)
run=Path('/home/songxw/hjepa-data/ckpts/fourroom/fourroom_hjepa_l2/seed42')
model=torch.load(run/'fourroom_hjepa_l2_object.ckpt',map_location='cpu',weights_only=False).cuda().eval()
for p in model.parameters():p.requires_grad=False
cfg=OmegaConf.load(ROOT/'h_jepa/config/probing/fourroom.yaml')
cfg.optimizer.decoder_lr=0.001
mean=torch.tensor([.485,.456,.406],device='cuda').view(1,3,1,1)
std=torch.tensor([.229,.224,.225],device='cuda').view(1,3,1,1)


def data(name,count,seed):
    rng=np.random.default_rng(seed)
    with h5py.File(Path('/home/songxw/hjepa-data')/name,'r') as f:
        indices=np.sort(rng.choice(len(f['pixels']),size=count,replace=False))
        pixels=torch.from_numpy(np.stack([f['pixels'][int(i)] for i in indices])).permute(0,3,1,2)
    pixels=pixels.cuda().float().div(255).sub(mean).div(std)
    encoded={1:[],2:[]}
    with torch.no_grad():
        for chunk in pixels.split(32):
            e=_dense_encode_no_stride(model,{'pixels_level1':chunk.unsqueeze(1)})
            for level in encoded:encoded[level].append(e[f'embed_{level}'][:,0].float())
    return pixels,{k:torch.cat(v) for k,v in encoded.items()}


train_x,train_z=data('fourroom_tp35_d1_probing.h5',1024,42)
eval_x,eval_z=data('fourroom_tp35_d1_probing_val.h5',128,43)


def score(decoders):
    out={}
    for level in [1,2]:
        module=decoders[str(level)].eval()
        total=0.
        with torch.no_grad():
            for start in range(0,len(eval_x),32):
                pred=module(eval_z[level][start:start+32])
                total+=float(F.mse_loss(pred,eval_x[start:start+32]))*len(pred)
        out[f'level{level}_heldout_mse']=total/len(eval_x)
    return out


old=nn.ModuleDict()
for level in [1,2]:
    d=torch.load(run/f'final_probing_decoding_eval/decoder_level{level}.ckpt',map_location='cpu',weights_only=False)
    module=CLSDecoder(cls_dim=d['cls_dim'],img_size=d['img_size'],patch_size=d['patch_size'],**d['decoder_kwargs'])
    module.load_state_dict(d['decoder_state_dict']);old[str(level)]=module.cuda()
old_scores=score(old)
print('feature_stats', {k:{'mean':float(v.mean()), 'std':float(v.std()), 'sample_std':float(v.std(dim=0).mean())} for k,v in train_z.items()}, flush=True)
with torch.no_grad():old_preview=old['1'](eval_z[1][:8]).cpu()
del old
decoders=nn.ModuleDict({str(level):CLSDecoder(cls_dim=192,img_size=224,patch_size=14,**dict(cfg.decoder)).cuda() for level in [1,2]})
opt=_optimizer(SimpleNamespace(probers=nn.ModuleDict(),decoders=decoders),cfg)
history=[{'step':0,**score(decoders)}]
print(json.dumps({'legacy':old_scores,'fresh_decoder':history[0]}),flush=True)
start=time.monotonic()
for step in range(1,1001):
    decoders.train()
    idx=torch.randperm(len(train_x),device='cuda')[:128]
    opt.zero_grad(set_to_none=True)
    loss=sum(F.mse_loss(decoders[str(level)](train_z[level][idx]),train_x[idx]) for level in [1,2])
    assert torch.isfinite(loss)
    loss.backward();opt.step()
    if step%250==0:
        row={'step':step,'training_batch_loss':float(loss.detach()),'query_grad_l1':float(decoders['1'].queries.grad.norm()),'queries_std_l1':float(decoders['1'].queries.std()),**score(decoders)}
        history.append(row);print(json.dumps(row),flush=True)
        (OUT/'decoder_validation_progress.json').write_text(json.dumps(history,indent=2))

result={'experiment':'repair validation only, not full paper posthoc protocol','world_model':str(run/'fourroom_hjepa_l2_object.ckpt'),
        'train_frames':1024,'heldout_frames':128,'steps':1000,'batch_size':128,'decoder_lr':.001,
        'old_60epoch_decoder':old_scores,'new_decoder':history[-1],'seconds':time.monotonic()-start,'history':history}
(OUT/'decoder_validation_results.json').write_text(json.dumps(result,indent=2))
with torch.no_grad():preds=[decoders[str(level)](eval_z[level][:8]).cpu() for level in [1,2]]


def rgb(x):
    return (x*std.cpu()[0]+mean.cpu()[0]).clamp(0,1).permute(1,2,0).numpy()


fig,axes=plt.subplots(8,4,figsize=(8,16))
for i in range(8):
    for j,x in enumerate([eval_x[i].cpu(),old_preview[i],preds[0][i],preds[1][i]]):
        axes[i,j].imshow(rgb(x));axes[i,j].axis('off')
        if i==0:axes[i,j].set_title(['Input','Old L1','New L1 (1000 steps)','New L2 (1000 steps)'][j],fontsize=8)
fig.tight_layout();fig.savefig(OUT/'decoder_validation.png',dpi=120);plt.close(fig)
torch.save({'format':'repair_validation_only','decoder_state_dict':decoders.state_dict(),'steps':1000,'config':OmegaConf.to_container(cfg)},OUT/'decoder_validation_only.ckpt')
print(json.dumps(result),flush=True)
