"""Two fixed single-factor G48 controls with a held-out confirmation pool."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
import traceback

from common import HERE, save, stamp, digest, stats
import numpy as np
import torch
import branch_F_tortuosity as BF
import train_controls as TC

OUT = HERE/'round2_paired_v1'


def advance(model, optimizer, pool, start, end, directory, seed):
    inp,target,mask,*_=pool
    var=target[mask>0].var().item()+1e-12
    model.train()
    begun=time.perf_counter()
    for it in range(start,end):
        idx=torch.randint(0,len(inp),(12,),device='cuda')
        T=int(torch.randint(4,153,(1,)).item())
        pred=model.run_train(inp[idx],T,seg=24)
        loss=BF.masked_nmse(pred,target[idx],mask[idx],var)
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            return None, {'status':'nonfinite_training_loss','iteration':it+1,'T':T}
        loss.backward()
        try:
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        except RuntimeError as exc:
            if 'non-finite' not in str(exc):
                raise
            return None, {'status':'nonfinite_training_gradient','iteration':it+1,'T':T}
        optimizer.step()
        if (it+1)%1000==0 or it+1==end:
            meta={'seed':seed,'iteration':it+1,'end':end,'loss':loss.item(),
                  'gradient_norm_before_clip':norm.item(),'additional_seconds':time.perf_counter()-begun}
            save(directory/'progress.json',meta)
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                        'metadata':meta},directory/'last_periodic.pt')
            print(f'extended s{seed} iter={it+1} loss={loss.item():.5f}',flush=True)
    return model, {'status':'completed','iterations':end,'additional_iterations':end-start,
                   'additional_seconds':time.perf_counter()-begun}


def fixed_summary(rec, means, tails, factor):
    m,t=means.astype(np.float64)*factor,tails.astype(np.float64)*factor
    rows=[]
    for old in rec['rows']:
        r=dict(old); i=r['pool_idx']
        r['knee_mean'],r['floor_mean']=TC.knee_of(m[i])
        r['knee_tail'],r['floor_tail']=TC.knee_of(t[i])
        rows.append(r)
    eligible=[r for r in rows if r['eligible']]
    valid=[r for r in eligible if r['knee_tail'] is not None]
    k,f=TC.knee_of(m.mean(axis=0))
    return {'aggregate_knee':k,'aggregate_floor':f,
            'aggregate_c1':k/rec['pool_Dgeo_mean'] if k is not None else None,
            'n_eligible':len(eligible),'tail_attained':len(valid),
            'tail_ceiling':len(valid)/len(eligible),
            'tail_conditional_ratios':{d:stats([r['knee_tail']/max(1.,r[d]) for r in valid])
                                       for d in ('dgeo','dfree','deuc')},
            'rows':rows},t


def deploy(summary,tails,c,flat,subset):
    rows=[r for r in summary['rows'] if r['eligible'] and (subset=='all' or r['pool_idx']%2==1)]
    arms={}
    for arm in ('flat','geo'):
        bs=[flat if arm=='flat' else max(1,math.ceil(c*max(1.,r['dgeo']))) for r in rows]
        actual=sum(b<=152 and np.isfinite(tails[r['pool_idx'],b-1]) and tails[r['pool_idx'],b-1]<=.05
                   for r,b in zip(rows,bs))
        served=sum(r['knee_tail'] is not None and r['knee_tail']<=b for r,b in zip(rows,bs))
        arms[arm]={'mean_steps':st.mean(bs),'actual_tail_quality_coverage':actual/len(rows),
                   'knee_service_coverage':served/len(rows),'beyond_horizon':sum(b>152 for b in bs)}
    arms['n']=len(rows)
    arms['saving_fraction']=1-arms['geo']['mean_steps']/arms['flat']['mean_steps']
    arms['actual_coverage_difference_pp']=100*(arms['geo']['actual_tail_quality_coverage']-arms['flat']['actual_tail_quality_coverage'])
    arms['calibration']='original pool even indices only; .95 attained-scene quantile, not unconditional guarantee'
    arms['flat_T']=flat; arms['geo_c']=c
    return arms


def evaluate_unit(model,pools,seed,unit):
    refvar=pools['original'][1][pools['original'][2]>0].var().item()+1e-12
    out={}; c=flat=None
    for name,pool in pools.items():
        native,means,tails=TC.evaluate(model,pool,48,seed,chunk=32)
        summary,fixed_tail=fixed_summary(native,means,tails,native['target_variance']/refvar)
        if name=='original':
            policy=native['heldout_policy_tail']
            if policy['status']=='ok':
                c,flat=policy['geo_c'],policy['flat_T']
        if c is not None:
            summary['frozen_original_calibration']=deploy(summary,fixed_tail,c,flat,'odd' if name=='original' else 'all')
        else:
            summary['frozen_original_calibration']={'status':'cannot_calibrate'}
        summary['normalization_variance']=refvar
        summary['native_normalization']={k:native[k] for k in ('aggregate_knee','aggregate_floor','per_metric','n_nonfinite_scenes')}
        summary['input_sha256']=hashlib.sha256(pool[0].cpu().numpy().tobytes()).hexdigest()
        save(unit/(name+'_result.json'),summary)
        np.savez_compressed(unit/(name+'_curves.npz'),mean=means,tail=tails)
        out[name]={k:v for k,v in summary.items() if k!='rows'}
        print(f'eval s{seed} {name}: knee={summary["aggregate_knee"]} tail={summary["tail_attained"]}/{summary["n_eligible"]}',flush=True)
    return out


def run():
    decision=json.loads((HERE/'round2_decision.json').read_text())
    assert decision['selected_arms']==['width','duration']
    assert json.loads((HERE/'round2_diagnosis_v1'/'status.json').read_text())['status']=='completed'
    OUT.mkdir(exist_ok=False)
    state={'status':'running','started_utc':stamp(),'pid':os.getpid(),'completed':[],
           'decision_sha256':digest(HERE/'round2_decision.json'), 'script_sha256':digest(__file__)}
    save(OUT/'status.json',state)
    protected=json.loads((HERE/'suite_v2_status.json').read_text())['protected_files']
    try:
        for p,h in protected.items():
            assert digest(p)==h,p
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=True
        TC.CAP=152
        trainpool=BF.build_pool_maze(512,48,48,'cuda',torch.Generator(device='cuda').manual_seed(0))
        pools={name:BF.build_pool_maze(256,48,48,'cuda',torch.Generator(device='cuda').manual_seed(sd))
               for name,sd in [('original',1),('diagnostic',20260913),('confirmation',20260914)]}
        result={'decision':decision,'per_unit':[],'reference_variance_pool':'original',
                'confirmation_pool_seed':20260914,'seeds':[0,1,2]}
        for arm in ('baseline','width','duration'):
            for seed in (0,1,2):
                tag=f'{arm}_s{seed}'; state['active_task']=tag;save(OUT/'status.json',state)
                unit=OUT/tag;unit.mkdir()
                print('START '+tag,flush=True)
                if arm=='baseline':
                    model=BF.NCA().cuda()
                    path=HERE/'g48_v1'/f's{seed}'/'final.pt'
                    model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
                    training={'status':'archived_baseline','checkpoint_sha256':digest(path)}
                elif arm=='width':
                    model,training=TC.train(seed,24,192,trainpool,20000,unit)
                    if model is not None:
                        old=json.loads((HERE/'g48_v1'/f's{seed}'/'progress.json').read_text())
                        assert training['first_32_sampling_sha256']==old['first_32_sampling_sha256']
                else:
                    model=BF.NCA().cuda()
                    path=HERE/'g48_v1'/f's{seed}'/'last_periodic.pt'
                    checkpoint=torch.load(path,map_location='cuda',weights_only=True)
                    assert checkpoint['metadata']['iteration']==20000
                    model.load_state_dict(checkpoint['model'])
                    final=torch.load(HERE/'g48_v1'/f's{seed}'/'final.pt',map_location='cuda',weights_only=True)
                    assert all(torch.equal(final[k],v) for k,v in model.state_dict().items())
                    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    assert all(g['lr']==.001 for g in optimizer.param_groups)
                    torch.set_rng_state(checkpoint['cpu_rng'].cpu())
                    torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
                    model,training=advance(model,optimizer,trainpool,20000,40000,unit,seed)
                    training['source_checkpoint_sha256']=digest(path)
                record={'arm':arm,'seed':seed,'training':training}
                if model is not None:
                    if arm!='baseline':
                        torch.save(model.state_dict(),unit/'final.pt')
                    record['parameters']=sum(p.numel() for p in model.parameters())
                    record['evaluation']=evaluate_unit(model,pools,seed,unit)
                    del model
                result['per_unit'].append(record)
                save(unit/'result.json',record);save(OUT/'results.partial.json',result)
                state['completed'].append(tag);save(OUT/'status.json',state)
                gc.collect();torch.cuda.empty_cache()
        for p,h in protected.items():
            assert digest(p)==h,p
        result['protected_files_unchanged']=True
        save(OUT/'results.json',result)
        state['status']='completed';state['active_task']=None
    except BaseException as exc:
        state['status']='failed';state['error']=str(exc);traceback.print_exc()
    state['finished_utc']=stamp();save(OUT/'status.json',state)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--detach',action='store_true');a=ap.parse_args()
    if a.detach:
        with (HERE/'round2_paired_v1.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],cwd=HERE,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':p.pid}))
    else:
        run()
