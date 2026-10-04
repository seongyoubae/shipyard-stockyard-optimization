"""Evaluation shared by the public trainer and original research training loop."""
from time import perf_counter
import numpy as np
import torch

def rollout(model, env, device):
    model.eval();state=env.reset(shuffle_schedule=False)
    reward_sum=0.;info={};started=perf_counter()
    for _ in range(env.max_steps):
        src,dst=env.get_masks()
        with torch.no_grad():
            actions,_,_,_=model.act_batch(state[None].to(device),src[None].to(device),dst[None].to(device),greedy=True)
        state,reward,done,info=env.step(tuple(actions[0].cpu().tolist()));reward_sum+=reward
        if done: break
    if device.type=='cuda': torch.cuda.synchronize()
    return reward_sum,info.get('final_blocking_metric',float('inf')),env.crane_move,perf_counter()-started

def evaluate_policy(model, env, device, verbose=False, return_blocked=False):
    if isinstance(env,(tuple,list)):
        values=[rollout(model,e,device) for e in env]
        return tuple(float(np.mean([r[i] for r in values])) for i in range(3))
    return rollout(model,env,device)
