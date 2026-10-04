"""Small CPU-friendly PPO runner using the uploaded yard and priority GRU network."""
import argparse
import csv
from pathlib import Path
import random
import torch
import numpy as np
from stockyard.data.sample import generate,to_plates
from stockyard.environment.yard import Locating
from stockyard.models.network import SteelPlateConditionalMLPModel

def make_model(env, embed_dim=32):
    return SteelPlateConditionalMLPModel(embed_dim=embed_dim,num_actor_layers=2,num_critic_layers=2,
        actor_init_std=.01,critic_init_std=1.,pile_feature_dim=env.actual_pile_feature_dim,
        num_heads=4,encoder_type='pi_gru_add')

def advantages(rewards,values,dones,last_value,gamma=.995,lam=.95):
    out=[];gae=0.
    for i in reversed(range(len(rewards))):
        nxt=last_value if i==len(rewards)-1 else values[i+1]
        mask=1.-float(dones[i]);delta=rewards[i]+gamma*nxt*mask-values[i]
        gae=delta+gamma*lam*mask*gae;out.append(gae)
    return torch.tensor(out[::-1],dtype=torch.float32)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--updates',type=int,default=20)
    parser.add_argument('--horizon',type=int,default=64);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',default='outputs');args=parser.parse_args()
    if args.updates<1 or args.horizon<2: parser.error('updates >= 1 and horizon >= 2 required')
    torch.set_num_threads(1);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    env=Locating(inbound_plates=to_plates(generate(args.seed)),max_stack=12,min_obstacles=0,max_obstacles=0,max_steps=200)
    model=make_model(env);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    state=env.reset();episode_reward=0.;completed=[]
    with (out/'training.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['update','loss','mean_episode_reward','completed_episodes'])
        for update in range(args.updates):
            states=[];sources=[];destinations=[];actions=[];logs=[];values=[];rewards=[];dones=[]
            for _ in range(args.horizon):
                src,dst=env.get_masks()
                with torch.no_grad(): act,log,value,_=model.act_batch(state[None],src[None],dst[None])
                states.append(state);sources.append(src);destinations.append(dst)
                actions.append(act[0]);logs.append(log[0]);values.append(value.item())
                state,reward,done,info=env.step(tuple(act[0].tolist()))
                rewards.append(reward);dones.append(done);episode_reward+=reward
                if done:
                    completed.append(episode_reward);episode_reward=0.
                    env=Locating(inbound_plates=to_plates(generate(args.seed+update+len(completed))),max_stack=12,min_obstacles=0,max_obstacles=0,max_steps=200)
                    state=env.reset()
            with torch.no_grad(): _,_,last=model(state[None])
            adv=advantages(rewards,values,dones,last.item());returns=adv+torch.tensor(values)
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            for _ in range(4):
                model.train()
                lp,v,entropy=model.evaluate(torch.stack(states),torch.stack(sources),torch.stack(destinations),torch.stack(actions))
                ratio=(lp.flatten()-torch.stack(logs)).exp()
                policy=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
                loss=policy+.5*(v.flatten()-returns).square().mean()-.01*entropy.mean()
                if not torch.isfinite(loss): raise RuntimeError('Non-finite PPO loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            writer.writerow([update+1,loss.item(),np.mean(completed[-10:]) if completed else '',len(completed)])
            print(f'update={update+1} loss={loss.item():.4f} episodes={len(completed)}')
    torch.save({'model_state_dict':model.state_dict(),'embed_dim':32,'seed':args.seed,'max_stack':12},out/'policy.pt')
if __name__=='__main__':main()
