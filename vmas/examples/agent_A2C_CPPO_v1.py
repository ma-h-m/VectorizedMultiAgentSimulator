import os 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import argparse
import gc
import sys
import psutil
from torch.cuda.amp import autocast as autocast
from memory_profiler import profile
class PPOmemory:
    def __init__(self, mini_batch_size, mini_batch_on = False):
        self.states = []  # 状态
        self.actions = []  # 实际采取的动作
        self.probs = []  # 动作概率
        self.vals = []  # critic输出的状态值
        self.rewards = []  # 奖励
        self.dones = []  # 结束标志
        self.next_vals = []
        
        self.next_states = []
       
        self.mini_batch_size = mini_batch_size  # minibatch的大小
        self.mini_batch_on = mini_batch_on

        self.dist=[]
    def sample(self):
        if self.mini_batch_on == False:
            return self.states,self.next_states, self.actions, self.probs, \
                self.vals,self.next_vals , self.rewards, self.dones,self.dist, [np.arange(len(self.states), dtype=np.int64)]
        n_states = len(self.states)  
        batch_start = np.arange(0, n_states, self.mini_batch_size)  # 每个batch开始的位置[0,5,10,15]
        indices = np.arange(n_states, dtype=np.int64)  # 记录编号[0,1,2....19]
        np.random.shuffle(indices)  # 打乱编号顺序[3,1,9,11....18]
        mini_batches = [indices[i:i + self.mini_batch_size] for i in batch_start]  # 生成4个minibatch，每个minibatch记录乱序且不重复
        return self.states,self.next_states, self.actions, self.probs, \
                self.vals,self.next_vals , self.rewards, self.dones, mini_batches

        # return np.array(self.states), np.array(self.actions), np.array(self.probs), \
        #       np.array(self.vals), np.array(self.rewards), np.array(self.dones), mini_batches

    # 每一步都存储trace到memory
    def push(self, states,next_states, action, prob, val, next_vals, rewards, done,dist):
        
        # for sta,usta,ne_sta,une_sta,act,pro,va,nva,rew,don in zip(states, uniform_states,next_states,next_uniform_states, action, prob, val, next_vals, rewards, done):
        for sta,ne_sta,act,pro,va,nva,rew,don,mean,s2 in zip(states,next_states, action, prob, val, next_vals, rewards, done,dist[0],dist[1]):
            self.states.append(sta)
            # self.uniform_states.append(usta)
            self.next_states.append(ne_sta)
            # self.next_uniform_states.append(une_sta)
            self.actions.append(act)
            self.probs.append(pro)
            self.vals.append(va)
            self.rewards.append(rew)
            self.dones.append(don)
            self.next_vals.append(nva)
            self.dist.append((mean,s2))
    # 固定步长更新完网络后清空memory
    def clear(self):

        del self.states
        del self.actions
        del self.probs
        del self.vals
        del self.next_vals
        del self.rewards
        # del self.uniform_states
        del self.dones
        del self.next_states
        del self.dist
        # del self.next_uniform_states
        gc.collect()
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.next_vals = []
        
        self.next_states = []
        self.dist = []
# actor:policy network
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, cfg):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(n_states, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),)

        # self.actor = nn.Sequential(
        #     nn.Linear(n_states, cfg.hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        #     nn.Tanh())
        self.mean = nn.Linear(cfg.hidden_dim,n_actions)
        self.s2 = nn.Linear(cfg.hidden_dim,n_actions)

        self.value = nn.Linear(cfg.hidden_dim,1)

    def forward(self, state):
        rt = self.actor(state)
        mean = self.mean(rt) 
        mean = torch.tanh(mean)
        
        s2 = self.s2(rt)
        s2 = F.softplus(s2) * 1e-2
        s2 = torch.clamp(s2,1e-8,1e6)
        return mean,s2
    
    def value_predict(self,state):
        rt = self.actor(state)
        return self.value(rt)
# critic:value network
class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.norm = nn.BatchNorm1d(1)
        
    def forward(self, state):
        value = self.critic(state)
        value = self.norm(value)
        return value
class Agent:
    def __init__(self, n_states, n_actions, cfg,agent_num):
        # 训练参数
        self.gamma = cfg.gamma  # 折扣因子
        self.n_epochs = cfg.n_epochs  # 每次更新重复次数
        self.gae_lambda = cfg.gae_lambda  # GAE参数
        self.policy_clip = cfg.policy_clip  # clip参数
        self.device = cfg.device  # 运行设备
        self.agent_num = agent_num
        # AC网络及优化器
        self.actor = Actor(n_states * agent_num, n_actions * agent_num, cfg).to(self.device)
        self.critic = Critic(n_states * agent_num, cfg.hidden_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.actor_optim, step_size=20, gamma=0.95)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.critic_optim, step_size=20, gamma=0.9) 
        # 经验池
        self.memory = PPOmemory(cfg.mini_batch_size)

        self.beta_clone = cfg.beta_clone
        self.aux_epochs = cfg.aux_epochs
    def choose_action(self, state):


        # state = torch.tensor(state, dtype=torch.float).to(self.device)  # 数组变成张量
        
        dist = self.actor(state) # action分布
        # for i in range(len(dist)):
        #     dist[i] = dist[i][0].detach(),dist[i][1].detach()
        # dist = dist[0].reshape(len(state),self.agent_num,-1).detach(),dist[1].reshape(len(state),self.agent_num,-1).detach()
        value = self.critic(state)  # state value值
        # action_rec = []
        # prob_rec = []

        # for action_mean, action_s2 in dist:
            # action = dist.sample()  # 随机选择action
        action_mean, action_s2 = dist
        action = torch.randn(size=dist[0].shape).to(self.device)
        action = action * torch.sqrt(action_s2) + action_mean
        action = torch.clamp(action,-1,1).detach()
        # action = action.detach()
        # action_rec.append(action)

        prob = 1 / (torch.sqrt(2 * torch.pi * action_s2)) * torch.exp(-(action - action_mean) ** 2 / 2 / action_s2)
        prob = torch.log(prob).detach()
        # prob_rec.append(prob)
        # prob = torch.squeeze(dist.log_prob(action)).item()  # action对数概率

        # action = torch.squeeze(action).item()
        # value = torch.squeeze(value).item()
        return action, prob, value,dist

    def learn(self):
        for _ in range(self.n_epochs):
            # memory中的trace以及处理后的mini_batches，mini_batches只是trace索引而非真正的数据
            states_arr,next_states_arr, actions_arr, old_probs_arr, vals_arr, next_vals_arr,\
                rewards_arr, dones_arr,_,mini_batches = self.memory.sample()

            # 计算GAE
            # values = vals_arr
            rewards_arr = torch.stack(rewards_arr,dim=0)
            # rewards_arr = torch.tensor(rewards_arr, dtype=torch.float).to(self.device)
            # next_vals_arr = torch.tensor(next_vals_arr, dtype=torch.float).to(self.device)
            # vals_arr = torch.tensor(vals_arr, dtype=torch.float).to(self.device)
            next_states_arr = torch.stack(next_states_arr,dim = 0).to(self.device)
            states_arr = torch.stack(states_arr,dim = 0).to(self.device)
            next_vals_arr = torch.squeeze(self.critic(next_states_arr).detach())
            vals_arr = torch.squeeze(self.critic(states_arr).detach())
            advantage = rewards_arr + self.gamma * self.gae_lambda * next_vals_arr - vals_arr
            # advantage = torch.tensor(np.zeros(len(rewards_arr), dtype=np.float32)).to(self.device)
            # for t in range(len(rewards_arr) - 1):
            #     discount = 1
            #     a_t = 0
            #     for k in range(t, len(rewards_arr) - 1):
            #         a_t += discount * (rewards_arr[k] + self.gamma * values[k+1] * (1 - int(dones_arr[k])) - values[k])
            #         discount *= self.gamma * self.gae_lambda
            #     advantage[t] = a_t
            # advantage = torch.tensor(advantage).to(self.device)
            advantage = advantage.detach()
            # mini batch 更新网络
            values = vals_arr
            states = states_arr

            for batch in mini_batches:
                with autocast():
                # states = torch.tensor(states_arr[batch], dtype=torch.float).to(self.device)
                # states = torch.stack (states_arr,dim = 0)[batch].detach()
                    old_probs = torch.stack (old_probs_arr,dim = 0)[batch].detach()
                    actions = torch.stack (actions_arr,dim = 0)[batch].detach()
                    
                    # old_probs = torch.tensor(old_probs_arr[batch]).to(self.device)
                    # actions = torch.tensor(actions_arr[batch]).to(self.device)

                    # mini batch 更新一次critic和actor的网络参数就会变化
                    # 需要重新计算新的dist,values,probs得到ratio,即重要性采样中的新旧策略比值
                    action_mean,action_s2 = self.actor(states)
                    critic_value = torch.squeeze(self.critic(states))
                    new_probs = 1 / (torch.sqrt(2 * torch.pi * action_s2)) * torch.exp(-(actions - action_mean) ** 2 / 2 / action_s2)
                    new_probs = torch.clamp(new_probs,1e-30)
                    new_probs = torch.log(new_probs)
                    # new_probs = dist.log_prob(actions)
                    prob_ratio = torch.sum(new_probs,dim=1).exp() / torch.sum(old_probs,dim=1).exp()
                    

                    # tmp = advantage[batch]
                    # actor loss
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clip_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,\
                        1 + self.policy_clip) * advantage[batch]
                    actor_loss = -torch.min(weighted_probs, weighted_clip_probs).mean()
                    # critic loss
                    # critic_loss = F.mse_loss(critic_value, rewards_arr[batch] + self.gamma * self.gae_lambda * next_vals_arr[batch])
                    critic_loss = rewards_arr[batch] + self.gamma * self.gae_lambda * next_vals_arr[batch] - critic_value
                    # returns = advantage[batch] + values[batch]
                    # critic_loss = (returns - critic_value) ** 2
                    critic_loss = critic_loss ** 2
                    critic_loss = critic_loss.mean()
                    # total_loss
                    total_loss = actor_loss + 0.5 * critic_loss

                # 更新
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.step()
                if torch.isnan(self.actor.s2.weight.data).any() or torch.isnan(self.actor.mean.weight.data).any():
                    print("Sth is nan!")
        # print("Learned once!")
        for _ in range(self.aux_epochs):
            states_arr,next_states_arr, actions_arr, old_probs_arr, vals_arr, next_vals_arr,\
                rewards_arr, dones_arr, dist,mini_batches = self.memory.sample()

            rewards_arr = torch.stack(rewards_arr,dim=0)
            next_states_arr = torch.stack(next_states_arr,dim = 0).to(self.device)
            states_arr = torch.stack(states_arr,dim = 0).to(self.device)
            next_vals_arr = torch.squeeze(self.critic(next_states_arr).detach())
            vals_arr = torch.squeeze(self.critic(states_arr).detach())
            
            states = states_arr
            action_mean,action_s2 = self.actor(states)
            critic_value = torch.squeeze(self.critic(states))


            advantage = (rewards_arr[batch] + self.gamma * self.gae_lambda * next_vals_arr[batch]).detach()

            old_mean = torch.stack([x[0] for x in dist],dim = 0).detach()
            old_s2 = torch.stack([x[1] for x in dist],dim=0).detach()

            

            actor_val = torch.squeeze(self.actor.value_predict(states_arr))
            loss_aux = (0.5 * (advantage - actor_val) ** 2)
            kl_div = (torch.log(action_s2 / old_s2) / 2 + (old_s2  + (old_mean - action_mean) ** 2) / 2 / action_s2 - 1 / 2).detach()
            kl_div = torch.mean(kl_div,dim=1)
            loss_joint = (loss_aux + self.beta_clone * kl_div).mean()
            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss_joint.backward()

            critic_loss = rewards_arr[batch] + self.gamma * self.gae_lambda * next_vals_arr[batch] - critic_value
            # returns = advantage[batch] + values[batch]
            # critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss ** 2
            critic_loss = critic_loss.mean()
            critic_loss.backward()
        self.memory.clear()
def get_args():
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='CPPO', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CartPole-v1', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=200, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--mini_batch_size', default=20, type=int, help='mini batch size')
    parser.add_argument('--n_epochs', default=1, type=int, help='update number')
    parser.add_argument('--aux_epochs', default=10, type=int, help='update number')
    parser.add_argument('--actor_lr', default=5e-5, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=5e-5, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.9, type=float, help='GAE lambda')
    parser.add_argument('--policy_clip', default=0.2, type=float, help='policy clip')
    parser.add_argument('-batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dim')
    parser.add_argument('--device', default='cuda', type=str, help="cpu or cuda")

    parser.add_argument('--beta_clone', default=1, type=float, help='GAE lambda')
    args = parser.parse_args()
    return args

# @profile
def train(cfg, env, agent, max_steps):
    print('开始训练！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards_rec = []
    steps = 0
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        done = torch.tensor([False] * state[0].shape[0]).to(cfg.device)
        mask = done == 0
        ep_reward = 0
        steps = 0
        uniform_state = torch.cat(state,dim=1)
        while (not done.all()) and steps < max_steps :
            
            action, prob, val,dist = agent.choose_action(uniform_state)

            # actions = action.view(-1,len(env.observation_space), env.action_space[0].shape[0]).transpose(0,1)
            # actions = [x for x in actions]
            
            state_, reward, done, _ = env.step(action.view(-1,len(env.observation_space), env.action_space[0].shape[0]).transpose(0,1))
            
            steps += 1
            ep_reward += torch.mean(torch.cat(reward,dim = 0)).item()

            if done.all():
                break
            mask = done == 0
            
            uniform_next_states = torch.cat(state_,dim=1)
            # states_ = torch.cat(state,dim=1).detach()
            next_vals = agent.critic(uniform_next_states)
            rewards = sum(reward)
            agent.memory.push(uniform_state[mask],uniform_next_states[mask], action[mask], prob[mask], val[mask], next_vals[mask], rewards[mask], done[mask],(dist[0][mask],dist[1][mask]))
            # for act,pro,rew,sta,sta_,va,neva,dis in zip(action,prob,reward,state,state_,val,next_vals,dist):

            #     agent.memory.push(sta[mask],sta_[mask], act[mask], pro[mask], va[mask], neva[mask], rew[mask], done[mask],(dis[0][mask],dis[1][mask]))
            mask = done == 0
            if steps % cfg.batch_size == 0:
                agent.learn()
            state = state_
        
    
        rewards_rec.append(ep_reward)
        agent.scheduler_actor.step()
        agent.scheduler_critic.step()

        if (i_ep + 1) % 1 == 0:
            print(f"回合：{i_ep + 1}/{cfg.train_eps}，奖励：{ep_reward:.2f}")

            # print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

            # print('env size:',sys.getsizeof(env) / 1024 / 1024, 'MB')
            # print('agent size:',sys.getsizeof(agent) / 1024 / 1024, 'MB')
            # print('state size:',sys.getsizeof(state) / 1024 / 1024, 'MB')
            # print('prob size:',sys.getsizeof(prob) / 1024 / 1024, 'MB')
            # print('val size:',sys.getsizeof(val) / 1024 / 1024, 'MB')
            # print('states_tmp size:',sys.getsizeof(states_tmp) / 1024 / 1024, 'MB')
            # print('next_vals size:',sys.getsizeof(next_vals) / 1024 / 1024, 'MB')
            # print('state_ size:',sys.getsizeof(state_) / 1024 / 1024, 'MB')
            # print('uniform_states size:',sys.getsizeof(uniform_states) / 1024 / 1024, 'MB')
            
            # print('done size:',sys.getsizeof(done) / 1024 / 1024, 'MB')
            # print('mask size:',sys.getsizeof(mask) / 1024 / 1024, 'MB')
            # print('ep_reward size:',sys.getsizeof(ep_reward) / 1024 / 1024, 'MB')
            # print('agent.memory size:',sys.getsizeof(agent.memory) / 1024 / 1024, 'MB')
            # print('agent.critic size:',sys.getsizeof(agent.critic) / 1024 / 1024, 'MB')
            
        if agent.device == 'cuda':
            torch.cuda.empty_cache()
    print('完成训练！')

def env_agent_config(cfg, seed=1):
    # env = gym.make(cfg.env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(n_states, n_actions, cfg)
    if seed != 0:
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)
    return env, agent


# cfg = get_args()
# env, agent = env_agent_config(cfg, seed=1)
# train(cfg, env, agent)
