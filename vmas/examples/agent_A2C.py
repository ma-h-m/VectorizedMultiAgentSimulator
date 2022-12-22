"""
CPPO version
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

class Agent(object):
    def __init__(self, env):
        self.env = env
    def make_action(self, observation, test=True):
        raise NotImplementedError("Subclasses should implement this!")
    def init_game_setting(self):
        raise NotImplementedError("Subclasses should implement this!")
    def run(self):
        raise NotImplementedError


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,action_space,agent_num,action_limit = 1,device = 'cpu'):
        super(ActorNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.input_size = input_size
        self.action_space = action_space #  每个智能体有几个要选择的动作
        self.action_limit = action_limit # 限制动作的范围
        self.agent_num = agent_num
        # ACTOR
        self.fc1 = nn.Linear(self.input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc_mean = nn.Linear(hidden_size[-1], output_size)
        self.fc_s2 = nn.Linear(hidden_size[-1], output_size)
        self.device = device

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        inputs = torch.Tensor(inputs)
        inputs = inputs.view(-1, self.input_size)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = F.relu(self.fc3(inputs))
        mean = self.fc_mean(inputs) 
        mean = torch.tanh(mean) * self.action_limit

        s2 = self.fc_s2(inputs)
        s2 = F.softplus(s2)

        mean = mean.view(-1,self.agent_num,self.action_space)
        s2 = s2.view(-1,self.agent_num,self.action_space)
        return mean,s2

class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1,device = 'cpu'):
        super(CriticNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        ##################
        self.input_size = input_size
        self.device = device
        # Critic
        self.fc1 = nn.Linear(self.input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc4 = nn.Linear(hidden_size[-1], output_size)

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        ##################
        inputs = torch.Tensor(inputs)
        inputs = inputs.view(-1, self.input_size)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = F.relu(self.fc3(inputs))
        inputs = self.fc4(inputs)
        return inputs


class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Trajectory buffer. It will clear the buffer after updating.
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer_rewards = []
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_size = buffer_size

        self.buffer_next_obs = []
        self.buffer_actions_mean = []
        self.buffer_actions_s2 = []

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        ##################
        return len(self.buffer_rewards)

    def push(self, reward, obs, action,action_distribution):
        ##################
        # YOUR CODE HERE #
        ##################
        if len(self.buffer_rewards) == self.buffer_size:
            self.buffer_rewards.pop(0)
            self.buffer_obs.pop(0)
            self.buffer_actions.pop(0)
            self.buffer_actions_mean.pop(0)
            self.buffer_actions_s2.pop(0)
        self.buffer_rewards.append(reward)
        self.buffer_obs.append(obs)
        self.buffer_actions.append(action)
        self.buffer_actions_mean.append(action_distribution[0])
        self.buffer_actions_s2.append(action_distribution[1])

    def push_multiple(self, reward, obs, actions, next_obs,action_distribution):
        ##################
        # YOUR CODE HERE #
        ##################
        # actions = [x for x in torch.stack(action,dim=1)]
        rewards = [x for x in reward]
        ob = [x for x in obs]
        next_ob = [x for x in next_obs]
        for rew, act,o,next_o,act_dis1,act_dis2 in zip(rewards, actions, ob, next_ob,action_distribution[0],action_distribution[1]):
            if len(self.buffer_rewards) == self.buffer_size:
                self.buffer_rewards.pop(0)
                self.buffer_obs.pop(0)
                self.buffer_actions.pop(0)
                self.buffer_actions_mean.pop(0)
                self.buffer_actions_s2.pop(0)
                self.buffer_next_obs.pop(0)
            self.buffer_rewards.append(rew)
            self.buffer_obs.append(o)
            self.buffer_actions.append(act)
            self.buffer_next_obs.append(next_o)
            self.buffer_actions_mean.append(act_dis1)
            self.buffer_actions_s2.append(act_dis2)

    def sample(self, batch_size: int):
        # indices = torch.randint(0, high=self.buffer_actions.shape[0], size=(batch_size,))
        # b_state = self.buffer_obs[indices]
        # b_next = self.buffer_next_obs[indices]
        # b_action = self.buffer_actions[indices]
        # b_reward = self.buffer_rewards[indices]
        # return b_reward, b_state, b_action,  b_next
        return self.buffer_rewards, self.buffer_obs, self.buffer_actions, self.buffer_next_obs,self.buffer_actions_mean,self.buffer_actions_s2


    def clean(self):
        ##################
        # YOUR CODE HERE #
        ##################
        self.buffer_rewards = []
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_actions_mean = []
        self.buffer_actions_s2 = []
        self.buffer_next_obs = []


    def totensor(self):
        self.buffer_rewards = torch.stack(self.buffer_rewards,dim=0)
        self.buffer_obs = torch.stack(self.buffer_obs,dim=0)
        self.buffer_actions = torch.stack(self.buffer_actions,dim=0)

        self.buffer_next_obs = torch.stack(self.buffer_next_obs,dim=0)
        self.buffer_actions_mean = torch.stack(self.buffer_actions_mean,dim=0)
        self.buffer_actions_s2 = torch.stack(self.buffer_actions_s2,dim=0)

class AgentA2C(Agent):
    def __init__(self, env, hidden_sizes,lr,gamma,grad_norm_clip,test,epoch,seed,max_steps,n_envs,batch_size, device):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentA2C, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        ##################
        self.hidden_size = hidden_sizes
        self.lr = lr
        self.gamma = gamma
        self.grad_norm_clip = grad_norm_clip
        self.test = test
        self.n_frames = epoch
        self.seed = seed
        self.env = env
        self.input_size = env.observation_space[0].shape[0] * len(env.observation_space)
        self.output_size = len(env.action_space) * env.action_space[0].shape[0]
        self.actor = ActorNetwork(self.input_size, self.hidden_size, self.output_size,env.action_space[0].shape[0],env.n_agents).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = CriticNetwork(self.input_size, self.hidden_size, 1).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.max_steps = max_steps  # 最大步数
        self.buffer_size = self.max_steps * n_envs
        self.buffer = ReplayBuffer(self.buffer_size)
        self.batch_size = batch_size
        self.speed = 0.95
        self.device = device

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.actor.load_state_dict(torch.load('a2cnetwork_params.pth'))

    def actorTrain(self, val):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # _, _, actions, _,actions_mean,actions_s2 = self.buffer.sample(self.batch_size)
        rewards, obs, actions, next_obs,actions_mean, actions_s2 = self.buffer.sample(self.batch_size)
        actions = actions.detach()

        _, (actions_mean,actions_s2) = self.make_action(obs)
        # obs = torch.stack(obs,dim=0)
        # rewards = torch.stack(rewards,dim=0)
        # next_obs = torch.stack(obs,dim=0)

        # pro_actions = self.actor(obs)
        # reward = torch.mean(rewards,dim=1)
        # vs = self.critic(obs).detach()
        
        # qs = reward + self.gamma * self.critic(next_obs).detach()
        # discounted_r = np.zeros_like(rewards)
        # for i in reversed(range(0, len(rewards))):
        #     final_reward = final_reward * self.gamma + rewards[i]
        #     discounted_r[i] = final_reward
        # qs = torch.Tensor(discounted_r)


        # adv_fun = qs - vs
        # loss = -torch.mean(torch.log(actions) * error)
        self.actor_optim.zero_grad()
        prob = 1 / (torch.sqrt(2 * torch.pi * actions_s2)) * torch.exp(-(actions - actions_mean) ** 2 / 2 / actions_s2)
        log_pi = torch.log(prob)
        # val_rep = val.unsqueeze(dim=2).repeat(1,log_pi.shape[1],log_pi.shape[2])
        # val_rep = val_rep.detach()

        # print(actions)
        # torch.log(pro_actions) * torch.tensor(actions)
        loss = -torch.mean(torch.sum(log_pi,dim=[1,2]) * val)
        # loss = - torch.mean(log_pi )
        loss.backward()
        self.actor_optim.step()


    def criticTrain(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        rewards, obs, _, next_obs,_, _= self.buffer.sample(self.batch_size)
        values = self.critic(obs)
        pro_values = self.critic(next_obs).detach()
        reward = torch.mean(rewards,dim=1).reshape(-1,1)
        # print(qs)
        # qslist = []
        # for i in range(len(qs)):
        #     qslist.append([qs[i]])
        # qs = torch.tensor(qslist)
        # print(qs)
        target = pro_values * self.gamma + reward 
        loss = F.mse_loss(target,values)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return (target - values).detach()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        # YOUR CODE HERE #
        ##################
        #
        action_mean,action_s2 = self.actor(observation)
        actions = torch.randn(size=action_mean.shape).to(self.device)
        actions = actions * torch.sqrt(action_s2.detach()) + action_mean.detach()
        # actions = actions.transpose(0,1)
        # actions = [x for x in actions]
        return torch.clamp(actions,-self.actor.action_limit,self.actor.action_limit),(action_mean,action_s2)

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        torch.manual_seed(self.seed)
        writer = SummaryWriter('./log')
        epi_array = []
        step = 0
        for i_episode in range(self.n_frames):
            final_reward = 0
            obs,  episode_reward = self.env.reset(), 0
            obs = torch.cat(obs,dim=1)
            for step in range(self.max_steps):
                step+=1
                #obs = torch.stack(obs,dim=1)
                # print(obs)
                
                # print(obs)
                action, action_distribution = self.make_action(obs, self.test)
                actions = action.transpose(0,1)
                actions = [x for x in actions]
                next_obs, rews, done, info = self.env.step(actions)
                #next_obs = torch.stack(next_obs,dim=2)
                
                # print(next_obs)
                next_obs = torch.cat(next_obs,dim=1)
                #next_obs = next_obs.reshape(obs.shape)
                # print(next_obs)
                rewards = torch.stack(rews,dim=1)
                global_reward = rewards.mean(dim=1)
                # one_hot_action = [int(k == action) for k in range(self.output_size)]
                # self.buffer.push(reward, obs, one_hot_action)

                if done.all():
                    break
                mask = done == 0


                self.buffer.push_multiple(rewards[mask], obs[mask], action[mask],next_obs[mask],(action_distribution[0][mask],action_distribution[1][mask]))
                self.buffer.totensor()
                val = self.criticTrain()
                self.actorTrain(val)


                self.buffer.clean()
                # print(rewards[mask].shape)
                # print(done)
                episode_reward += global_reward
                obs = next_obs

                # self.batch_size += 1
            #     if done:
            #         break
            # if not done:
            # obs = torch.cat(obs,dim=1)
            
            #final_reward = self.critic(obs[mask])
            
            
            # qs = self.actorTrain(final_reward)
            # self.criticTrain(qs)
            # self.buffer.clean()
            tmp = torch.mean(episode_reward)
            print("episode" + str(i_episode) + ";reward:" + str(tmp))
            epi_array.append(tmp)
            #writer.add_scalar('a2c_reward2', episode_reward, i_episode)
            #torch.save(self.actor.state_dict(), 'a2cnetwork_params2.pth')  # 只保存网络中的参数

        s = np.linspace(0,1,n_frames + 1)
        import matplotlib.pyplot as plt
        plt.plot(s,epi_array)
        plt.show()
