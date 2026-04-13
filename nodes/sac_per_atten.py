#! /usr/bin/env python
# Authors deitieslulces #

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.nn as nn
from torch.optim import Adam
from replay_buffer import Per_ReplayBuffer

# ---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low) #[low,high]
    action = np.clip(action, low, high)
    return action



# class SelfAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#         self.scale = input_dim ** 0.5

#     def forward(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#         attention_weights = F.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
#         return attention_weights @ V


class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedAttention, self).__init__()
        self.gate = nn.Sequential(
 
            nn.Linear(input_dim, hidden_dim),#inputdim256,hiddendim256
            nn.Sigmoid()
        )
        self.transform = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        gate = self.gate(x)
        return self.transform(x) * gate
    
# class GatedAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(GatedAttention, self).__init__()

#         # 使用He初始化，适合ReLU激活函数
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.Sigmoid()
#         )
#         self.transform = nn.Linear(input_dim, hidden_dim)#input_dim=256,hidden_dim=256

#         # 初始化transform的权重
#         nn.init.kaiming_normal_(self.transform.weight, mode='fan_out', nonlinearity='relu')

#         # 添加Batch Normalization层
#         self.batch_norm = nn.BatchNorm1d(hidden_dim)
#         print("xian")
#     def forward(self, x):
        
#         gate = self.gate(x) #gate_shape=torch.Size([1, 256])
        
#         transformed = self.transform(x)  #trans_shape=torch.Size([1, 256])
#         # 在应用门控之前使用批归一化
#         # transformed = self.batch_norm(transformed)
#         # print(11111)
#         return transformed * gate
    

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        #Qnetwork

        #QNetwork(
        # (linear1_q1): Linear(in_features=30, out_features=256, bias=True)
        # (linear2_q1): Linear(in_features=256, out_features=256, bias=True)
        # (linear3_q1): Linear(in_features=256, out_features=256, bias=True)
        # (linear4_q1): Linear(in_features=256, out_features=1, bias=True)
        # (linear1_q2): Linear(in_features=30, out_features=256, bias=True)
        # (linear2_q2): Linear(in_features=256, out_features=256, bias=True)
        # (linear3_q2): Linear(in_features=256, out_features=256, bias=True)
        # (linear4_q2): Linear(in_features=256, out_features=1, bias=True)
        #)

        # Q1
        # self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear1_q1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.gated_attention_q1 = GatedAttention(hidden_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q1 = nn.Linear(hidden_dim, 1)
        # batch_size=100
        
        # Q2
        # self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear1_q2 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.gated_attention_q2 = GatedAttention(hidden_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q2 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)

    def forward(self, state, action):
        #state torch.Size([100, 2])
        #action torch.Size([100, 28])
        x_state_action = torch.cat([state, action], 1)  #torch.Size([100, 30])
       
        x1 = F.relu(self.linear1_q1(x_state_action))#torch.Size([100, 256])
        x1 = self.gated_attention_q1(x1)
        x1 = F.relu(self.linear2_q1(x1))   
        x1 = F.relu(self.linear3_q1(x1))
        
        x1 = self.linear4_q1(x1)
 


        x2 = F.relu(self.linear1_q2(x_state_action))
        x2 = self.gated_attention_q1(x2)
        x2 = F.relu(self.linear2_q2(x2))
        x2 = F.relu(self.linear3_q2(x2))
        x2 = self.linear4_q2(x2)

        return x1, x2


class PolicyNetwork(nn.Module):
    # log_std_min=-20, log_std_max=2 can little change
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()


        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_bound = action_bound

        # self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.gated_attention = GatedAttention(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state):
        # state 的 torch.Size([1, 28])
        
        x = F.relu(self.linear1(state))  #torch.Size([1, 256])
        # print("****x:{}".format(x.unsqueeze(0).shape))#x.unsqueeze(0) torch.Size([1,1,256])

        x = self.gated_attention(x) #torch.Size([1, 256])
        # print("****x.shape{}".format(x.shape))
        x = F.relu(self.linear2(x))     
        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    
    def sample(self, state, epsilon=1e-6):
        # torchsize([1, 28])
        
        mean, log_std = self.forward(state)  #得到state的均值和对数的标准差
        std = log_std.exp()     #得到state实际的标准差
        normal = Normal(mean, std) # 创建一个正态分布对象，从该分布中进行采样
        x_t = normal.rsample()  #从正态分布中进行再参数化抽样，使得动作采样过程可导，支持梯度反向传播。
        action = torch.tanh(x_t) #归一化，得到动作

        log_prob = normal.log_prob(x_t) #得到x_t的对数概率
        log_prob -= torch.log(1 - action.pow(2) + epsilon)#对对数概率进行修正
        log_prob = log_prob.sum(1, keepdim=True) #计算概率
        action = action * self.action_bound  #动作缩放
        return action, log_prob, mean, log_std


class SAC(object):
    def __init__(self, state_dim, action_dim, gamma=0.98, replay_buffer_size=100000, tau=0.2, alpha=0.2,
                 hidden_dim=256, action_bound=1.5, lr=0.0008, batch_size=100, per_flag=True, ACTION_V_MIN=0.0,  # m/s
                 ACTION_W_MIN=-1., ACTION_V_MAX=0.3, ACTION_W_MAX=2.):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.replay_buffer_size = replay_buffer_size
        self.tau = tau
        self.alpha = alpha
        self.hidden_dim = hidden_dim
        self.action_bound = action_bound
        self.lr = lr
        self.batch_size = batch_size
        self.per_flag = per_flag

        # self.action_range = [action_space.low, action_space.high]

        self.ACTION_V_MIN = ACTION_V_MIN
        self.ACTION_W_MIN = ACTION_W_MIN
        self.ACTION_V_MAX = ACTION_V_MAX
        self.ACTION_W_MAX = ACTION_W_MAX

        self.target_update_interval = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        # print('entropy', self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, action_bound).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        self.replay_buffer = Per_ReplayBuffer()

    def select_action(self, state, eval=False):
        #(28,)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)#参数代表了智能体的当前状态，激光雷达的数据
        #torch.Size([1, 28])  即转化成了tensor数据类型

        if eval == False:
            # 有四个返回值
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)

        action = action.detach().cpu().numpy()[0]  #选择第一个维度
        # print(action)
        #(2,)   [0.9995601  0.38583654]
        return action

    def store_transition(self, transition):
        self.replay_buffer.store(transition)

    def update_parameters(self, batch_size):
        # Sample a batch from memory
        #b_idx 用于存储采样的经验索引，和memory经验数据。ISWeights 用于存储重要性采样权重
        idx, memory, ISWeights = self.replay_buffer.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [], [], [], [], []
        for i in range(len(memory)):
            state_batch.append(memory[i][0])
            action_batch.append(memory[i][1])
            reward_batch.append(memory[i][2])
            next_state_batch.append(memory[i][3])
            done_batch.append(memory[i][4])

        state_batch = torch.FloatTensor(np.float32(state_batch)).to(self.device)
        next_state_batch = torch.FloatTensor(np.float32(next_state_batch)).to(self.device)
        action_batch = torch.FloatTensor(np.float32(action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.float32(reward_batch)).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(np.float32(done_batch)).to(self.device).unsqueeze(1)
        ISWeights = torch.FloatTensor(ISWeights).to(self.device)

        with torch.no_grad():
            # vf_next_target = self.value_target(next_state_batch)
            # next_q_value = reward_batch + (1 - done_batch) * self.gamma * (vf_next_target)
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target
            # abs_errors = torch.abs(next_q_value - torch.min(qf1_next_target, qf2_next_target)).cpu().numpy()

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  #
        qf2_loss = F.mse_loss(qf2, next_q_value)  #

        qf_loss = qf1_loss + qf2_loss
        td_error = (torch.abs(qf1 - next_q_value)).squeeze().detach().cpu().numpy()
        qf_loss = (qf_loss * ISWeights).mean()

        self.replay_buffer.batch_update(idx, td_error)

        self.critic_optim.zero_grad()
        # print("qf_loss: ", qf_loss)
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, mean, log_std = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        # Regularization Loss
        # reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        # policy_loss += reg_loss

        self.policy_optim.zero_grad()
        # print("policy_loss: ", policy_loss)
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        soft_update(self.critic_target, self.critic, self.tau)

    def buffer_size(self):
        return self.replay_buffer.__len__()

    # Save model parameters
    def save_models(self,ep):
        self.dirPath = dirPath
        model_dir = os.path.join(self.dirPath, 'model')
        policy_path = os.path.join(model_dir, f'{ep}_SAC_pertrans_policy_net.pth')
        value_path = os.path.join(model_dir, f'{ep}_SAC_pertrans_value_net.pth')
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.critic.state_dict(), value_path)
        
        print(f"Model has been saved for epoch {ep}...")

    # Load model parameters
    def load_models(self):
        self.policy.load_state_dict(torch.load(dirPath + '/model_save/PA_SAC/1240_SAC_pertrans_policy_net.pth'))
        self.critic.load_state_dict(torch.load(dirPath + '/model_save/PA_SAC/1240_SAC_pertrans_value_net.pth'))
        hard_update(self.critic_target, self.critic)
        print('***Models load********')


