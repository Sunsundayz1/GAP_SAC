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
from replay_buffer import ReplayBuffer, Per_ReplayBuffer

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
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1
        self.linear1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q1 = nn.Linear(hidden_dim, 1)

        # Q2
        self.linear1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4_q2 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        x_state_action = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1_q1(x_state_action))
        x1 = F.relu(self.linear2_q1(x1))
        x1 = F.relu(self.linear3_q1(x1))
        x1 = self.linear4_q1(x1)

        x2 = F.relu(self.linear1_q2(x_state_action))
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

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        action = action * self.action_bound
        return action, log_prob, mean, log_std


class SAC(object):
    def __init__(self, state_dim,
                 action_dim, gamma=0.99,
                 replay_buffer_size=100000,
                 tau=1e-2,
                 alpha=0.2,
                 hidden_dim=256, action_bound=1.5,
                 lr=0.0003, batch_size=100,
                 per_flag=True,
                 ACTION_V_MIN=0.0,  # m/s
                 ACTION_W_MIN=-1.,  # rad/s
                 ACTION_V_MAX=0.3,  # m/s
                 ACTION_W_MAX=2.  # rad/s
                 ):
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

        # self.automatic_entropy_tuning = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.ISWeights = np.empty((self.batch_size, 1))

        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        # print('entropy', self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, action_bound).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def select_action(self, state, eval=False):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
            action = torch.tanh(action)

        action = action.detach().cpu().numpy()[0]
        # print("*****{}".format(action))
        return action

    def store_transition(self, transition):
        (state, action, reward, next_state, done) = transition
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_parameters(self, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        # ISWeights = torch.FloatTensor(ISWeights).to(self.device)

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
    def save_models(self, episode_count):
        if not os.path.exists(dirPath + '/model/'):
            os.makedirs(dirPath + '/model/')
            torch.save(self.policy.state_dict(), dirPath + '/model/' + str(episode_count) + '_policy_net.pth')
            torch.save(self.critic.state_dict(), dirPath + '/model/' + str(episode_count) + '_value_net.pth')
            print("Model has been saved...")

        else:
            if os.path.exists(dirPath + '/model/' + str(episode_count) + '_policy_net.pth'):
                os.remove(dirPath + '/model/' + str(episode_count) + '_policy_net.pth')
                os.remove(dirPath + '/model/' + str(episode_count) + '_value_net.pth')
                torch.save(self.policy.state_dict(), dirPath + '/model/' + str(episode_count) + '_policy_net.pth')
                torch.save(self.critic.state_dict(), dirPath + '/model/' + str(episode_count) + '_value_net.pth')
                print("Model has been saved...")

            else:
                torch.save(self.policy.state_dict(), dirPath + '/model/' + str(episode_count) + '_policy_net.pth')
                torch.save(self.critic.state_dict(), dirPath + '/model/' + str(episode_count) + '_value_net.pth')
                print("Model has been saved...")

    # Load model parameters
    def load_models(self):
        self.policy.load_state_dict(torch.load(dirPath + '/model_save/SAC/1220_policy_net.pth'))
        self.critic.load_state_dict(torch.load(dirPath + '/model_save/SAC/1220_value_net.pth'))
        hard_update(self.critic_target, self.critic)
        print('***Models load***')


