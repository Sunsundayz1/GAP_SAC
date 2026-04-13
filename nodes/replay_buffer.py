#! /usr/bin/env python
# Authors deitieslulces #

import os.path
import pickle
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('save_model/'):
            os.makedirs('sac_model/')

        if save_path is None:
            save_path = "save_model/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, 'rb') as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity


class SumTree(object):

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1) #创建一个numpy数组，所有元素都为0
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]

        self.data = list(np.zeros(capacity, dtype=object))  # 创建一个长度为capacity的新NumPy数组，所有元素初始化为None
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, transition):
        #方法用于将新的经验及其优先级添加到经验回放缓冲区中，并更新树结构中的优先级信息。
        tree_idx = self.data_pointer + self.capacity-1  #指针+长度-1
        self.data[self.data_pointer] = transition  # 将经验存在数组中的 data.pointer位置
        # print("data_pointer:  ",self.data_pointer)
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # 超过了最大长度
            self.data_pointer = 0

    def update(self, tree_idx, p):
        #方法用于将新的经验及其优先级添加到经验回放缓冲区中，并更新树结构中的优先级信息。
        change = p - self.tree[tree_idx]  #新优先级与当前节点优先级之间的差异
        #change=0.0553818941116333
        self.tree[tree_idx] = p

        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2  #父亲节点
            self.tree[tree_idx] += change  #父亲节点加上变化

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # 左子
            cr_idx = cl_idx + 1         #右子
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
        # 叶子节点的索引、叶子节点的优先级以及叶子节点对应的经验元组

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Per_ReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree

    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self,
                 state_dim=28,
                 action_dim=2,
                 size=100000,
                 ):
        self.tree = SumTree(size)
        self.full_flag = False
        self.memory_num = 0
        self.memory_size = size  #最大经验

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])# 找到数组的最大值
        if max_p == 0:
            max_p = self.abs_err_upper
       
        self.tree.add(max_p, transition)   # set the max p for new p
        if self.memory_num < self.memory_size:
            self.memory_num += 1

    def sample(self, batch_size=32):
        n = batch_size #n = 100
        
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))  #b_idx 用于存储采样的经验索引，ISWeights 用于存储重要性采样权重
        b_memory = []
        
        pri_seg = self.tree.total_p / n       # 分割经验
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # 计算最小概率
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b) # ab中随机选一个值
            idx, p, data = self.tree.get_leaf(v)
            '''
            while not p:
                v = np.random.uniform(a, b)
                idx, p, data = self.tree.get_leaf(v)

            if not p:
                print("idx: ", idx, "  p: ", p)
                print("self.total.p :  ", self.tree.total_p, "number: ", self.memory_num, " v: ", v)
            '''
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory.append(data)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        # print ("tree_idx: ", tree_idx, "abs_errors: ", abs_errors)
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self):
        return self.memory_num


class PrioritizedReplay(object):
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        P = probs / probs.sum()

        # gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)

