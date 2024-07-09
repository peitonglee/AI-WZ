import os

import torch
from torch import optim

from argparses import args, device
from memory import ReplayMemory, Transition
from net_actor import NetActor
from net_critic import NetCritic
import torch.nn.functional as F


# PPO 算法
class PPO_Agent:
    def __init__(self, action_dim, buffer_capacity):
        self.actor = NetActor(action_dim).to(device)
        self.critic = NetCritic(action_dim).to(device)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.memory = ReplayMemory(buffer_capacity)
        self._load_models()

    def _load_models(self):
        # 加载预训练模型
        actor_path = os.path.join(args.model_dir, 'ppo_actor.pth')
        critic_path = os.path.join(args.model_dir, 'ppo_critic.pth')
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
            print(f'加载预训练actor模型：{actor_path}')
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path))
            print(f'加载预训练critic模型：{critic_path}')

    def _save_models(self):
        # 保存模型
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(args.model_dir, 'ppo_actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(args.model_dir, 'ppo_critic.pth'))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def store_transition(self, *args):
        self.memory.push(*args)

    def train(self):
        if len(self.memory) < args.batch_size:
            return
        transitions = self.memory.sample(args.batch_size)
        batch = Transition(*zip(*transitions))
        # 将 batch 中的数据转换为 PyTorch 张量
        state_batch = torch.FloatTensor(batch.state).to(device)
        action_batch = torch.FloatTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(device)
        done_batch = torch.FloatTensor(batch.done).to(device)

        # 计算当前状态的值
        values = self.critic(state_batch, action_batch).squeeze()

        # 计算目标值
        next_values = self.critic(next_state_batch, action_batch).squeeze()
        target_values = reward_batch + args.gamma * next_values * (1 - done_batch)

        # 计算优势
        advantages = target_values - values

        # 更新 Actor 和 Critic
        for _ in range(args.ppo_epoch):
            # 计算新的动作概率
            new_action_probs = self.actor(state_batch)

            # 计算旧动作概率
            with torch.no_grad():
                old_action_probs = self.actor(state_batch)

            # 计算概率比
            ratio = (new_action_probs / (old_action_probs + 1e-10))

            # 计算 surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # 优化 Actor 网络
            self.optimizer.zero_grad()
            actor_loss.backward()
            self.optimizer.step()

            # 计算 Critic 网络的损失
            critic_loss = F.mse_loss(values, target_values.detach())

            # 优化 Critic 网络
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self._save_models()
