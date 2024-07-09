import os

import torch
from torch import optim

from argparses import device, args
from memory import ReplayMemory, Transition
from net_actor import NetActor
from net_critic import NetCritic
import torch.nn.functional as F


class TD3_Agent:
    def __init__(self, action_dim, buffer_capacity):
        self.actor = NetActor(action_dim).to(device)
        self.actor_target = NetActor(action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = NetCritic(action_dim).to(device)
        self.critic2 = NetCritic(action_dim).to(device)
        self.critic1_target = NetCritic(action_dim).to(device)
        self.critic2_target = NetCritic(action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                           lr=args.critic_lr)
        self.memory = ReplayMemory(buffer_capacity)
        self.total_it = 0
        self._load_models()

    def _load_models(self):
        # 加载预训练模型
        actor_path = os.path.join(args.model_dir, 'td3_actor.pth')
        critic1_path = os.path.join(args.model_dir, 'td3_critic1.pth')
        critic2_path = os.path.join(args.model_dir, 'td3_critic2.pth')
        if os.path.exists(actor_path):
            self.actor.load_state_dict(torch.load(actor_path))
            self.actor_target.load_state_dict(torch.load(actor_path))
            print(f'加载预训练actor模型：{actor_path}')
        if os.path.exists(critic1_path):
            self.critic1.load_state_dict(torch.load(critic1_path))
            self.critic1_target.load_state_dict(torch.load(critic1_path))
            print(f'加载预训练critic1模型：{critic1_path}')
        if os.path.exists(critic2_path):
            self.critic2.load_state_dict(torch.load(critic2_path))
            self.critic2_target.load_state_dict(torch.load(critic2_path))
            print(f'加载预训练critic2模型：{critic2_path}')

    def _save_models(self):
        # 保存模型
        os.makedirs(args.model_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(args.model_dir, 'td3_actor.pth'))
        torch.save(self.critic1.state_dict(), os.path.join(args.model_dir, 'td3_critic1.pth'))
        torch.save(self.critic2.state_dict(), os.path.join(args.model_dir, 'td3_critic2.pth'))

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

        # 生成目标动作
        noise = torch.FloatTensor(action_batch).data.normal_(0, args.td3_noise).to(device)
        noise = noise.clamp(-0.5, 0.5)
        next_action_batch = (self.actor_target(next_state_batch) + noise).clamp(-1, 1)

        # 计算目标 Q 值
        target_q1 = self.critic1_target(next_state_batch, next_action_batch).squeeze()
        target_q2 = self.critic2_target(next_state_batch, next_action_batch).squeeze()
        target_q = torch.min(target_q1, target_q2)
        target_q = reward_batch + (1 - done_batch) * args.gamma * target_q

        # 计算当前 Q 值
        current_q1 = self.critic1(state_batch, action_batch).squeeze()
        current_q2 = self.critic2(state_batch, action_batch).squeeze()

        # 计算 Critic 损失
        critic1_loss = F.mse_loss(current_q1, target_q.detach())
        critic2_loss = F.mse_loss(current_q2, target_q.detach())

        # 优化 Critic 网络
        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新 Actor 网络和目标网络
        self.total_it += 1
        if self.total_it % args.td3_delay == 0:
            actor_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()

            # 优化 Actor 网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        self._save_models()
