import os
import threading
import time

import torch
from torch import optim

from argparses import device, args, globalInfo
from memory import Transition

import torch.nn.functional as F

from net_attack_actor import NetAttackActor
from net_critic import NetCritic


class TD3_Agent:
    def __init__(self):
        self.device = device
        self.args = args
        self.actor = NetAttackActor().to(device)
        self.actor_target = NetAttackActor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = NetCritic(5).to(device)
        self.critic2 = NetCritic(5).to(device)
        self.critic1_target = NetCritic(5).to(device)
        self.critic2_target = NetCritic(5).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                           lr=args.critic_lr)

        self.total_it = 0
        self._load_models()

    def _load_models(self):
        actor_path = os.path.join(self.args.model_dir, 'td3_actor.pth')
        critic1_path = os.path.join(self.args.model_dir, 'td3_critic1.pth')
        critic2_path = os.path.join(self.args.model_dir, 'td3_critic2.pth')
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
        os.makedirs(self.args.model_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(self.args.model_dir, 'td3_actor.pth'))
        torch.save(self.critic1.state_dict(), os.path.join(self.args.model_dir, 'td3_critic1.pth'))
        torch.save(self.critic2.state_dict(), os.path.join(self.args.model_dir, 'td3_critic2.pth'))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self):
        while True:
            if not globalInfo.is_memory_bigger_batch_size_td3():
                time.sleep(1)
                continue
            transitions = globalInfo.random_batch_size_memory_td3()
            batch = Transition(*zip(*transitions))
            state_batch = torch.FloatTensor(batch.state).to(self.device)
            action_batch = torch.FloatTensor(batch.action).to(self.device)
            reward_batch = torch.FloatTensor(batch.reward).to(self.device)
            next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
            done_batch = torch.FloatTensor(batch.done).to(self.device)

            noise = torch.FloatTensor(action_batch).data.normal_(0, self.args.td3_noise).to(self.device)
            noise = noise.clamp(-0.5, 0.5)
            next_action_batch = (self.actor_target(next_state_batch) + noise).clamp(-1, 1)

            target_q1 = self.critic1_target(next_state_batch, next_action_batch).squeeze()
            target_q2 = self.critic2_target(next_state_batch, next_action_batch).squeeze()
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1 - done_batch) * self.args.gamma * target_q

            current_q1 = self.critic1(state_batch, action_batch).squeeze()
            current_q2 = self.critic2(state_batch, action_batch).squeeze()

            critic1_loss = F.mse_loss(current_q1, target_q.detach())
            critic2_loss = F.mse_loss(current_q2, target_q.detach())

            self.critic_optimizer.zero_grad()
            critic1_loss.backward()
            critic2_loss.backward()
            self.critic_optimizer.step()

            self.total_it += 1
            if self.total_it % self.args.td3_delay == 0:
                actor_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self._soft_update(self.actor.parameters(), self.actor_target.parameters(), self.args.tau)
                self._soft_update(self.critic1.parameters(), self.critic1_target.parameters(), self.args.tau)
                self._soft_update(self.critic2.parameters(), self.critic2_target.parameters(), self.args.tau)

            self._save_models()

    def _soft_update(self, source_params, target_params, tau):
        for param, target_param in zip(source_params, target_params):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def start_train(self):
        training_thread = threading.Thread(target=self.train)
        training_thread.start()
        training_thread.join()
