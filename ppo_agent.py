import os
import random
import threading
import time

import cv2
import torch
from torch import optim

from argparses import args, device, globalInfo
from memory import Transition
from net_actor import NetActor
from net_critic import NetCritic
import torch.nn.functional as F


# PPO 算法
class PPO_Agent:
    def __init__(self):
        self.actor = NetActor().to(device)
        self.critic = NetCritic(3).to(device)  # 修改 action_dim 为 3，因为我们有三个输出
        self.optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
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
        if random.randint(0, 100000) < 10:

            # 随机生成 move_action, angle, info_action
            move_action = torch.randint(0, 2, (1, 1), dtype=torch.long, device=device)  # 0 或 1
            angle = torch.randint(0, 360, (1, 1), dtype=torch.long, device=device)  # 0 到 359
            info_action = torch.randint(0, 9, (1, 1), dtype=torch.long, device=device)  # 0 到 8

            action = torch.cat((move_action, angle, info_action), dim=-1)
            return action.cpu().data.numpy().flatten(), move_action, angle, info_action
        else:
            tmp_state_640_640 = self.preprocess_image(state)
            move_action, angle, info_action = self.actor(tmp_state_640_640)
            action = torch.cat((move_action, angle, info_action), dim=-1)
            return action.cpu().data.numpy().flatten(), move_action, angle, info_action

    def preprocess_image(self, image, target_size=(640, 640)):
        # 调整图像大小
        resized_image = cv2.resize(image, target_size)
        # 转换为张量并调整维度顺序 [height, width, channels] -> [channels, height, width]
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1)
        return tensor_image.to(device).unsqueeze(0)

    def train(self):
        while True:
            if not globalInfo.is_memory_bigger_batch_size_ppo():
                time.sleep(1)
                continue

            print("ppo training ...")
            transitions = globalInfo.random_batch_size_memory_ppo()
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
                move_action, angle, info_action = self.actor(state_batch)
                new_action_probs = torch.cat((move_action, angle, info_action), dim=-1)

                # 计算旧动作概率
                with torch.no_grad():
                    old_move_action, old_angle, old_info_action = self.actor(state_batch)
                    old_action_probs = torch.cat((old_move_action, old_angle, old_info_action),
                                                 dim=-1)

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

    def start_train(self):
        training_thread = threading.Thread(target=self.train)
        training_thread.start()
        # training_thread.join()
