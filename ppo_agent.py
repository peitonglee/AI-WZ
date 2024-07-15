import os
import random
import threading
import time

import cv2
import numpy as np
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
        self.critic = NetCritic(8).to(device)  # action_dim 为 8，因为我们有8个输出
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
        if random.randint(0, 10) < 1:
            # 随机生成 move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3
            move_action = torch.randint(0, 2, (1, 1), dtype=torch.long, device=device)  # 0 或 1
            angle = torch.randint(0, 360, (1, 1), dtype=torch.long, device=device)  # 0 到 359
            info_action = torch.randint(0, 9, (1, 1), dtype=torch.long, device=device)  # 0 到 8
            attack_action = torch.randint(0, 11, (1, 1), dtype=torch.long, device=device)  # 0 到 10
            action_type = torch.randint(0, 3, (1, 1), dtype=torch.long, device=device)  # 0 到 2
            arg1 = torch.randint(0, 465, (1, 1), dtype=torch.long, device=device)  # 0 到 464
            arg2 = torch.randint(0, 100, (1, 1), dtype=torch.long, device=device)  # 0 到 99
            arg3 = torch.randint(0, 5, (1, 1), dtype=torch.long, device=device)  # 0 到 4

            action = torch.cat((move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3), dim=-1)
            return action.cpu().data.numpy().flatten(), move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3
        else:
            tmp_state_640_640 = self.preprocess_image(state).unsqueeze(0)
            move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3 = self.actor(
                tmp_state_640_640)
            action = torch.cat((move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3), dim=-1)
            return action.cpu().data.numpy().flatten(), move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3

    def preprocess_image(self, image, target_size=(640, 640)):
        # 调整图像大小
        resized_image = cv2.resize(image, target_size)
        # 转换为张量并调整维度顺序 [height, width, channels] -> [channels, height, width]
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1)
        return tensor_image.to(device)

    def train(self):
        torch.autograd.set_detect_anomaly(True)

        while True:
            if not globalInfo.is_memory_bigger_batch_size_ppo():
                time.sleep(1)
                continue

            print("ppo training ...")
            transitions = globalInfo.random_batch_size_memory_ppo()
            batch = Transition(*zip(*transitions))

            # 将 batch 中的数据转换为 PyTorch 张量，并确保图像张量的维度正确
            state_batch = torch.stack([self.preprocess_image(state) for state in batch.state]).to(device)
            action_batch = torch.tensor(np.array(batch.action), dtype=torch.float32).to(device)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
            next_state_batch = torch.stack([self.preprocess_image(state) for state in batch.next_state]).to(device)
            done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

            # 计算当前状态的值
            values = self.critic(state_batch, action_batch).squeeze()

            # 计算目标值
            with torch.no_grad():
                next_values = self.critic(next_state_batch, action_batch).squeeze()
            target_values = reward_batch + args.gamma * next_values * (1 - done_batch)

            # 计算优势
            advantages = (target_values - values).unsqueeze(1)  # 将 advantages 转换为二维张量以匹配 ratio 的形状

            # 更新 Actor 和 Critic
            for _ in range(args.ppo_epoch):
                # 计算新的动作概率
                move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3 = self.actor(state_batch)
                new_action_probs = torch.cat(
                    (move_action, angle, info_action, attack_action, action_type, arg1, arg2, arg3), dim=-1)

                # 计算旧动作概率
                with torch.no_grad():
                    old_move_action, old_angle, old_info_action, old_attack_action, old_action_type, old_arg1, old_arg2, old_arg3 = self.actor(
                        state_batch)
                    old_action_probs = torch.cat((old_move_action, old_angle, old_info_action, old_attack_action,
                                                  old_action_type, old_arg1, old_arg2, old_arg3), dim=-1)

                # 计算概率比
                ratio = new_action_probs / (old_action_probs.detach() + 1e-10)

                print(ratio.shape)
                print(advantages.shape)
                print(advantages.unsqueeze(-1))

                # 计算 surrogate loss
                surr1 = ratio * advantages  # 确保 advantages 与 ratio 的形状匹配
                surr2 = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * advantages  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数

                # 计算 Critic 网络的损失
                values = self.critic(state_batch, action_batch)  # 使用 clone() 避免原地操作

                critic_loss = torch.mean(
                    F.mse_loss(values, target_values.detach()))

                self.optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                # 优化 Actor 网络
                actor_loss.backward()
                # 优化 Critic 网络
                critic_loss.backward()

                self.optimizer.step()
                self.critic_optimizer.step()

            self._save_models()

    def start_train(self):
        training_thread = threading.Thread(target=self.train)
        training_thread.daemon = True
        training_thread.start()
