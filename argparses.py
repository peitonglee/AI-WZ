# config.py
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=str, default='528e7355', help="device_id")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--tau', type=float, default=0.005, help="Target smoothing coefficient")
    parser.add_argument('--actor_lr', type=float, default=3e-4, help="Learning rate for actor")
    parser.add_argument('--critic_lr', type=float, default=3e-4, help="Learning rate for critic")
    parser.add_argument('--ppo_clip', type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument('--ppo_epoch', type=int, default=10, help="Number of PPO epochs")
    parser.add_argument('--td3_delay', type=int, default=2, help="Delay for TD3 updates")
    parser.add_argument('--td3_noise', type=float, default=0.1, help="Noise for TD3 target policy smoothing")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--buffer_capacity', type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument('--min_buffer_size', type=int, default=1000, help="Minimum buffer size before training starts")
    parser.add_argument('--num_episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--model_dir', type=str, default='./models', help="Directory to save models")
    return parser.parse_args()


# 解析参数并存储在全局变量中
args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
