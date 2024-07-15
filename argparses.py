# config.py
import argparse

import torch

from globalInfo import GlobalInfo

# 移动坐标和滑动半径
move_actions_detail = {
    1: {'action_name': '移动', 'position': (0.164, 0.798), 'radius': 200}
}

# 点击坐标
#  购买装备1， 购买装备2，发起进攻，开始撤退，请求集合，升级1技能，升级2技能，升级3技能，升级4技能
info_actions_detail = {
    1: {'action_name': '购买装备1', 'position': (0.133, 0.4), 'radius': 0},
    2: {'action_name': '购买装备2', 'position': (0.133, 0.51), 'radius': 0},
    3: {'action_name': '发起进攻', 'position': (0.926, 0.14), 'radius': 0},
    4: {'action_name': '开始撤退', 'position': (0.926, 0.22), 'radius': 0},
    5: {'action_name': '请求集合', 'position': (0.926, 0.31), 'radius': 0},
    6: {'action_name': '升级1技能', 'position': (0.668, 0.772), 'radius': 0},
    7: {'action_name': '升级2技能', 'position': (0.717, 0.59), 'radius': 0},
    8: {'action_name': '升级3技能', 'position': (0.8, 0.48), 'radius': 0}
}

# 无操作, 攻击，攻击小兵，攻击塔，回城，恢复，装备技能, 1技能，2技能，3技能,
attack_actions_detail = {
    1: {'action_name': '攻击', 'position': (0.85, 0.85), 'radius': 0},
    2: {'action_name': '攻击小兵', 'position': (0.776, 0.91), 'radius': 0},
    3: {'action_name': '攻击塔', 'position': (0.88, 0.71), 'radius': 0},
    4: {'action_name': '回城', 'position': (0.518, 0.9), 'radius': 0},
    5: {'action_name': '恢复', 'position': (0.579, 0.9), 'radius': 0},
    6: {'action_name': '装备技能', 'position': (0.84, 0.39), 'radius': 0},
    7: {'action_name': '召唤师技能', 'position': (0.64, 0.9), 'radius': 50},
    8: {'action_name': '1技能', 'position': (0.71, 0.874), 'radius': 100},
    9: {'action_name': '2技能', 'position': (0.76, 0.69), 'radius': 100},
    10: {'action_name': '3技能', 'position': (0.844, 0.58), 'radius': 100}
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=str, default='528e7355', help="device_id")
    parser.add_argument('--window_title', type=str, default='wzry_ai', help="device_id")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--tau', type=float, default=0.005, help="Target smoothing coefficient")
    parser.add_argument('--actor_lr', type=float, default=3e-4, help="Learning rate for actor")
    parser.add_argument('--critic_lr', type=float, default=3e-4, help="Learning rate for critic")
    parser.add_argument('--ppo_clip', type=float, default=0.2, help="Clipping parameter for PPO")
    parser.add_argument('--ppo_epoch', type=int, default=10, help="Number of PPO epochs")
    parser.add_argument('--td3_delay', type=int, default=2, help="Delay for TD3 updates")
    parser.add_argument('--td3_noise', type=float, default=0.1, help="Noise for TD3 target policy smoothing")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--buffer_capacity', type=int, default=100000, help="Replay buffer capacity")
    parser.add_argument('--min_buffer_size', type=int, default=1000, help="Minimum buffer size before training starts")
    parser.add_argument('--num_episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--model_dir', type=str, default='./models', help="Directory to save models")
    return parser.parse_args()


# 解析参数并存储在全局变量中
args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局状态
globalInfo = GlobalInfo(batch_size=args.batch_size, buffer_capacity=args.buffer_capacity)
