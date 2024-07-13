# config.py
import argparse

import torch

from globalInfo import GlobalInfo

# 移动坐标和滑动半径
move_actions_detail = {
    1: {'action_name': '移动', 'position': (178, 382), 'radius': 150}
}

# 点击坐标
#  购买装备1， 购买装备2，发起进攻，开始撤退，请求集合，升级1技能，升级2技能，升级3技能，升级4技能
info_actions_detail = {
    1: {'action_name': '购买装备1', 'position': (142, 194), 'radius': 0},
    2: {'action_name': '购买装备2', 'position': (142, 248), 'radius': 0},
    3: {'action_name': '发起进攻', 'position': (1000, 68), 'radius': 0},
    4: {'action_name': '开始撤退', 'position': (1000, 108), 'radius': 0},
    5: {'action_name': '请求集合', 'position': (1000, 148), 'radius': 0},
    6: {'action_name': '升级1技能', 'position': (721, 377), 'radius': 0},
    7: {'action_name': '升级2技能', 'position': (774, 288), 'radius': 0},
    8: {'action_name': '升级3技能', 'position': (864, 236), 'radius': 0},
    9: {'action_name': '升级4技能', 'position': (0, 0), 'radius': 0}
}

# 无操作, 攻击，攻击小兵，攻击塔，回城，恢复，装备技能, 1技能，2技能，3技能,
attack_actions_detail = {
    1: {'action_name': '攻击', 'position': (912, 416), 'radius': 0},
    2: {'action_name': '攻击小兵', 'position': (838, 444), 'radius': 0},
    3: {'action_name': '攻击塔', 'position': (950, 346), 'radius': 0},
    4: {'action_name': '回城', 'position': (560, 436), 'radius': 0},
    5: {'action_name': '恢复', 'position': (626, 436), 'radius': 0},
    6: {'action_name': '装备技能', 'position': (906, 190), 'radius': 0},
    7: {'action_name': '召唤师技能', 'position': (906, 190), 'radius': 0},
    8: {'action_name': '1技能', 'position': (768, 430), 'radius': 100},
    9: {'action_name': '2技能', 'position': (820, 336), 'radius': 100},
    10: {'action_name': '3技能', 'position': (910, 284), 'radius': 100}
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=str, default='192.168.0.75', help="device_id")
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

# 全局状态
globalInfo = GlobalInfo(buffer_capacity=args.buffer_capacity)
