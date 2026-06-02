import pyrealsense2 as rs
import cv2
import torch
import numpy as np
import os
import sys
import time
import threading
import logging
from scipy.spatial.transform import Rotation as R

# 路径与环境配置
BASE_DIR = os.path.abspath('graspnet-baseline') 
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'dataset'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append("..")

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# 全局变量与线程同步
target_lock = threading.Lock()
global_target_pos = None 
first_grasp_ready = threading.Event() 

T_cam2base = np.load('T_cam_to_base.npy')

# 强化学习策略网络
class Actor(torch.nn.Module):
    def __init__(self, obs_dim=16, act_dim=7):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU()
        )
        self.mu = torch.nn.Linear(128, act_dim)
        self.register_buffer('running_mean', torch.zeros(obs_dim))
        self.register_buffer('running_var', torch.ones(obs_dim))

    def forward(self, x):
        x = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        x = torch.clamp(x, min=-5.0, max=5.0)
        return self.mu(self.mlp(x))

def build_observation(actual_q, actual_qd, target_pos, ur_grasp_pos, last_gripper_state):
    LOWER_LIMITS = torch.tensor([-6.28] * 6, dtype=torch.float32)
    UPPER_LIMITS = torch.tensor([6.28] * 6, dtype=torch.float32)
    
    q_tensor = torch.tensor(actual_q, dtype=torch.float32)
    qd_tensor = torch.tensor(actual_qd, dtype=torch.float32)
    
    q_scaled = (2.0 * (q_tensor - LOWER_LIMITS) / (UPPER_LIMITS - LOWER_LIMITS + 1e-5)) - 1.0
    gripper_binary_state = torch.tensor([last_gripper_state], dtype=torch.float32)
    qd_scaled = qd_tensor * 0.1 
    
    # 计算相对向量 (Target - Current)
    to_target = torch.tensor(target_pos, dtype=torch.float32) - torch.tensor(ur_grasp_pos, dtype=torch.float32) 
    
    obs_tensor = torch.cat([
        q_scaled, gripper_binary_state, 
        qd_scaled,
        to_target
    ], dim=0)
    
    return obs_tensor.unsqueeze(0), to_target

# 坐标转换和工具函数
def transform_cam_to_base(cam_point):
    p_cam = np.array([cam_point[0], cam_point[1], cam_point[2], 1.0])
    p_base = T_cam2base @ p_cam
    return p_base[:3].tolist()

def calculate_grasp_point(tcp_pose, z_offset=0.18):
    tcp_pos = np.array(tcp_pose[0:3])
    return tcp_pos.tolist()

# 视觉处理子线程 (检测5次取最高分)
def vision_loop():
    global global_target_pos
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, is_training=False).to(device)
    net.load_state_dict(torch.load('checkpoint/checkpoint-rs.tar', map_location=device)['model_state_dict'])
    net.eval()

    pipe = rs.pipeline(); cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    pipe.start(cfg); align = rs.align(rs.stream.color); pc = rs.pointcloud()

    # 记录前5次检测的变量
    detection_count = 0
    max_detections = 5
    candidate_grasps = []  # 存储格式: [(score, [x, y, z]), ...]
    
    print("视觉检测器启动成功，开始检测目标坐标...")

    while True:
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        depth = aligned.get_depth_frame(); color = aligned.get_color_frame()
        if not depth or not color: continue

        if detection_count < max_detections:
            pcd_pts = np.asanyarray(pc.calculate(depth).get_vertices()).view(np.float32).reshape(-1, 3)
            filtered = pcd_pts[(pcd_pts[:, 2] > 0.01) & (pcd_pts[:, 2] < 0.25)]
            
            if len(filtered) >= 100:
                indices = np.random.choice(len(filtered), 20000, replace=(len(filtered)<20000))
                input_data = {'point_clouds': torch.from_numpy(filtered[indices][np.newaxis].astype(np.float32)).to(device)}
                
                with torch.no_grad():
                    gg = GraspGroup(pred_decode(net(input_data))[0].detach().cpu().numpy())
                    gg.nms(); gg.sort_by_score()
                    
                    if len(gg) > 0:
                        # 提取当前帧中分数最高的一个抓取点
                        best_score = float(gg[0].score)
                        base_translation = transform_cam_to_base(gg[0].translation)
                        
                        candidate_grasps.append((best_score, base_translation))
                        detection_count += 1
                        print(f"[GraspNet] 获取到第 {detection_count}/5 个目标候选，置信度: {best_score:.4f}")

                        # 当收集满5次时，进行最控制决策
                        if detection_count == max_detections:
                            # 按照 score 降序排序 (x[0] 是 score)
                            candidate_grasps.sort(key=lambda x: x[0], reverse=True)
                            best_grasp = candidate_grasps[0] # 取出最高分的那一组
                            
                            with target_lock:
                                global_target_pos = best_grasp[1]
                            
                            print(f"\n>>> [GraspNet] 目标检测完成！已选取最高分目标 <<<")
                            print(f">>> 最高置信度: {best_grasp[0]:.4f}")
                            print(f">>> 最终锁定坐标: {[round(p, 4) for p in global_target_pos]}\n")
                            
                            # 唤醒主线程，允许机械臂开始移动
                            if not first_grasp_ready.is_set(): 
                                first_grasp_ready.set()
                                
        cv2.imshow('VisionMonitor', np.asanyarray(color.get_data()))
        if cv2.waitKey(1) & 0xFF == ord('q'): # 添加了按 'q' 退出的保护
            break

# RTDE通信策略控制主循环
def main_control_loop():
    ROBOT_HOST = "192.168.56.101"
    conf = rtde_config.ConfigFile("control_loop_configuration.xml")
    state_names, state_types = conf.get_recipe("state")
    setp_names, setp_types = conf.get_recipe("setp")
    watchdog_names, watchdog_types = conf.get_recipe("watchdog")

    con = rtde.RTDE(ROBOT_HOST, 30004); con.connect()
    con.send_output_setup(state_names, state_types, 60)
    setp = con.send_input_setup(setp_names, setp_types)
    watchdog = con.send_input_setup(watchdog_names, watchdog_types)
    con.send_start()

    # 模型加载与权重解析
    checkpoint = torch.load("UR5Grasp/nn/last_UR_ep_350_rew_63449.66.pth", map_location="cpu")
    model = Actor(obs_dim=16, act_dim=7)
    ms = checkpoint['model']
    sd = {
        'mlp.0.weight': ms['a2c_network.actor_mlp.0.weight'], 'mlp.0.bias': ms['a2c_network.actor_mlp.0.bias'],
        'mlp.2.weight': ms['a2c_network.actor_mlp.2.weight'], 'mlp.2.bias': ms['a2c_network.actor_mlp.2.bias'],
        'mlp.4.weight': ms['a2c_network.actor_mlp.4.weight'], 'mlp.4.bias': ms['a2c_network.actor_mlp.4.bias'],
        'mu.weight': ms['a2c_network.mu.weight'], 'mu.bias': ms['a2c_network.mu.bias'],
        'running_mean': ms['running_mean_std.running_mean'], 'running_var': ms['running_mean_std.running_var']
    }
    model.load_state_dict(sd); model.eval()

    current_gripper_sim = 0.0
    prev_gripper_sim = 0.0
    last_gripper_state = 0.0
    dt = 1/60.0
    last_print_time = time.time()

    while True:
        state = con.receive()
        if state is None: break

        # 握手逻辑，在机器人准备好时计算并下发
        if state.output_int_register_0 == 1:
            with target_lock: 
                target_p = list(global_target_pos)
                manual_offset = np.array([-0.16, 0.06, 0.04])
                target_p = (target_p + manual_offset).tolist()
            ur_p = calculate_grasp_point(state.actual_TCP_pose)
            
            # 物理反馈计算
            gripper_vel_sim = (current_gripper_sim - prev_gripper_sim) / dt
            prev_gripper_sim = current_gripper_sim

            # 构建观测并获取相对向量
            obs, to_target_vec = build_observation(state.actual_q, state.actual_qd, target_p, ur_p, last_gripper_state)
            
            with torch.no_grad(): 
                action = model(obs).squeeze(0).numpy()
            
            # 目标关节与夹爪计算
            action_multiplier = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
            target_q = np.array(state.actual_q) + dt * (action[0:6] * action_multiplier) * 7.5
            target_q = np.array(state.actual_q) + np.clip(target_q - np.array(state.actual_q), -0.05, 0.05)
            
            if action[6] > 0.0:
                last_gripper_state = 1.0
                real_cmd = 255.0  # 发给真实夹爪的完全闭合指令
            else:
                last_gripper_state = 0.0
                real_cmd = 0.0    # 发给真实夹爪的完全张开指令

            # 调试信息
            if time.time() - last_print_time > 0.5:
                dist = np.linalg.norm(to_target_vec.numpy())
                print(f"\n" + "="*50)
                print(f"[目标状态] 目标Z轴: {target_p[2]:.3f} | 剩余距离: {dist:.4f}m")
                print(f"[绝对坐标] 目标物体 (X,Y,Z): {[round(p, 4) for p in target_p]}")
                print(f"[绝对坐标] 当前夹爪 (X,Y,Z): {[round(p, 4) for p in ur_p]}")
                print(f"[相对向量] to_target (X,Y,Z): {np.round(to_target_vec.numpy(), 4)}")
                print(f"[控制下发] 关节(度): {[round(np.rad2deg(i), 1) for i in target_q]}")
                print(f"[控制下发] 夹爪值: {int(real_cmd)} | Action[6]: {round(action[6], 3)}")
                print("="*50)
                last_print_time = time.time()

            # 发送寄存器数据
            for i in range(6): setp.__dict__[f"input_double_register_{i}"] = target_q[i]
            setp.input_double_register_6 = float(real_cmd)
            con.send(setp)

            # 触发 URP 脚本中的 Move 逻辑
            watchdog.input_int_register_0 = 1
            con.send(watchdog)
            
        elif state.output_int_register_0 == 0:
            watchdog.input_int_register_0 = 0
            con.send(watchdog)

if __name__ == '__main__':
    threading.Thread(target=vision_loop, daemon=True).start()
    print("[系统] 等待视觉识别...")
    first_grasp_ready.wait()
    main_control_loop()
    