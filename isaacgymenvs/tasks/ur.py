import os
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import math
import numpy as np
import os
import torch
import gym
from gym import spaces

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_apply
from .base.vec_task import VecTask

import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms.functional as TF

# UR 任务环境，封装了 UR 机械臂与柜子交互的仿真环境
class UR(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # 环境配置参数 (Environment Configuration)
        self.cfg = cfg

        self.enable_camera_sensors = self.cfg["env"].get("enableCameraSensors", False)

        self.start_pos = self.cfg["env"]["startPos"]
        self.start_rot = self.cfg["env"]["startRot"]

        self.cab_start_pos = self.cfg["env"]["cabinetStartPos"]

        self.max_bound = self.cfg["env"]["maxBound"]
        self.min_bound = self.cfg["env"]["minBound"]

        self.cam_w = self.cfg["env"]["cameraWidth"]
        self.cam_h = self.cfg["env"]["cameraHeight"]
        self.fov = self.cfg["env"]["fov"]

        self.intrinsics = {
            'cx': self.cam_w / 2.0,
            'cy': self.cam_h / 2.0,
            'fx': self.cam_w / (2.0 * math.tan(self.fov * math.pi / 360.0)),
            'fy': self.cam_w / (2.0 * math.tan(self.fov * math.pi / 360.0))
        }

        self.cam_pos = self.cfg["env"]["cameraPos"]
        self.cam_target = self.cfg["env"]["cameraTarget"]

        # 最大回合长度：如果步数超过此值，环境将重置 (episodeLength)
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        # 动作缩放系数：将策略网络(Policy)输出的归一化数值转换为实际应用到关节的目标值
        self.action_scale = self.cfg["env"]["actionScale"]
        
        # 初始噪声：在 Reset 时添加到物体位置和旋转上的随机噪声，用于增加环境泛化能力
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        
        # 道具数量：柜子抽屉里放置的小方块数量
        self.num_props = self.cfg["env"]["numProps"]
        
        # 聚合模式
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # 关节速度缩放：用于在 Observation 中归一化速度数据
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        
        # 距离奖励权重：鼓励 ur 夹爪靠近柜子把手
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]

        # 对齐奖励权重
        self.align_reward_scale = self.cfg["env"]["alignRewardScale"]
        
        # 手指距离奖励权重
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        
        # 动作惩罚权重，用于惩罚过大的动作幅度
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        # 禁区惩罚权重
        self.restricted_zone_reward_scale = self.cfg["env"]["restrictedZoneRewardScale"]

        # 调试可视化
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # 定义环境的“上方”是 Z 轴 (索引为 2)
        self.up_axis = "z"
        self.up_axis_idx = 2

        # 距离偏移量：在计算奖励时防止机械臂以不正确的姿态(如穿模)接近把手
        self.distX_offset = 0.04
        # 仿真时间步长 (60Hz)
        self.dt = 1/60.

        # 道具(Prop)的物理尺寸参数
        self.prop_width = 0.04
        self.prop_height = 0.04
        self.prop_length = 0.04
        self.prop_spacing = 0.09 # 道具排列间距

        num_obs = 7 + 6 + 3 
        num_acts = 7

        self.cfg["env"]["numObservations"] = num_obs

        self.cfg["env"]["numActions"] = num_acts

        self.cfg["env"]["numStates"] = num_obs

        # 初始化父类 VecTask，启动 Isaac Gym 仿真引擎
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)

        self.env_success_history = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.env_align_err_history = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.env_action_jitter_history = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # 记录所有 9 个长方体物体的初始全局坐标 [num_envs, 9, 3]
        self.all_prop_init_pos = torch.zeros((self.num_envs, self.num_props, 3), device=self.device, dtype=torch.float)

        # 记录每个环境当前黄色的目标物体索引
        self.target_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)        

        if not self.headless:
            # 设置相机位置 (Camera Position)
            cam_pos = gymapi.Vec3(*self.cam_pos)

            cam_target = gymapi.Vec3(*self.cam_target)

            center_env_idx = self.num_envs // 2

            self.gym.viewer_camera_look_at(self.viewer, self.envs[center_env_idx], cam_pos, cam_target)

        self.start_pos = torch.tensor(self.cfg["env"]["startPos"], device=self.device)
        self.start_rot = torch.tensor(self.cfg["env"]["startRot"], device=self.device)

        self.extract_offset = torch.tensor(self.cfg["env"]["extractOffset"], device=self.device)

        # 获取 Actor 根节点状态张量 (包含全局位置、旋转、线速度、角速度)
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # 获取关节(DOF)状态张量 (包含关节位置、速度)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # 获取刚体(Rigid Body)状态张量 (包含每个刚体组件的状态)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # 刷新张量数据，确保拿到最新状态
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # ur 机器人的默认复位关节角度
        self.ur_default_dof_pos = to_torch([0, -1.5708, 1.5708, -1.5708, -1.5708, 0, 0, 0, 0, 0, 0, 0], device=self.device)
        
        # 封装关节状态张量
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        # 提取 ur 机器人的关节状态 (前 num_ur_dofs 个)
        self.ur_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur_dofs]
        self.ur_dof_pos = self.ur_dof_state[..., 0] # ur 关节位置
        self.ur_dof_vel = self.ur_dof_state[..., 1] # ur 关节速度
        
        # 提取柜子(Cabinet)的关节状态 (ur 之后的关节)
        self.cabinet_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_ur_dofs:]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0] # 柜子关节位置 (主要关注抽屉滑动关节)
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1] # 柜子关节速度

        # 封装刚体状态张量
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        # 封装根节点状态张量
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        # 获取当前每局随机指定的黄色目标物体的全局坐标
        batch_indices = torch.arange(self.num_envs, device=self.device)
        self.init_target_pos = self.root_state_tensor[batch_indices, 2 + self.target_indices, 0:3]
        self.target_pos = self.root_state_tensor[batch_indices, 2 + self.target_indices, 0:3]

        # 如果有道具，提取道具的状态 (索引2以后，因为0是ur，1是Cabinet)
        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        
        self.ur_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        num_actors_per_env = 2 + self.num_props
        self.global_indices = torch.arange(self.num_envs * num_actors_per_env, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.global_step_counter = 0

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        self.ep_success_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.ep_align_err_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # 标记每个环境在当前全局回合是否已经出过成绩
        self.ep_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        # 创建地面平面
        self._create_ground_plane()
        
        # 创建具体的环境 (Environments)，包含机器人、柜子等
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        # 创建地面平面参数对象
        plane_params = gymapi.PlaneParams()
        
        # 设置平面的法向量 (Normal Vector)，(0, 0, 1) 表示平面垂直于 Z 轴，正面朝上
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        
        # 将地面添加到仿真中
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # 定义每个环境的边界范围 (用于物理引擎的空间划分)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # URDF根目录路径
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")

        # 默认的 ur 机器人 URDF 文件路径
        ur_asset_file = "urdf/ur_description/robots/ur5e.urdf"
        # 默认的柜子 (Cabinet) URDF 文件路径
        cabinet_asset_file = "urdf/cabinet_model/urdf/cabinet.urdf"

        # 如果配置文件中有自定义资源路径，则覆盖默认值
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            ur_asset_file = self.cfg["env"]["asset"].get("assetFileNameUR", ur_asset_file)
            cabinet_asset_file = self.cfg["env"]["asset"].get("assetFileNameCabinet", cabinet_asset_file)

        # 加载 asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        # 从文件加载 ur asset
        ur_asset = self.gym.load_asset(self.sim, asset_root, ur_asset_file, asset_options)

        ur_dof_dict = self.gym.get_asset_dof_dict(ur_asset)
        self.idx_master = ur_dof_dict["robotiq_85_left_knuckle_joint"]
        self.idx_r_knuckle = ur_dof_dict["robotiq_85_right_knuckle_joint"]
        self.idx_l_inner = ur_dof_dict["robotiq_85_left_inner_knuckle_joint"]
        self.idx_r_inner = ur_dof_dict["robotiq_85_right_inner_knuckle_joint"]
        self.idx_l_tip = ur_dof_dict["robotiq_85_left_finger_tip_joint"]
        self.idx_r_tip = ur_dof_dict["robotiq_85_right_finger_tip_joint"]

        # 加载柜子 asset
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, asset_options)

        # 定义 ur 关节的刚度 (Stiffness/Kp) 和阻尼 (Damping/Kd)
        ur_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        ur_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # 获取各资产的刚体数量和关节数量
        self.num_ur_bodies = self.gym.get_asset_rigid_body_count(ur_asset)
        self.num_ur_dofs = self.gym.get_asset_dof_count(ur_asset)
        self.num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        self.num_cabinet_dofs = self.gym.get_asset_dof_count(cabinet_asset)

        print("num ur bodies: ", self.num_ur_bodies)
        print("num ur dofs: ", self.num_ur_dofs)
        print("num cabinet bodies: ", self.num_cabinet_bodies)
        print("num cabinet dofs: ", self.num_cabinet_dofs)

        # 设置 ur 关节属性
        ur_dof_props = self.gym.get_asset_dof_properties(ur_asset)
        self.ur_dof_lower_limits = []
        self.ur_dof_upper_limits = []
        for i in range(self.num_ur_dofs):
            ur_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                ur_dof_props['stiffness'][i] = ur_dof_stiffness[i]
                ur_dof_props['damping'][i] = ur_dof_damping[i]
            else:
                ur_dof_props['stiffness'][i] = 7000.0
                ur_dof_props['damping'][i] = 50.0

            # 记录关节的上下限位
            self.ur_dof_lower_limits.append(ur_dof_props['lower'][i])
            self.ur_dof_upper_limits.append(ur_dof_props['upper'][i])

        self.ur_dof_lower_limits = to_torch(self.ur_dof_lower_limits, device=self.device)
        self.ur_dof_upper_limits = to_torch(self.ur_dof_upper_limits, device=self.device)
        # 关节速度缩放
        self.ur_dof_speed_scales = torch.ones_like(self.ur_dof_lower_limits)
        # 增加夹爪的最大出力
        ur_dof_props['effort'][7] = 200
        ur_dof_props['effort'][8] = 200

        for finger_idx in range(6, 12):
            ur_dof_props['effort'][finger_idx] = 200.0  # 提高夹持力上限
            ur_dof_props['stiffness'][finger_idx] = 5000
            ur_dof_props['damping'][finger_idx] = 1.0e2

        # 设置柜子关节属性
        cabinet_dof_props = self.gym.get_asset_dof_properties(cabinet_asset)
        for i in range(self.num_cabinet_dofs):
            cabinet_dof_props['damping'][i] = 10.0

        # 创建道具 (Prop) asset
        box_opts = gymapi.AssetOptions()
        box_opts.density = 4000 # 密度
        prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        # 定义 ur 的初始位姿 (Pose)
        ur_start_pose = gymapi.Transform()
        ur_start_pose.p = gymapi.Vec3(*self.start_pos) # 位置 (1.0, 0, 0)
        ur_start_pose.r = gymapi.Quat(*self.start_rot) # 旋转 (绕Y轴旋转180度，使机器人面向柜子)

        # 定义柜子的初始位姿
        cabinet_start_pose = gymapi.Transform()
        cabinet_start_pose.p = gymapi.Vec3(*self.cab_start_pos)

        # 计算聚合所需的总刚体和形状数量 (Optimization)
        num_ur_bodies = self.gym.get_asset_rigid_body_count(ur_asset)
        num_ur_shapes = self.gym.get_asset_rigid_shape_count(ur_asset)
        num_cabinet_bodies = self.gym.get_asset_rigid_body_count(cabinet_asset)
        num_cabinet_shapes = self.gym.get_asset_rigid_shape_count(cabinet_asset)
        num_prop_bodies = self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = self.gym.get_asset_rigid_shape_count(prop_asset)
        # 计算每个环境需要的最大聚合数
        max_agg_bodies = num_ur_bodies + num_cabinet_bodies + self.num_props * num_prop_bodies
        max_agg_shapes = num_ur_shapes + num_cabinet_shapes + self.num_props * num_prop_shapes

        self.urs = []
        self.cabinets = []
        self.default_prop_states = []
        self.prop_start = []
        self.camera_handles = []
        self.camera_tensors = []
        self.camera_rgb_tensors = []
        self.camera_depth_tensors = []
        self.camera_seg_tensors = []
        self.envs = []

        # 创建环境实例
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # 开启聚合 (Aggregate Mode >= 3): 聚合整个环境中的所有 Actor
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            env_ur_pose = gymapi.Transform()
            
            # 使用现有的 start_position_noise 参数，在 X 和 Y 轴方向添加噪声
            # Z轴保持原样，以确保机械臂仍然贴在地面上
            env_ur_pose.p.x = ur_start_pose.p.x + self.start_position_noise * (np.random.rand() - 0.5)
            env_ur_pose.p.y = ur_start_pose.p.y + self.start_position_noise * (np.random.rand() - 0.5)
            env_ur_pose.p.z = ur_start_pose.p.z
            
            env_ur_pose.r = ur_start_pose.r
            
            # 创建 ur Actor
            ur_actor = self.gym.create_actor(env_ptr, ur_asset, env_ur_pose, "ur", i, 1, 0)

            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, ur_actor)
            for shape in shape_props:
                shape.friction = 2.0        # 极高的静摩擦力
                shape.rolling_friction = 0.01 
                shape.restitution = 0.0     # 毫无弹性
            self.gym.set_actor_rigid_shape_properties(env_ptr, ur_actor, shape_props)

            # 应用设置好的关节属性 
            self.gym.set_actor_dof_properties(env_ptr, ur_actor, ur_dof_props)

            # 开启聚合 (Aggregate Mode == 2): 不聚合 ur，只聚合后面的
            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 创建柜子 Actor
            cabinet_pose = gymapi.Transform()
            # 对柜子位置添加随机噪声
            cabinet_pose.p.x = cabinet_start_pose.p.x + self.start_position_noise * (np.random.rand() - 0.5)
            cabinet_pose.p.y = cabinet_start_pose.p.y + self.start_position_noise * (np.random.rand() - 0.5)
            cabinet_pose.p.z = cabinet_start_pose.p.z  
            cabinet_pose.r = cabinet_start_pose.r

            cabinet_actor = self.gym.create_actor(env_ptr, cabinet_asset, cabinet_pose, "cabinet", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, cabinet_actor, cabinet_dof_props)

            # 开启聚合 (Aggregate Mode == 1): 只聚合道具
            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 创建道具 (Props)
            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))

                # 计算道具排列的网格布局
                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = 0.5 * self.prop_spacing * (props_per_row - 1)
                yzmin = 0.5 * self.prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    prop_up = yzmin + j * self.prop_spacing
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        propx = xmin + k * self.prop_spacing
                        prop_state_pose = gymapi.Transform()

                        # 第一层：0.02， 第二层：0.31， 第三层0.61
                        tiers = [0.02, 0.31, 0.61]
                        y_offsets = [-0.2, 0.0, 0.2]
                        x_offsets = 0.1

                        # 基于抽屉的位置计算道具的相对位置，使其位于抽屉内
                        prop_state_pose.p.x = cabinet_pose.p.x + x_offsets
                        propz, propy = 0, prop_up
                        prop_state_pose.p.y = cabinet_pose.p.y + y_offsets[k]
                        prop_state_pose.p.z = cabinet_pose.p.z + tiers[j] + (self.prop_height / 2.0) + 0.01

                        prop_state_pose.r = gymapi.Quat(0, 0, 0, 1) # 无旋转
                        # 创建道具 Actor
                        prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)

                        # 道具摩擦力
                        prop_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, prop_handle)
                        prop_shape_props[0].friction = 2.0
                        prop_shape_props[0].restitution = 0.0
                        self.gym.set_actor_rigid_shape_properties(env_ptr, prop_handle, prop_shape_props)

                        # 给物体随即上色
                        color = gymapi.Vec3(np.random.uniform(0.2, 1.0), np.random.uniform(0.2,1.0), np.random.uniform(0.2,1.0))
                        self.gym.set_rigid_body_color(env_ptr, prop_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

                        prop_count += 1

                        # 记录道具的默认状态，用于 Reset
                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x, prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # 创建摄像头
            if self.enable_camera_sensors:
                camera_props = gymapi.CameraProperties()
                camera_props.horizontal_fov = self.fov
                camera_props.width = self.cam_w
                camera_props.height = self.cam_h
                camera_props.enable_tensors = True

                camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

                # 设置相机观测位置
                cam_pos = gymapi.Vec3(*self.cam_pos)

                cam_target = gymapi.Vec3(*self.cam_target)
                self.gym.set_camera_location(camera_handle, env_ptr, cam_pos, cam_target)
                self.camera_handles.append(camera_handle)

            # # 手腕相机
            # # 找到机械臂末端手腕的刚体句柄
            # body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur_actor, "wrist_3_link")

            # # 设置相机相对于手腕的局部偏移位姿
            # local_transform = gymapi.Transform()
            
            # # 位置偏移 (x, y, z)，避免相机嵌在模型内部或被完全遮挡
            # local_transform.p = gymapi.Vec3(0.0, -0.08, 0.05) 
            
            # # 旋转偏移：让相机镜头对准夹爪的正前方
            # angle = np.radians(-90)
            # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), angle)
            
            # # 将相机绑定在刚体上
            # self.gym.attach_camera_to_body(
            #     camera_handle, 
            #     env_ptr, 
            #     body_handle, 
            #     local_transform, 
            #     gymapi.FOLLOW_TRANSFORM
            # )

            # self.camera_handles.append(camera_handle)

            self.envs.append(env_ptr)
            self.urs.append(ur_actor)

        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur_actor, "robotiq_85_left_finger_tip_link")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur_actor, "robotiq_85_right_finger_tip_link")

        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)
            
        ee_link_name = "wrist_3_link"

        self.hand_handle = self.gym.find_actor_rigid_body_index(
            env_ptr, 
            ur_actor, 
            ee_link_name, 
            gymapi.DOMAIN_ENV
        )

        if self.hand_handle == -1:
            print(f"Error: Could not find link {ee_link_name}")

        self.init_data()

        self.last_actions = torch.zeros((self.num_envs, self.cfg["env"]["numActions"]), device=self.device, dtype=torch.float)
        self.episode_action_jitter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_action_mag = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def init_data(self):
        hand_name = "ee_link"

        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.urs[0], hand_name)
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.urs[0], "robotiq_85_left_finger_tip_link")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.urs[0], "robotiq_85_right_finger_tip_link")

        # 获取初始位姿 (Global Pose)
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        ur_local_grasp_pose = gymapi.Transform()
        
        ur_local_grasp_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        ur_local_grasp_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # 无额外局部旋转

        self.ur_local_grasp_pos = to_torch([ur_local_grasp_pose.p.x, ur_local_grasp_pose.p.y,
                                                ur_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.ur_local_grasp_rot = to_torch([ur_local_grasp_pose.r.x, ur_local_grasp_pose.r.y,
                                                ur_local_grasp_pose.r.z, ur_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        # 初始化用于存储运行时抓取点状态的 buffer (全零初始化)
        self.ur_grasp_pos = torch.zeros_like(self.ur_local_grasp_pos)
        self.ur_grasp_rot = torch.zeros_like(self.ur_local_grasp_rot)
        self.ur_grasp_rot[..., -1] = 1  # xyzw 中的 w 初始化为 1 (单位四元数)

        self.ur_lfinger_pos = torch.zeros_like(self.ur_local_grasp_pos)
        self.ur_rfinger_pos = torch.zeros_like(self.ur_local_grasp_pos)
        self.ur_lfinger_rot = torch.zeros_like(self.ur_local_grasp_rot)
        self.ur_rfinger_rot = torch.zeros_like(self.ur_local_grasp_rot)
        
        # 获取 wrist_3_link 在 UR Actor 内部的局部索引 (Domain 为 ACTOR)
        self.ee_j_idx = self.gym.find_actor_rigid_body_index(
            self.envs[0], self.urs[0], "wrist_3_link", gymapi.DOMAIN_ACTOR)

        # 记录相机安装位置
        self.cam_ideal_pos_tensor = torch.tensor(self.cam_pos, device=self.device).repeat((self.num_envs, 1))
        self.cam_pos_err = torch.zeros((self.num_envs, 3), device=self.device)
        self.cam_rot_err = torch.zeros((self.num_envs, 4), device=self.device)
        self.cam_rot_err[:, 3] = 1.0 # 默认单位四元数 (xyzw 中的 w=1)


        inv_view_matrices = []
        if self.enable_camera_sensors:
            cam_pos_noise = 0.015    # 位置噪声：比如 +/- 1.5 cm 的安装位置误差
            cam_target_noise = 0.03  # 瞄准点噪声：比如 +/- 3 cm，相当于给视角(旋转)引入误差

            for i in range(self.num_envs):
                ideal_pos = gymapi.Vec3(*self.cam_pos)
                ideal_target = gymapi.Vec3(*self.cam_target)
                self.gym.set_camera_location(self.camera_handles[i], self.envs[i], ideal_pos, ideal_target)
                
                # 获取 OpenGL 标准的理想视图矩阵
                view_matrix = self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.camera_handles[i])
                view_tensor = torch.tensor(view_matrix, device=self.device, dtype=torch.float32)
                inv_view_matrices.append(torch.inverse(view_tensor))

                rgb_ptr = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
                self.camera_rgb_tensors.append(gymtorch.wrap_tensor(rgb_ptr))

                depth_ptr = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH)
                self.camera_depth_tensors.append(gymtorch.wrap_tensor(depth_ptr))

                seg_ptr = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_SEGMENTATION)
                self.camera_seg_tensors.append(gymtorch.wrap_tensor(seg_ptr))

                noised_pos = gymapi.Vec3(
                    self.cam_pos[0] + np.random.uniform(-cam_pos_noise, cam_pos_noise),
                    self.cam_pos[1] + np.random.uniform(-cam_pos_noise, cam_pos_noise),
                    self.cam_pos[2] + np.random.uniform(-cam_pos_noise, cam_pos_noise)
                )
                noised_target = gymapi.Vec3(
                    self.cam_target[0] + np.random.uniform(-cam_target_noise, cam_target_noise),
                    self.cam_target[1] + np.random.uniform(-cam_target_noise, cam_target_noise),
                    self.cam_target[2] + np.random.uniform(-cam_target_noise, cam_target_noise)
                )
                # 重新应用带噪声的位置到物理引擎中
                self.gym.set_camera_location(self.camera_handles[i], self.envs[i], noised_pos, noised_target)
            
            self.inv_view_matrices = torch.stack(inv_view_matrices)

            v_grid, u_grid = torch.meshgrid(
                torch.arange(self.cam_h, device=self.device, dtype=torch.float32), 
                torch.arange(self.cam_w, device=self.device, dtype=torch.float32), 
                indexing='ij'
            )
            self.u_grid = u_grid.expand(self.num_envs, -1, -1)
            self.v_grid = v_grid.expand(self.num_envs, -1, -1)
            
            # 初始化全局视觉目标张量
            self.target_pos_vision = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

    def compute_reward(self, actions):
        all_prop_pos = self.root_state_tensor[:, 2:11, 0:3]
        
        self.rew_buf[:], self.reset_buf[:], self.is_success = compute_ur_reward(
            self.reset_buf, self.progress_buf, self.actions, 
            self.hand_pos,
            self.extract_offset,
            self.ur_grasp_pos, self.ur_grasp_rot, 
            self.ur_lfinger_pos, self.ur_rfinger_pos, 
            all_prop_pos, self.all_prop_init_pos, 
            self.target_indices, # 传入这局的目标索引
            self.action_penalty_scale, 
            dist_reward_scale=self.dist_reward_scale, 
            align_reward_scale=self.align_reward_scale, 
            grasp_reward_scale=2.0, 
            lift_reward_scale=10.0, 
            finger_dist_reward_scale=self.finger_dist_reward_scale, 
            max_episode_length=self.max_episode_length
        )


    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.enable_camera_sensors:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
                
            seg_tensors = torch.stack(self.camera_seg_tensors).squeeze()
            depth_tensors = torch.stack(self.camera_depth_tensors).squeeze()

            # 生成目标掩码并计算面积
            target_id = 255
            mask = (seg_tensors == target_id).float()
            mask = mask.view(self.num_envs, self.cam_h, self.cam_w)
            area = mask.sum(dim=(1, 2))  # Shape: (num_envs,)

            valid_mask = area > 0
            safe_area = torch.where(valid_mask, area, torch.ones_like(area))

            # 计算质心 (U, V)
            u_center = (self.u_grid * mask).sum(dim=(1, 2)) / safe_area
            v_center = (self.v_grid * mask).sum(dim=(1, 2)) / safe_area

            masked_depth = depth_tensors * mask
            valid_depth_pixels = (masked_depth < 0).float() * (masked_depth > -20.0).float() * mask
            valid_depth_area = valid_depth_pixels.sum(dim=(1, 2))
            safe_depth_area = torch.where(valid_depth_area > 0, valid_depth_area, torch.ones_like(valid_depth_area))

            safe_masked_depth = torch.where(valid_depth_pixels > 0, masked_depth, torch.zeros_like(masked_depth))
            
            mean_depth = (safe_masked_depth * valid_depth_pixels).sum(dim=(1, 2)) / safe_depth_area
            z_dist = torch.abs(mean_depth)

            # 批量 3D 反投影
            cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
            fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
            
            x_c = (u_center - cx) * z_dist / fx
            y_c = -(v_center - cy) * z_dist / fy
            z_c = -z_dist

            # 组合齐次坐标 p_cam, Shape: (num_envs, 4)
            p_cam = torch.stack([x_c, y_c, z_c, torch.ones_like(z_c)], dim=1)

            p_world = torch.bmm(p_cam.unsqueeze(1), self.inv_view_matrices).squeeze(1)
            vision_3d_pos = p_world[:, :3] # 取前三维 X, Y, Z

            self.target_pos_vision = torch.where(
                valid_mask.unsqueeze(1), 
                vision_3d_pos, 
                self.target_pos_vision  # 如果没看到，保持上一帧的位置
            )

            # 提取第 0 个环境的 RGB 图像
            cam_tensor = self.camera_rgb_tensors[0] 
            cam_img = cam_tensor.cpu().numpy()

            self.gym.end_access_image_tensors(self.sim)
            cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGBA2BGR)
            cam_img_display = cv2.resize(cam_img_bgr, (512, 512), interpolation=cv2.INTER_NEAREST)

            # 实时显示画面
            cv2.imshow("Env 0 Camera View", cam_img_display)
            cv2.waitKey(1)

        # 提取关键刚体 (Rigid Body) 的状态
        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.ur_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.ur_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.ur_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.ur_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.ur_grasp_pos = (self.ur_lfinger_pos + self.ur_rfinger_pos) * 0.5
        
        self.ur_grasp_rot = hand_rot

        dof_pos_scaled = (2.0 * (self.ur_dof_pos - self.ur_dof_lower_limits)
                          / (self.ur_dof_upper_limits - self.ur_dof_lower_limits + 1e-5) - 1.0)
        
        # 获取当前每局随机指定的黄色目标物体的全局坐标
        batch_indices = torch.arange(self.num_envs, device=self.device)
        self.target_pos = self.root_state_tensor[batch_indices, 2 + self.target_indices, 0:3]
        
        if self.enable_camera_sensors:
            # 计算从抓取点指向目标的相对向量 (3维)
            to_target = self.target_pos_vision - self.ur_grasp_pos
        else:
            real_cam_pos = self.cam_ideal_pos_tensor + self.cam_pos_err
            rel_pos = self.target_pos - real_cam_pos
                
            # 施加相机的旋转安装偏差
            perceived_rel_pos = quat_apply(self.cam_rot_err, rel_pos)
            vision_3d_pos = self.cam_ideal_pos_tensor + perceived_rel_pos

            noise_scale = 0.005        
            # 高斯噪声
            dynamic_noise = torch.randn_like(vision_3d_pos) * noise_scale
        
            self.noisy_target_pos = vision_3d_pos + dynamic_noise

            to_target = self.noisy_target_pos - self.ur_grasp_pos

        gripper_pos = dof_pos_scaled[:, self.idx_master]
        # > 0.0 视为闭合状态(1.0)，<= 0.0 视为张开状态(0.0)
        gripper_binary_state = (gripper_pos > 0.0).float().unsqueeze(-1)

        self.obs_buf[:] = torch.cat((
            dof_pos_scaled[:, 0:6], 
            dof_pos_scaled[:, self.idx_master].unsqueeze(-1), 
            gripper_binary_state,
            (self.ur_dof_vel * self.dof_vel_scale)[:, 0:6],
            to_target,
        ), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        progress_ratio = min(1.0, self.global_step_counter / 5000.0) # 假设 5000 回合后达到最大难度

        # 为当前重置的环境重新生成一组固定的手眼标定误差
        cam_pos_noise_scale = 0.015 * progress_ratio  # 模拟 +/- 1.5 cm 的安装平移误差
        cam_rot_noise_scale = math.radians(3.0)  # 模拟 +/- 3 度的安装旋转误差
        
        num_resets = len(env_ids)
        # 随机平移误差
        self.cam_pos_err[env_ids] = (torch.rand((num_resets, 3), device=self.device) - 0.5) * 2.0 * cam_pos_noise_scale
        
        # 随机旋转误差 (生成微小旋转的四元数)
        axes = torch.randn((num_resets, 3), device=self.device)
        axes = axes / torch.norm(axes, dim=-1, keepdim=True)
        angles = (torch.rand((num_resets,), device=self.device) - 0.5) * 2.0 * cam_rot_noise_scale
        sin_half_angles = torch.sin(angles / 2.0).unsqueeze(-1)
        cos_half_angles = torch.cos(angles / 2.0).unsqueeze(-1)
        self.cam_rot_err[env_ids] = torch.cat([axes * sin_half_angles, cos_half_angles], dim=-1)

        raw_pos = self.ur_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_ur_dofs), device=self.device) - 0.5)

        # 强制应用夹爪联动逻辑
        master_pos = raw_pos[:, 6]  # 获取主关节的随机化位置
        
        raw_pos[:, self.idx_r_knuckle] = master_pos       # 比例 1
        raw_pos[:, self.idx_l_inner] = master_pos         # 比例 1
        raw_pos[:, self.idx_r_inner] = master_pos         # 比例 1
        raw_pos[:, self.idx_l_tip] = -master_pos          # 比例 -1 (保持平行)
        raw_pos[:, self.idx_r_tip] = -master_pos          # 比例 -1

        pos = tensor_clamp(raw_pos, self.ur_dof_lower_limits, self.ur_dof_upper_limits)
        
        self.ur_dof_pos[env_ids, :] = pos
        self.ur_dof_vel[env_ids, :] = torch.zeros_like(self.ur_dof_vel[env_ids])
        self.ur_dof_targets[env_ids, :self.num_ur_dofs] = pos

        # 将柜子关节状态(位置和速度)全部重置为零
        self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])

        # 重置道具
        if self.num_props > 0:
            # 获取需要重置的道具的全局索引
            prop_indices = self.global_indices[env_ids, 2:].flatten()
            
            # 将道具状态恢复到默认状态 (default_prop_states 在 _create_envs 中记录)
            self.prop_states[env_ids] = self.default_prop_states[env_ids]
            # 更新物理引擎中的 Actor 根节点状态
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(prop_indices), len(prop_indices))
            
            # 随机分配黄色目标，并更新颜色
            for env_id in env_ids.cpu().numpy():
                env_ptr = self.envs[env_id]

                target_idx = np.random.randint(self.num_props)

                # 随机选择 0 到 8 之间的一个索引作为这局的黄色目标
                self.target_indices[env_id] = target_idx
                
                for prop_i in range(self.num_props):
                    prop_handle = self.gym.find_actor_handle(env_ptr, f"prop{prop_i}")
                    if prop_i == target_idx:
                        color = gymapi.Vec3(1.0, 1.0, 0.0)  # 纯黄色 (R:1, G:1, B:0)
                        self.gym.set_rigid_body_segmentation_id(env_ptr, prop_handle, 0, 255)
                    else:
                        # 随机其他颜色
                        r = np.random.uniform(0.0, 1.0)
                        g = np.random.uniform(0.0, 0.4) # 限制绿色通道
                        b = np.random.uniform(0.3, 1.0) # 保证有蓝色通道
                        color = gymapi.Vec3(r, g, b)

                        self.gym.set_rigid_body_segmentation_id(env_ptr, prop_handle, 0, 0)
                    
                    # 重新将颜色应用到物理引擎中
                    self.gym.set_rigid_body_color(env_ptr, prop_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()
        
        # 更新关节目标张量
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ur_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        # 更新关节状态张量
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        
        # 重置缓冲区
        self.progress_buf[env_ids] = 0 # 步数清零
        self.reset_buf[env_ids] = 0    # 重置标记清零

        if hasattr(self, 'last_actions'):
            self.last_actions[env_ids] = 0.0
            self.episode_action_jitter[env_ids] = 0.0
            self.episode_action_mag[env_ids] = 0.0
            self.episode_rewards[env_ids] = 0.0
        
        # 0 是 UR机械臂, 1 是柜子, 2 到 10 是这 9 个长方体
        self.all_prop_init_pos[env_ids] = self.root_state_tensor[env_ids, 2:11, 0:3].clone()

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        arm_targets = self.ur_dof_targets[:, 0:6] + \
                    self.ur_dof_speed_scales[0:6] * self.dt * self.actions[:, 0:6] * self.action_scale
        
        self.ur_dof_targets[:, 0:6] = tensor_clamp(
            arm_targets, self.ur_dof_lower_limits[0:6], self.ur_dof_upper_limits[0:6])
        
        gripper_action = self.actions[:, 6]

        # 设定阈值 > 0 表示闭合，<= 0 表示张开
        is_closing = gripper_action > 0.0
        is_opening = gripper_action < -0.1

        safe_open_pos = torch.ones_like(self.ur_dof_lower_limits[6]) * 0.05 
        current_target = self.ur_dof_targets[:, 6] # 获取当前位置
        
        # 假设 upper_limits 是闭合，lower_limits 是张开
        target_intent = torch.where(is_closing, self.ur_dof_upper_limits[6], 
                                    torch.where(is_opening, safe_open_pos, current_target))
       
        # 模拟 Robotiq 夹爪的恒定闭合/张开速度
        gripper_speed = 1.0 
        max_step_change = gripper_speed * self.dt
        
        # 让夹爪的当前目标值以恒定速度逼近 "意图终点" (防止突变导致物理爆炸)
        diff = target_intent - self.ur_dof_targets[:, 6]
        step_change = torch.clamp(diff, min=-max_step_change, max=max_step_change)
        
        master_target = self.ur_dof_targets[:, 6] + step_change
        master_target = torch.clamp(master_target, self.ur_dof_lower_limits[6], self.ur_dof_upper_limits[6])
        self.ur_dof_targets[:, 6] = master_target

        # 同步其余 5 个夹爪从动关节
        self.ur_dof_targets[:, self.idx_r_knuckle] = master_target       # multiplier 1
        self.ur_dof_targets[:, self.idx_l_inner] = master_target         # multiplier 1
        self.ur_dof_targets[:, self.idx_r_inner] = master_target         # multiplier 1
        self.ur_dof_targets[:, self.idx_l_tip] = -master_target          # multiplier -1 (反向旋转以保持平行)
        self.ur_dof_targets[:, self.idx_r_tip] = -master_target
        
        # 将完整的目标发送给物理引擎
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ur_dof_targets))

    def post_physics_step(self):
        # 增加当前回合的步数计数器
        self.progress_buf += 1

        # 检查重置缓冲区 (reset_buf)，找出需要重置的环境索引 (非零值为需要重置)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # 如果有环境需要重置，调用 reset_idx 函数
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # 计算当前的观测值 (Observations)，作为 Policy 的输入
        self.compute_observations()
        # 计算当前的奖励值 (Rewards)，用于训练
        self.compute_reward(self.actions)
        
        vec1 = self.ur_grasp_pos - self.hand_pos
        vec1_dir = vec1 / (torch.norm(vec1, p=2, dim=-1, keepdim=True) + 1e-5)
        
        vec2 = self.target_pos - self.ur_grasp_pos
        vec2_dir = vec2 / (torch.norm(vec2, p=2, dim=-1, keepdim=True) + 1e-5)
        
        align_dot = torch.sum(vec1_dir * vec2_dir, dim=-1)
        align_error = 1.0 - align_dot

        current_actions = self.actions.detach()
        step_action_jitter = torch.mean((current_actions - self.last_actions) ** 2, dim=-1)
        self.episode_action_jitter += step_action_jitter
        # 更新上一帧动作
        self.last_actions.copy_(current_actions)
        
        # 计算平均每步的动作抖动
        mean_action_jitter = self.episode_action_jitter / torch.clamp(self.progress_buf.float(), min=1.0)

        step_action_mag = torch.mean(current_actions ** 2, dim=-1)
        self.episode_action_mag += step_action_mag
        mean_action_mag = self.episode_action_mag / torch.clamp(self.progress_buf.float(), min=1.0)
        self.episode_rewards += self.rew_buf

        if "episode" not in self.extras:
            self.extras["episode"] = {}
        
        self.extras["episode"]["action_jitter"] = mean_action_jitter.clone().to(torch.float32)
        self.extras["episode"]["action_magnitude"] = mean_action_mag.clone().to(torch.float32)
        self.extras["episode"]["reward"] = self.episode_rewards.clone()

        self.global_step_counter += 1
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        if len(reset_env_ids) > 0:
            # 找出在当前全局回合中，【还没记录过成绩】的环境
            unrecorded_mask = ~self.ep_recorded[reset_env_ids]
            valid_reset_ids = reset_env_ids[unrecorded_mask]
            self.env_success_history[reset_env_ids] = self.is_success[reset_env_ids].float()
            # 强制刷新它的误差和抖动记录
            self.env_align_err_history[reset_env_ids] = align_error[reset_env_ids].float()
            self.env_action_jitter_history[reset_env_ids] = mean_action_jitter[reset_env_ids].float()

            if len(valid_reset_ids) > 0:
                # 盖章记录它们结束这一瞬间的成绩
                self.ep_success_buf[valid_reset_ids] = self.is_success[valid_reset_ids].float()
                self.ep_align_err_buf[valid_reset_ids] = align_error[valid_reset_ids].float()
                # 标记为已出成绩
                self.ep_recorded[valid_reset_ids] = True

        true_success_rate = self.env_success_history.mean().item()
        self.print_success_rate = true_success_rate # 传递给终端打印
        self.extras["episode"]["success_rate"] = true_success_rate

        true_align_err = self.env_align_err_history.mean().item()
        self.print_align_err = true_align_err
        self.extras["episode"]["alignment_error"] = align_error

        # 取当前步所有环境的平均即时奖励
        mean_rew = self.rew_buf.mean().item()
        if self.global_step_counter % self.max_episode_length == 0:
            current_ep_sr = self.ep_success_buf.mean().item()
            current_ep_err = self.ep_align_err_buf.mean().item()

            # 计算这是第几个全局回合
            global_ep_idx = self.global_step_counter // self.max_episode_length

            print(f"DEBUG: Global Episode {global_ep_idx:04d} | Mean Step Reward: {mean_rew:>8.4f} | SR: {current_ep_sr:>6.2%} | Align Err: {current_ep_err:>6.4f}", flush=True)

            import sys
            sys.stdout.flush()

            # 清空成绩单，开启下一轮全局回合的统计
            self.ep_recorded[:] = False
            self.ep_success_buf[:] = 0.0
            self.ep_align_err_buf[:] = 0.0

        # 如果开启了 Viewer 且启用了调试可视化 (debug_viz)
        if self.viewer and self.debug_viz:
            # 清除上一帧绘制的线条
            self.gym.clear_lines(self.viewer)
            # 刷新刚体状态张量 (确保可视化位置是最新的)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # 遍历每个环境绘制坐标轴
            for i in range(self.num_envs):
                # 绘制 ur 手爪中心点的坐标轴 (RGB: 红X, 绿Y, 蓝Z)
                # 计算 X 轴末端点位置
                px = (self.ur_grasp_pos[i] + quat_apply(self.ur_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.8)).cpu().numpy()
                # 计算 Y 轴末端点位置
                py = (self.ur_grasp_pos[i] + quat_apply(self.ur_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.8)).cpu().numpy()
                # 计算 Z 轴末端点位置
                pz = (self.ur_grasp_pos[i] + quat_apply(self.ur_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.8)).cpu().numpy()

                p0 = self.ur_grasp_pos[i].cpu().numpy()
                
                # 画线
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # 绘制抽屉把手目标点的坐标轴
                px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.drawer_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                # 绘制左指坐标轴
                px = (self.ur_lfinger_pos[i] + quat_apply(self.ur_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.ur_lfinger_pos[i] + quat_apply(self.ur_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ur_lfinger_pos[i] + quat_apply(self.ur_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.ur_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                # 绘制右指坐标轴
                px = (self.ur_rfinger_pos[i] + quat_apply(self.ur_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.ur_rfinger_pos[i] + quat_apply(self.ur_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ur_rfinger_pos[i] + quat_apply(self.ur_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.ur_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                t_idx = self.target_indices[i]
                
                target_pos = self.root_state_tensor[i, 2 + t_idx, 0:3]
                target_rot = self.root_state_tensor[i, 2 + t_idx, 3:7]

                # 计算坐标轴末端点 (将长度设为 0.1 米，因为道具边长是 0.08)
                axis_length = 0.1
                px_tgt = (target_pos + quat_apply(target_rot, to_torch([1, 0, 0], device=self.device) * axis_length)).cpu().numpy()
                py_tgt = (target_pos + quat_apply(target_rot, to_torch([0, 1, 0], device=self.device) * axis_length)).cpu().numpy()
                pz_tgt = (target_pos + quat_apply(target_rot, to_torch([0, 0, 1], device=self.device) * axis_length)).cpu().numpy()
                
                p0_tgt = target_pos.cpu().numpy()

                # 青色代表 X 轴
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0_tgt[0], p0_tgt[1], p0_tgt[2], px_tgt[0], px_tgt[1], px_tgt[2]], [0.0, 1.0, 1.0])
                # 洋红色代表 Y 轴
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0_tgt[0], p0_tgt[1], p0_tgt[2], py_tgt[0], py_tgt[1], py_tgt[2]], [1.0, 0.0, 1.0])
                # 黄色代表 Z 轴 (正好与物体的向上方向一致)
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0_tgt[0], p0_tgt[1], p0_tgt[2], pz_tgt[0], pz_tgt[1], pz_tgt[2]], [1.0, 1.0, 0.0]) 

    

    
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_ur_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor, actions: torch.Tensor, 
    hand_pos: torch.Tensor,
    extract_offset: torch.Tensor,
    ur_grasp_pos: torch.Tensor, ur_grasp_rot: torch.Tensor, 
    ur_lfinger_pos: torch.Tensor, ur_rfinger_pos: torch.Tensor, 
    all_prop_pos: torch.Tensor, all_prop_init_pos: torch.Tensor,
    target_indices: torch.Tensor, # 目标索引参数
    action_penalty_scale: float, dist_reward_scale: float, 
    align_reward_scale: float, grasp_reward_scale: float, lift_reward_scale: float,
    finger_dist_reward_scale: float, 
    max_episode_length: int
):

    num_envs = all_prop_pos.shape[0]
    batch_indices = torch.arange(num_envs, device=all_prop_pos.device)

    target_pos = all_prop_pos[batch_indices, target_indices]

    d = torch.norm(ur_grasp_pos - target_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2) + 5 * torch.exp(- d / 0.35) + 10 * torch.exp(- d / 0.25) + 20 * torch.exp(- d / 0.15) + 40 * torch.exp(- d / 0.05)

    # 计算对齐奖励
    vec1 = ur_grasp_pos - hand_pos
    vec1_dir = vec1 / (torch.norm(vec1, p=2, dim=-1, keepdim=True) + 1e-5)

    # 向量 2：ur_grasp_pos 指向 目标物体 
    vec2 = target_pos - ur_grasp_pos
    vec2_dir = vec2 / (torch.norm(vec2, p=2, dim=-1, keepdim=True) + 1e-5)

    # 计算点积：结果在 [-1, 1] 之间。越接近 1 越对齐
    align_dot = torch.sum(vec1_dir * vec2_dir, dim=-1)
    
    # 离得越近，姿态对齐的奖励越明显
    align_reward = torch.clamp(align_dot, min=0.0) * torch.exp(-2.0 * d)

    lfinger_dist = torch.norm(ur_lfinger_pos - target_pos, p=2, dim=-1)
    rfinger_dist = torch.norm(ur_rfinger_pos - target_pos, p=2, dim=-1)
    
    finger_reward = torch.exp(-10.0 * lfinger_dist) + torch.exp(-10.0 * rfinger_dist)
    is_close = (d < 0.04).float()

    # 计算两个指尖之间的真实距离
    fingers_width = torch.norm(ur_lfinger_pos - ur_rfinger_pos, p=2, dim=-1)

    is_aligned_and_close = (d < 0.04) & (align_dot > 0.8)
    
    # 差值越小，挤压奖励越高
    squeeze_reward = torch.exp(-15.0 * torch.abs(fingers_width - 0.05)) * is_aligned_and_close.float()
    
    grasp_pose_reward = (is_close * finger_reward) + (squeeze_reward * 10.0)

    # 提取初始Z
    target_init_z = all_prop_init_pos[batch_indices, target_indices, 2]

    current_lift_height = target_pos[:, 2] - target_init_z

    # 稳固抓取判定：距离把手近 + 稍微离开底面 + 指尖正在收拢
    is_grasped = (d < 0.07) & (current_lift_height > 0.005) & (fingers_width < 0.09)

    grasp_keep_bonus = is_grasped.float() * 200

    obj_goal_pos = all_prop_init_pos[batch_indices, target_indices].clone()
    obj_goal_pos += extract_offset

    # 计算物体当前位置到虚拟终点的距离
    obj_to_goal_dist = torch.norm(obj_goal_pos - ur_grasp_pos, p=2, dim=-1)

    clamped_extract = torch.clamp(obj_to_goal_dist, min=0.0, max=0.3)
    extract_reward = (0.3 - clamped_extract) * is_grasped.float() * 100.0

    # 成功将物体抬起，则视为完成任务
    steps_remaining = max_episode_length - progress_buf
    is_success = (current_lift_height > 0.05) & is_grasped
    success_bonus = is_success.float() * (500 + steps_remaining * 650)

    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = (
        dist_reward_scale * dist_reward +
        align_reward_scale * align_reward +
        finger_dist_reward_scale * grasp_pose_reward +  
        grasp_reward_scale * is_close * 5.0 +  
        grasp_keep_bonus +      
        extract_reward + 
        success_bonus -
        action_penalty_scale * action_penalty
    )


    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    # 一旦黄色物体被成功抓起超过10cm，回合立刻胜利并重置
    reset_buf = torch.where(is_success, torch.ones_like(reset_buf), reset_buf) 
    # 如果黄色物体不慎掉落到地上，判定失败并重置
    is_dropped = (target_pos[:, 2] < 0.01) & (progress_buf > 30) 
    rewards = torch.where(is_dropped, rewards - 50.0, rewards) # 掉落扣大分
    reset_buf = torch.where(is_dropped, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf, is_success              
