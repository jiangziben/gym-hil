import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from .aubo_c5 import AuboC5
from typing import Union,Dict
from .create_actor import *
from .rand_create_actor import *
import torch
from mani_skill.sensors.camera import CameraConfig
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
import numpy as np
from .admittance_controller import AdmittanceController
from .force_vis import RealTimeForceVisualizer
from .image_vis import RealTimeImagePlotter
import torch.nn.functional as F
import matplotlib.pyplot as plt

@register_env("ChargingTask-v0", max_episode_steps=1000)
class ChargingTaskEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["aubo_c5"]

    agent: Union[AuboC5]
    def __init__(self, *args,robot_uids="aubo_c5",image_obs = True, **kwargs):
        # 控制周期
        dt = 0.01
        # 控制参数
        M_diag = [1, 1, 1, 0.5, 0.5, 0.5]
        D_diag = [100, 50, 50, 5, 5, 5]
        K_diag = [20, 0, 0, 10, 10, 10]
        # 初始化阻抗控制器
        self.admittance_controller = AdmittanceController(M_diag, D_diag, K_diag, dt)
        super().__init__(*args,robot_uids=robot_uids,reconfiguration_freq=1, **kwargs)
        info = self.get_info()
        obs = self.get_obs(info)
        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({
                "front": spaces.Box(low=0, high=255, shape=obs["sensor_data"]["front"]['rgb'].cpu().numpy()[0].shape, dtype=np.uint8),
                "wrist": spaces.Box(low=0, high=255, shape=obs["sensor_data"]["wrist"]['rgb'].cpu().numpy()[0].shape, dtype=np.uint8),
            }),
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=np.concatenate([obs["agent"]["qpos"].cpu().numpy()[0],
                                                                                    obs["agent"]["qvel"].cpu().numpy()[0],
                                                                                    self.tcp.pose.raw_pose.cpu().numpy()[0],
                                                                                    self.tcp_force
                                                                                    ]).shape, dtype=np.float32)
            # "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=np.concatenate([obs["agent"]["qpos"].cpu().numpy()[0],
            #                                                                         obs["agent"]["qvel"].cpu().numpy()[0],
            #                                                                         self.tcp.pose.raw_pose.cpu().numpy()[0]
            #                                                                         ]).shape, dtype=np.float32)
        })
        # self.control_force_vis = RealTimeForceVisualizer("control_force")
        self.state_force_vis = RealTimeForceVisualizer("state_force")
        # self.image_vis = RealTimeImagePlotter("front", "wrist")
        
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))
        self.all_links = self.agent.robot.get_links()
        self.wrist_camera_link = self.agent.robot.find_link_by_name('wrist_camera_Link')
        self.tcp = self.agent.robot.find_link_by_name('charging_gun_Link')
        
    def _load_scene(self, options: dict):
        tmp = sapien.Pose(p=[0,0,0])
        tmp.set_rpy([0, -np.pi / 2, -np.pi / 2])
        charging_socket_quat = tmp.get_q()
        charging_socket_pose = rand_pose(
            xlim=[0.5,0.7],
            ylim=[-0.1,0.1],
            zlim=[0.6,0.8],
            qpos=charging_socket_quat,
            ylim_prop=False,
            rotate_rand=False,
            rotate_lim=[0,0,0],
        )
        # charging_socket_pose = sapien.Pose([0.5, 0.0, 0.8])
        # charging_socket_pose.set_rpy([0,-np.pi/2,-np.pi/2])

        self.charging_socket,_ = create_glb(
            self.scene,
            pose=charging_socket_pose,
            modelname="charging_socket_simplify",
            convex=False,
            is_static=True
        )
        
    def reset(self,**kwargs):
        self.control_input_sum = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs,info = super().reset(**kwargs)
        new_obs = {}
        new_obs["pixels"] = {}
        new_obs["pixels"]["front"] = obs["sensor_data"]["front"]['rgb'].cpu().numpy()[0]
        new_obs["pixels"]["wrist"] = obs["sensor_data"]["wrist"]['rgb'].cpu().numpy()[0]
        new_obs["agent_pos"] = np.concatenate([obs["agent"]["qpos"].cpu().numpy()[0],obs["agent"]["qvel"].cpu().numpy()[0],
                                               self.tcp.pose.raw_pose.cpu().numpy()[0],
                                               self.tcp_force
                                               ])
        # new_obs["agent_pos"] = np.concatenate([obs["agent"]["qpos"].cpu().numpy()[0],obs["agent"]["qvel"].cpu().numpy()[0],
        #                                        self.tcp.pose.raw_pose.cpu().numpy()[0]
        #                                        ])
        
        self.admittance_controller.reset(self.tcp_init_pose.cpu().numpy()[0])
        
        return new_obs,info
    
    def clip_action(self, action):
        return np.clip(action, [0.1,-0.9, 0.1], [0.9,0.9,0.9])
    
    def compute_force_and_torque(self, scene, ee_link):
        dt = scene.get_timestep()
        total_force = np.zeros(6, dtype=np.float32)
        ee_pos = np.array(ee_link.pose.p)

        contacts = scene.get_contacts()
        for contact in contacts:
            if ee_link._objs[0] in contact.bodies:
                for point in contact.points:
                    impulse = np.array(point.impulse)
                    force = impulse / dt
                    total_force[:3] += force

                    r = np.array(point.position) - ee_pos
                    torque = np.cross(r[0], force)
                    total_force[3:] += torque

        return total_force  
    def orientation_penalty(self,q_current, q_ref, threshold_rad=np.pi/6, penalty_scale=1.0):
        """
        计算当前姿态和参考姿态之间的旋转角度差异惩罚，
        当旋转差超过阈值时，产生平方惩罚。

        返回:
            penalty: (N,) 张量，batch中每个样本的惩罚值
        """
        # 保证输入是2维batch
        if q_current.ndim == 1:
            q_current = q_current.unsqueeze(0)
        if q_ref.ndim == 1:
            q_ref = q_ref.unsqueeze(0)

        # 归一化四元数
        q_current = q_current / q_current.norm(dim=1, keepdim=True).clamp(min=1e-8)
        q_ref = q_ref / q_ref.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # 计算两个四元数的点积 |<q1, q2>|
        dot = torch.abs(torch.sum(q_current * q_ref, dim=1)).clamp(max=1.0)

        # 计算夹角θ = 2 * acos(|dot|)
        angle = 2.0 * torch.acos(dot)

        # 计算超过阈值的部分
        excess = torch.relu(angle - threshold_rad)

        # 惩罚：超出部分的平方，并乘以惩罚系数
        penalty = penalty_scale * excess * excess
        return penalty
    
    def orientation_penalty_x(self,q_current, penalty_scale=0.1,threshold_rad = np.pi/18):
        """
        对当前姿态绕x轴（roll）的角度进行平方惩罚，目标是越接近0越好（不使用阈值）。

        Args:
            q_current: (N, 4) 当前四元数，格式 (w, x, y, z)
            penalty_scale: 惩罚系数

        Returns:
            penalty: (N,) 张量，每个样本的姿态惩罚
        """

        def quat_to_roll(q):
            # q: (N, 4)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = torch.atan2(sinr_cosp, cosr_cosp)
            return roll

        if q_current.ndim == 1:
            q_current = q_current.unsqueeze(0)

        q_current = F.normalize(q_current, dim=1)
        roll = quat_to_roll(q_current)
        if np.fabs(roll) > threshold_rad:
            penalty = penalty_scale * roll ** 2
        else:
            penalty = 0.0
        return penalty
        
    def step(self, action):
        # print("action: ",action)
        # action[3:6] = np.array([0,0,0],dtype=np.float32)  # 禁止转动
        # new_action = action
        # self.control_input_sum += new_action[:3]
        # # # new_action[0:3] = self.tcp_init_pose[0][:3] + self.control_input_sum
        # new_action[:3] = self.clip_action(self.tcp_init_pose[0][:3].cpu().numpy() + self.control_input_sum)
        # self.control_input_sum = new_action[:3] - self.tcp_init_pose[0][:3].cpu().numpy()
        # print("new_action: ",new_action)
        # print("self.control_input_sum: ",self.control_input_sum)
        input_force = self.tcp_force[0:3] #输入力
        input_torque = self.tcp_force[3:6] #输入力矩
        self.admittance_controller.set_state(self.tcp.pose.get_p().cpu().numpy()[0],
                                             self.tcp.linear_velocity.cpu().numpy()[0],
                                             self.tcp.pose.get_q().cpu().numpy()[0],
                                             self.tcp.angular_velocity.cpu().numpy()[0])
        new_action = self.admittance_controller.step(action[0:3],action[3:6],input_force,input_torque)
        obs, reward, done,truncated, info = super().step(new_action[:6])
        # import time
        # start = time.time()
        self.tcp_force = self.compute_force_and_torque(self.scene, self.tcp)
        # used_time = time.time() - start
        # print("used time: ",used_time)
        # print(self.tcp_force)

        new_obs = {}
        new_obs["pixels"] = {}
        new_obs["pixels"]["front"] = obs["sensor_data"]["front"]['rgb'].cpu().numpy()[0]
        new_obs["pixels"]["wrist"] = obs["sensor_data"]["wrist"]['rgb'].cpu().numpy()[0]
        new_obs["agent_pos"] = np.concatenate([obs["agent"]["qpos"].cpu().numpy()[0],obs["agent"]["qvel"].cpu().numpy()[0],
                                               self.tcp.pose.raw_pose.cpu().numpy()[0],
                                               self.tcp_force])
        # new_obs["agent_pos"] = np.concatenate([obs["agent"]["qpos"].cpu().numpy()[0],obs["agent"]["qvel"].cpu().numpy()[0],
        #                                        self.tcp.pose.raw_pose.cpu().numpy()[0]])
        # print("tcp pose: ",self.tcp.pose)
        # # show
        # self.control_force_vis.update_force(action) # 更新力可视化
        self.state_force_vis.update_force(self.tcp_force) # 更新力可视化
        # self.control_force_vis.show() # 显示力可视化
        self.state_force_vis.show() # 显示力可视化
        # self.image_vis.update(new_obs["pixels"]["front"],
        #                      new_obs["pixels"]["wrist"])
        self.render()
        # # # # 增加姿态惩罚
        # λ_orient = 0.1
        # reward = reward.float() - self.orientation_penalty_x(self.tcp.pose.q ,penalty_scale=λ_orient)
            
        # print("reward: ",reward.float())
        # print("agent_pos: ",obs["agent"]["qpos"].cpu().numpy()[0])
        return new_obs, reward,done,truncated, info
    
    def evaluate(self):
        charging_gun_pos = self.all_links[8].pose.p
        charging_socket_pos = self.charging_socket.pose.p
        charging_socket_pos[:,0] += 0.034
        eps = torch.tensor([0.001,0.01,0.01])
        # print("abs(charging_gun_pos - charging_socket_pos): ",abs(charging_gun_pos - charging_socket_pos))
        contact_forces = self.tcp.get_net_contact_forces()
        contact_forces_flag = (contact_forces[:,0] < -1.0) #力约束
        pos_flag = torch.all(abs(charging_gun_pos - charging_socket_pos) < eps,axis=1) #位置约束
        return {
            "success": contact_forces_flag & pos_flag,
        }
    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        camera_config = []
        pose = sapien_utils.look_at(eye=[-1.0, 0, 1.5], target=[0.5, 0.0, 0.8])
        front_camera = CameraConfig("front", pose=pose, width=320, height=240, fov=55/180.0 * np.pi, near=0.01, far=100)
        camera_config.append(front_camera)
        camera_config.append(
            CameraConfig(
                uid="wrist", pose=sapien_utils.Pose.create_from_pq([0,0,0]), width=320,
                height=240, fov=np.pi / 2, near=0.01,
                far=100, mount=self.wrist_camera_link)
            )
        return camera_config
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # using torch.device context manager to auto create tensors 
        # on CPU/CUDA depending on self.device, the device the env runs on
        with torch.device(self.device):
            b = len(env_idx)
            # init_pose = torch.ones((b, self.agent.robot.dof), device=self.device)*0.01 
            # init_pose[:,0] = np.pi / 2
            init_pose = torch.tensor([[1.5623385,  0.7971761,  1.6842455,  0.8892992,  0.01736733, 0.01151336]],device=self.device)
            self.agent.robot.set_qpos(
                init_pose
            )
            self.tcp_init_pose = self.tcp.pose.raw_pose
            self.tcp_force = self.compute_force_and_torque(self.scene, self.tcp)

            
    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.tcp.pose.raw_pose,
        )
        return obs