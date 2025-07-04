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


@register_env("ChargingTask-v0", max_episode_steps=1000)
class ChargingTaskEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["aubo_c5"]

    agent: Union[AuboC5]
    def __init__(self, *args,robot_uids="aubo_c5", **kwargs):
        super().__init__(*args,robot_uids=robot_uids, **kwargs)
        
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))
        self.all_links = self.agent.robot.get_links()
        self.wrist_camera_link = self.agent.robot.find_link_by_name('wrist_camera_Link')
        self.tcp = self.agent.robot.find_link_by_name('charging_gun_Link')
        
    def _load_scene(self, options: dict):
        # charging_socket_pose = rand_pose(
        #     xlim=[-0.5,0.5],
        #     ylim=[-0.5,-0.7],
        #     zlim=[0.3,0.9],
        #     qpos=[ 0, 0.707107, 0,0.707107],
        #     ylim_prop=False,
        #     rotate_rand=False,
        #     rotate_lim=[0,0,0],
        # )
        charging_socket_pose = sapien.Pose([0.5, 0.0, 0.8])
        charging_socket_pose.set_rpy([0,-np.pi/2,-np.pi/2])

        self.charging_socket,_ = create_glb(
            self.scene,
            pose=charging_socket_pose,
            modelname="charging_socket_simplify",
            convex=False,
            is_static=True
        )
    def reset(self,**kwargs):
        obs,info = super().reset(**kwargs)
        new_obs = {}
        new_obs["pixels"] = {}
        new_obs["pixels"]["front"] = obs["sensor_data"]["front"]['rgb'].cpu().numpy()
        new_obs["pixels"]["wrist"] = obs["sensor_data"]["wrist"]['rgb'].cpu().numpy()
        new_obs["agent_pos"] = np.concatenate([obs["agent"]["qpos"].cpu().numpy(),obs["agent"]["qvel"].cpu().numpy()],axis=1)
        
        return new_obs,info
    
    def step(self, action):
        print("action: ",action)
        obs, reward, done,truncated, info = super().step(action[:6])
        new_obs = {}
        new_obs["pixels"] = {}
        new_obs["pixels"]["front"] = obs["sensor_data"]["front"]['rgb'].cpu().numpy()
        new_obs["pixels"]["wrist"] = obs["sensor_data"]["wrist"]['rgb'].cpu().numpy()
        new_obs["agent_pos"] = np.concatenate([obs["agent"]["qpos"].cpu().numpy(),obs["agent"]["qvel"].cpu().numpy()],axis=1)
        print("tcp pose: ",self.tcp.pose)
        if self.render_mode == "human":
            self.render()
        return new_obs, reward, done,truncated, info
    
    def evaluate(self):
        charging_gun_pos = self.all_links[8].pose.p
        charging_socket_pos = self.charging_socket.pose.p
        charging_socket_pos[:,1] -= 0.034
        eps = torch.tensor([0.01,0.01,0.01])

        return {
            "success": torch.all(abs(charging_gun_pos - charging_socket_pos) < eps,axis=1),
        }
    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        camera_config = []
        pose = sapien_utils.look_at(eye=[0, 0, 1.5], target=[0.0, -1.0, 0.5])
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
            init_pose = torch.ones((b, self.agent.robot.dof), device=self.device)*0.01 
            init_pose[:,0] = np.pi / 2
            self.agent.robot.set_qpos(
                init_pose
            )
    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.tcp.pose.raw_pose,
        )
        return obs