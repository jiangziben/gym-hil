import numpy as np
from scipy.spatial.transform import Rotation as R

class AdmittanceController:
    def __init__(self, M, D, K, dt):
        self.M_pos = np.diag(M[0:3])
        self.D_pos = np.diag(D[0:3])
        self.K_pos = np.diag(K[0:3])
        self.M_ori = np.diag(M[3:6])
        self.D_ori = np.diag(D[3:6])
        self.K_ori = np.diag(K[3:6])
        self.dt = dt

        # 状态变量（线性）
        self.x = np.zeros(3)  # 位置
        self.dx = np.zeros(3)  # 速度
        self.x_des = np.zeros(3)  # 期望位置

        # 状态变量（姿态）
        self.q = R.from_quat([0,0,0,1])  # 姿态：scipy四元数对象
        self.w = np.zeros(3)  # 姿态角速度
        self.w_des = np.zeros(3)  # 姿态角位移（积分得到）

    def reset(self, pose: np.ndarray):
        """
        pose: 7维，前3维为位置，后4维为四元数（wxyz）
        """
        self.x = pose[:3].copy()
        self.dx = np.zeros(3)
        self.q = R.from_quat([pose[4], pose[5], pose[6], pose[3]])  # 四元数格式为 [x, y, z, w]
        self.w = np.zeros(3)
        self.x_des = self.x.copy()
        self.w_des = np.zeros(3)  # 姿态角位移（积分得到）
        
    def set_state(self, x: np.ndarray, dx: np.ndarray, q: np.ndarray, w: np.ndarray):
        self.x = x.copy()
        self.dx = dx.copy()
        self.q = R.from_quat([q[1],q[2],q[3],q[0]])  # 四元数格式为 [x, y, z, w]
        self.w = w.copy()

    def limit_rotation_vector(self,R_input: R, R_ref: R, max_angle_rad: float):
        # 相对旋转
        R_rel = R_ref.inv() * R_input
        rotvec = R_rel.as_rotvec()
        angle = np.linalg.norm(rotvec)
        
        # 限制角度
        if angle > max_angle_rad:
            rotvec = rotvec / angle * max_angle_rad
            R_rel_limited = R.from_rotvec(rotvec)
            R_limited = R_ref * R_rel_limited
            return R_limited
        else:
            return R_input

    def step(self, x_des: np.ndarray, q_des: np.ndarray, force: np.ndarray, torque: np.ndarray):
        """
        输入：
            x_des: 期望位置 (3,)
            q_des: 期望四元数 (4,) -> [x, y, z, w]
            force: 传感器测得的外力 (3,)
            torque: 传感器测得的外力矩 (3,)
        返回：
            pose: 当前控制器更新后的7维 pose（位置 + 四元数）
        """
        # --- 线性导纳控制 ---
        fx = force - self.D_pos @ self.dx #- self.K_pos @ (self.x - self.x_des)
        ddx = np.linalg.inv(self.M_pos) @ fx
        dx = self.dx + ddx * self.dt
        # self.x_des += dx * self.dt
        delta_x = dx * self.dt

        # --- 姿态导纳控制 ---
        torque_feedback = torque - self.D_ori @ self.w #- self.K_ori @ q_err
        dw = np.linalg.inv(self.M_ori) @ torque_feedback
        self.w += dw * self.dt
        # self.w_des += self.w * self.dt  # 角速度积分得到角位移
        # Rot = self.limit_rotation_vector(R.from_rotvec(self.w_des),self.q, np.pi/6) #计算rpy角（弧度）
        # self.w_des = Rot.as_rotvec() #更新姿态
        # rpy = Rot.as_euler("xyz", degrees=False)  # 转换为欧拉角（弧度）
        delta_q = R.from_rotvec(self.w * self.dt)
        delta_rpy = delta_q.as_euler("xyz", degrees=False)  # 转换为欧拉角（弧度）
        return np.concatenate([delta_x, delta_rpy])
