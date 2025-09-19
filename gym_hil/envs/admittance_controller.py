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
        self.rpy_des = np.zeros(3)  # 期望姿态角（欧拉角）

        self.rpy_limits_min = np.array([-np.pi/6, -np.pi/6, -np.pi/6])  # 姿态角最小值
        self.rpy_limits_max = np.array([np.pi/6, np.pi/6, np.pi/6])

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

    def compute_rotation_error(self, r_cur:R, r_ref:R):
        delta_r = r_ref.inv() * r_cur
        rotvec = delta_r.as_rotvec()  # 旋转向量
        return rotvec
    
    def set_rpy_limits(self, rpy_limits_min: np.ndarray,rpy_limits_max: np.ndarray):
        """
        设置姿态角限制
        rpy_limits: 3维数组，表示每个轴的最大旋转角度（弧度）
        """
        self.rpy_limits_min = rpy_limits_min
        self.rpy_limits_max = rpy_limits_max

    def step(self, delta_x_des: np.ndarray, rpy_des: np.ndarray, force: np.ndarray, torque: np.ndarray):
        # --- 线性导纳控制 ---
        self.x_des = self.x + delta_x_des
        fx = force - self.D_pos @ self.dx - self.K_pos @ (self.x - self.x_des)
        ddx = np.linalg.inv(self.M_pos) @ fx
        dx = self.dx + ddx * self.dt
        x_des_new = self.x_des + dx * self.dt
        delta_x_cmd = x_des_new - self.x  # 计算位置增量

        # # --- 姿态导纳控制 ---
        self.rpy_des = np.clip(rpy_des,self.rpy_limits_min, self.rpy_limits_max)
        r_des = R.from_euler('xyz', self.rpy_des, degrees=False)  # 期望rpy
        r_cur = self.q
        rot_error = self.compute_rotation_error(r_cur=r_des, r_ref=r_cur)
        torque_feedback = torque - self.D_ori @ self.w - self.K_ori @ rot_error
        dw = np.linalg.inv(self.M_ori) @ torque_feedback
        self.w += dw * self.dt
        r_comp = R.from_rotvec(self.w * self.dt)
        r_cmd = r_des * r_comp
        # rotvec = R.from_rotvec(r_des.as_rotvec() + self.w * self.dt)  
        delta_rpy_cmd = (r_cmd.inv() * r_cur).as_euler("xyz", degrees=False)
        return np.concatenate([delta_x_cmd, delta_rpy_cmd])
