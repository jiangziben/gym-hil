import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class RealTimeForceVisualizer:
    def __init__(self, name, arrow_scale=1.0):
        self.force = np.array([0.0, 0.0, 0.0])
        self.torque = np.array([0.0, 0.0, 0.0])
        self.arrow_scale = arrow_scale

        self.fig = plt.figure(figsize=(10, 5))

        # 子图1: 力
        self.ax_force = self.fig.add_subplot(121, projection='3d')
        self.quiver_fx = self.ax_force.quiver(0, 0, 0, 0, 0, 0, color='red', label='Fx')
        self.quiver_fy = self.ax_force.quiver(0, 0, 0, 0, 0, 0, color='green', label='Fy')
        self.quiver_fz = self.ax_force.quiver(0, 0, 0, 0, 0, 0, color='blue', label='Fz')
        self.ax_force.set_xlim([-100, 100])
        self.ax_force.set_ylim([-100, 100])
        self.ax_force.set_zlim([-100, 100])
        self.ax_force.set_xlabel('Fx')
        self.ax_force.set_ylabel('Fy')
        self.ax_force.set_zlabel('Fz')
        self.ax_force.set_title(f"{name} - Force")
        self.ax_force.legend()

        # 子图2: 力矩
        self.ax_torque = self.fig.add_subplot(122, projection='3d')
        self.quiver_tx = self.ax_torque.quiver(0, 0, 0, 0, 0, 0, color='purple', label='Tx')
        self.quiver_ty = self.ax_torque.quiver(0, 0, 0, 0, 0, 0, color='orange', label='Ty')
        self.quiver_tz = self.ax_torque.quiver(0, 0, 0, 0, 0, 0, color='cyan', label='Tz')
        self.ax_torque.set_xlim([-20, 20])
        self.ax_torque.set_ylim([-20, 20])
        self.ax_torque.set_zlim([-20, 20])
        self.ax_torque.set_xlabel('Tx')
        self.ax_torque.set_ylabel('Ty')
        self.ax_torque.set_zlabel('Tz')
        self.ax_torque.set_title(f"{name} - Torque")
        self.ax_torque.legend()

        self.ani = FuncAnimation(self.fig, self._update_plot, interval=100, blit=False)

    def update_force(self, force):
        """外部调用：更新力和力矩的分量"""
        self.force = np.array(force[:3])
        self.torque = np.array(force[3:6]) if len(force) > 3 else np.array([0.0, 0.0, 0.0])

    def _update_plot(self, frame):
        # 更新力箭头
        self.quiver_fx.remove()
        self.quiver_fy.remove()
        self.quiver_fz.remove()
        fx, fy, fz = self.force
        scale = self.arrow_scale
        self.quiver_fx = self.ax_force.quiver(0, 0, 0, fx, 0, 0, color='red', length=scale * abs(fx), normalize=True)
        self.quiver_fy = self.ax_force.quiver(0, 0, 0, 0, fy, 0, color='green', length=scale * abs(fy), normalize=True)
        self.quiver_fz = self.ax_force.quiver(0, 0, 0, 0, 0, fz, color='blue', length=scale * abs(fz), normalize=True)

        # 更新力矩箭头
        self.quiver_tx.remove()
        self.quiver_ty.remove()
        self.quiver_tz.remove()
        tx, ty, tz = self.torque
        self.quiver_tx = self.ax_torque.quiver(0, 0, 0, tx, 0, 0, color='purple', length=scale * abs(tx), normalize=True)
        self.quiver_ty = self.ax_torque.quiver(0, 0, 0, 0, ty, 0, color='orange', length=scale * abs(ty), normalize=True)
        self.quiver_tz = self.ax_torque.quiver(0, 0, 0, 0, 0, tz, color='cyan', length=scale * abs(tz), normalize=True)

    def show(self):
        """启动可视化窗口"""
        plt.pause(0.01)

if __name__ == "__main__":
    import time

    vis = RealTimeForceVisualizer("Force and Torque Visualization")

    # 模拟实时数据输入（例如来自传感器 / 仿真）
    t = 0
    while True:
        fx = 100 * np.sin(t)
        fy = 100 * np.cos(t)
        fz = 100 * np.sin(2 * t)
        tx = 50 * np.cos(t)
        ty = 50 * np.sin(t)
        tz = 50 * np.cos(2 * t)
        vis.update_force([fx, fy, fz, tx, ty, tz])  # 更新力和力矩
        t += 0.1
        vis.show()
        time.sleep(0.05)