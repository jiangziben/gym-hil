import matplotlib.pyplot as plt
import numpy as np

class RealTimeImagePlotter:
    def __init__(self, title1="Image 1", title2="Image 2"):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.ax1.set_title(title1)
        self.ax2.set_title(title2)

        # 初始化为黑图
        self.image1 = self.ax1.imshow(np.zeros((100, 100), dtype=np.uint8), cmap='gray', vmin=0, vmax=255)
        self.image2 = self.ax2.imshow(np.zeros((100, 100), dtype=np.uint8), cmap='gray', vmin=0, vmax=255)

        plt.tight_layout()
        plt.ion()  # 打开交互模式
        plt.show()

    def update(self, img1, img2):
        """
        更新图像内容。
        :param img1: numpy.ndarray, 第一幅图像 (2D灰度图或3通道RGB)
        :param img2: numpy.ndarray, 第二幅图像 (2D灰度图或3通道RGB)
        """
        self.image1.set_data(img1)
        self.image2.set_data(img2)

        # 如果图像大小变化，重新设置轴范围
        if img1.shape != self.image1.get_array().shape:
            self.image1.set_extent((0, img1.shape[1], img1.shape[0], 0))
        if img2.shape != self.image2.get_array().shape:
            self.image2.set_extent((0, img2.shape[1], img2.shape[0], 0))

        self.ax1.draw_artist(self.ax1.patch)
        self.ax1.draw_artist(self.image1)
        self.ax2.draw_artist(self.ax2.patch)
        self.ax2.draw_artist(self.image2)

        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def close(self):
        plt.ioff()
        plt.close(self.fig)

import time

if __name__ == "__main__":
    plotter = RealTimeImagePlotter()

    for i in range(100):
        # 生成两张测试图像
        img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = np.full((100, 100), i % 255, dtype=np.uint8)

        plotter.update(img1, img2)
        time.sleep(0.05)

    plotter.close()
