import numpy as np

class LPF:
    def __init__(self, alpha=0.2):
        """
        初始化油门滤波器
        :param alpha: 滑动平均的权重因子，[0, 1] 范围内，用于控制平滑程度
        """
        self.alpha = alpha
        self.prev_filtered_data = None

    def filter(self, current_data):
        """
        对当前油门值进行平滑滤波和滞后处理
        :param current_data: 当前遥控器油门值
        :return: 过滤后的油门值
        """
        # 初始化时直接返回当前值
        if self.prev_filtered_data is None:
            self.prev_filtered_data = current_data
            return current_data

        # 计算滑动平均
        self.prev_filtered_data = self.alpha * current_data + (1 - self.alpha) * self.prev_filtered_data
        return self.prev_filtered_data