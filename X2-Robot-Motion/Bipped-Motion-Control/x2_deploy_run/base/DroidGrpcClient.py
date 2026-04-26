import math
import time
from grpc import insecure_channel
from base.ArmBase_0dof import ArmBase
from base.LegBase import LegBase
class NanoSleep:
    def __init__(self, ms):
        self.duration_sec = ms * 0.001  # 转化为单位秒

    def waiting(self, _start_time):
        while True:
            current_time = time.perf_counter()
            elapsed_time = current_time - _start_time
            if elapsed_time >= self.duration_sec:
                break

class DroidGrpcClient(ArmBase, LegBase):
    def __init__(self, _cfg):
        ArmBase.__init__(self, _cfg)
        LegBase.__init__(self, _cfg)
        self.robot_actions = self.armActions + self.legActions

    def get_robot_config(self):
        self.get_arm_config()
        self.get_leg_config()

    def get_robot_state(self):
        self.get_arm_state()
        self.get_leg_state()

    def set_robot_command(self):
        self.set_arm_command()
        self.set_leg_command()

    def joint_plan(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = [0.0] * self.legActions
        for idx in range(self.legActions):
            q0[idx] = self.legState.position[idx]
        timer = NanoSleep(2)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            self.get_robot_state()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            for idx in range(self.legActions):  # 假设关节数量是18
                qt = s0 * q0[idx] + s1 * qd[idx]
                self.legCommand.position[idx] = qt
            self.set_robot_command()
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长


if __name__ == '__main__':
    channel = insecure_channel('192.168.55.13:50051')
    gBot = DroidGrpcClient(channel)
    time.sleep(2)
    gBot.get_robot_config()
    gBot.get_robot_state()

    T = 0.6  # 总时间
    dt0 = [0.] * 18  # 假设 NMC 是一个定义好的常量，表示关节数量
    dt1 = [0.] * 18  # 创建一个 NMC 长度的列表，初始值为 0
    dt2 = [0.] * 18
    D2R = math.pi / 180.0

    dt1[1] = 10 * D2R
    dt1[2] = 60 * D2R
    dt1[3] = -80 * D2R
    dt1[4] = -80 * D2R
    dt1[10] = 0 * D2R
    dt1[13] = 100 * D2R
    dt1[14] = 0.0 * D2R
    dt1[17] = 100 * D2R

    dt2[6] = 10 * D2R
    dt2[7] = 60 * D2R
    dt2[8] = -80 * D2R
    dt2[9] = -80 * D2R
    dt2[10] = 0 * D2R
    dt2[13] = 100 * D2R
    dt2[14] = 0.0 * D2R
    dt2[17] = 100 * D2R

    for i in range(10000):
        print("wave round %d" % (i * 2 + 1))
        dt1[10] = gBot.legState.position[2]-0.5
        dt2[10] = gBot.legState.position[7]-0.5
        dt1[14] = gBot.legState.position[7]-0.5
        dt2[14] = gBot.legState.position[2]-0.5
        gBot.joint_plan(T, dt1)
        print("wave round %d" % (i * 2 + 2))
        gBot.joint_plan(T, dt2)
    print("return to zero")
    gBot.joint_plan(T, dt0)
