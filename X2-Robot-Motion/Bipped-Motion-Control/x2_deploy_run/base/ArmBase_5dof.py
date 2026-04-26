import os
import sys
from base.Base import *
from grpc import insecure_channel
from scripts.Config import Config
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../protos'))
from protos import arm_service_pb2_grpc as arm_pb2_grpc
from protos import droid_msg_pb2 as msg_pb2
from base.read_csv import load_action_sequences
import csv

class ArmBase:
    def __init__(self, _cfg):
        self.cfg = _cfg
        self.armActions = self.cfg.env.num_arm_actions
        # grpc defines
        self.armConfigs = msg_pb2.DroidConfigs()
        self.armState = msg_pb2.DroidArmResponse()
        self.armCommand = msg_pb2.DroidCommandRequest()
        channel = insecure_channel(self.cfg.env.grpc_channel + ":50052")
        self.armStub = arm_pb2_grpc.ArmServiceStub(channel)
        init_command(self.armCommand, self.cfg.env.num_arm_actions)
        for idx in range(12):
            self.armCommand.finger.append(10)
        # 建立通信，获取机器人底层信息
        self.get_arm_config()
        self.get_arm_state()
        for idx in range(self.armActions):
            self.armCommand.position[idx] = self.armState.position[idx]
        # 电机空间控制或关节空间控制
        set_motor_mode(self.armCommand, self.armConfigs)

        # path
        self.dt = 0.01  # step时间步长
        self.td = 0.    # 轨迹结束时间
        self.tt = 0.    # 轨迹当前时间
        self.N: int = 0   # 动作的总行数
        self.Nt: int = 0  # 给定行数
        self.Nc: int = -1  # 当前行数
        self.armSt = "init"  # 给定状态
        self.armSc = "none"  # 当前状态
        self.armStatus = "doing" # "doing"
        self.do_again = False
        self.qc = np.zeros(self.armActions, dtype=float)
        self.q0 = np.zeros(self.armActions, dtype=float)
        self.qd = np.zeros(self.armActions, dtype=float)
        self.qt = np.zeros(self.armActions, dtype=float)

        for idx in range(self.armActions):
            self.qc[idx] = self.armState.position[idx]
        self.qt[:] = self.qc[:]

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.action_lib = load_action_sequences(script_dir + "/arm_action_lib/5dof")
        self.finger_lib = load_action_sequences(script_dir + "/arm_action_lib/finger")
        self.traj_lib = load_action_sequences(script_dir + "/arm_action_lib/traj")
        # self.open_file()
        self.time_cnt = 0
        self.dancing = "none"

    def get_arm_config(self):
        empty_request = msg_pb2.Empty()
        self.armConfigs = self.armStub.GetArmConfig(empty_request)

    def get_arm_state(self):
        empty_request = msg_pb2.Empty()
        self.armState = self.armStub.GetArmState(empty_request)
        # print_state(self.armState, self.armConfigs)

    def set_arm_command(self):
        response = self.armStub.SetArmCommand(self.armCommand)
        if not response:  # Assuming the RPC method returns a response
            print("RPC failed")

    def set_arm_path(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = [0.0] * self.armActions
        for idx in range(self.armActions):
            q0[idx] = self.armState.position[idx]
        timer = NanoSleep(2)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            self.get_arm_state()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0

            for idx in range(self.armActions):  # 假设关节数量是18
                qt = s0 * q0[idx] + s1 * qd[idx]
                self.armCommand.position[idx] = qt
            self.set_arm_command()
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def testArm(self):
        T = 2  # 总时间
        D2R = math.pi / 180.0
        dt0 = [-30, 10, 0, 80, -100, -30, 10, 0, 80, -100]  # 假设 NMC 是一个定义好的常量，表示关节数量
        dt0 = [x * D2R for x in dt0]
        # 填充 dt1 和 dt2 列表
        dt1 = [-30 * D2R, 10 * D2R, 0, 100 * D2R, -100 * D2R, 30 * D2R, 10 * D2R, 0, 100 * D2R, -100 * D2R]
        dt2 = [30 * D2R, 10 * D2R, 0, 100 * D2R, -100 * D2R, -30 * D2R, 10 * D2R, 0, 100 * D2R, -100 * D2R]

        # 执行关节规划
        for i in range(2):
            print("wave round %d" % (i * 2 + 1))
            for idx in range(12):
                self.armCommand.finger[idx] = 50
            self.set_arm_path(T, dt1)
            print("wave round %d" % (i * 2 + 2))
            for idx in range(12):
                self.armCommand.finger[idx] = 5
            self.set_arm_path(T, dt2)
        print("return to zero")
        gBot.set_arm_path(T, dt0)

    def setTraj(self, td, qd):
        self.tt = 0.
        self.td = td
        self.q0[:] = self.qt[:]
        self.qd[:] = qd[:]

    def getTraj(self):
        st = min(self.tt / self.td, 1.0)
        s0 = 0.5 * (1.0 + math.cos(math.pi * st))
        s1 = 1 - s0
        self.qt = s0 * self.q0 + s1 * self.qd
        for idx in range(self.armActions):
            self.armCommand.position[idx] = self.qt[idx]


    def setFinger(self, action):
        for idx in range(12):
            self.armCommand.finger[idx] = int(action[idx])

    def do_action(self, action):
        self.N = action.shape[0]
        if self.Nt <= self.N-1:    #动作结束判断
            if self.Nc != self.Nt: #行数更新，重新设置轨迹
                self.Nc = self.Nt
                self.tt = 0
                for i in range(self.armActions):
                    self.qt[i] = self.armCommand.position[i]
                self.setTraj(action[self.Nc][0], action[self.Nc][1:1+self.armActions])
            else:                  #计算轨迹
                self.tt += self.dt
                self.getTraj()
                if self.tt > action[self.Nc][0]: # 当前行轨迹执行完成，进入下一行
                    self.Nt += 1
            self.armStatus = "doing"
        else:
            self.armStatus = "done"
        # print("St=%3d Sc=%3d Nt=%3d Nc=%3d tt=%7.3f qt=%7.3f" %(self.St.value, self.Sc.value, self.Nt, self.Nc, self.tt, self.qt[1]))
    def do_traj(self, action):
        self.N = action.shape[0]
        _T = 1.0
        # 第一行规划
        if self.Nt == 0:
            if self.Nc != self.Nt: #行数更新，重新设置轨迹
                self.Nc = self.Nt
                self.tt = 0
                for i in range(self.armActions):
                    self.qt[i] = self.armCommand.position[i]
                self.setTraj(_T, action[0][1:1+self.cfg.env.num_actions])
            else:                  #计算轨迹
                self.tt += 0.02
                self.getTraj()
                if self.tt > _T: # 当前行轨迹执行完成，进入下一行
                    self.Nt += 1
            self.armStatus = "doing"
        elif self.N-1 >= self.Nt >= 1:    #动作结束判断
            joint_pt = action[self.Nt]
            for idx in range(self.armActions):
                self.armCommand.position[idx] = 0.9 * self.armCommand.position[idx] + 0.1 * joint_pt[1+idx]
            self.Nt +=1
            self.armStatus = "doing"
        else:
            self.armStatus = "done"

    def arm_task_str(self, arm_mode):
        if arm_mode == "action1" or arm_mode == "action2" or arm_mode == "action3" or arm_mode == "action4" or arm_mode == "dance":
            return
        self.armSt = arm_mode
        if self.armSc != self.armSt or self.do_again: #状态更新，重新初始化动作的行数
            self.armStatus = "doing"
            self.armSc = self.armSt
            self.do_again = False
            self.tt = 0.
            self.Nt = 0
            self.Nc = -1

    def arm_task_str_with_dance(self, arm_mode):
        if arm_mode == "dance":
            if self.dancing == "action1":
                self.armSt = "action1"
                if self.armSc == "action1" and self.armStatus == "done":
                    self.dancing = "action2"
            elif self.dancing == "action2":
                self.armSt = "action2"
                if self.armSc == "action2" and self.armStatus == "done":
                    self.dancing = "action3"
            elif self.dancing == "action3":
                self.armSt = "action3"
                if self.armSc == "action3" and self.armStatus == "done":
                    self.dancing = "action4"
            elif self.dancing == "action4":
                self.armSt = "action4"
                self.dancing = "none"
        else:
            self.armSt = arm_mode

        if self.armSc != self.armSt or self.do_again: #状态更新，重新初始化动作的行数
            self.armStatus = "doing"
            self.armSc = self.armSt
            self.do_again = False
            self.tt = 0.
            self.Nt = 0
            self.Nc = -1

    def arm_task(self, state):
        if state.RB:
            if state.Y:
                self.armSt = "good"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.X:
                self.armSt = "ye"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True
        else:
            if state.Y:
                self.armSt = "init"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.X:
                self.armSt = "hello"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.A:
                self.armSt = "handshake"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True

        if self.armSc != self.armSt or self.do_again: #状态更新，重新初始化动作的行数
            self.armSc = self.armSt
            self.tt = 0.
            self.Nt = 0
            self.Nc = -1
            self.armStatus = "doing"
            self.do_again = False

    def arm_task_with_dance(self, state):
        if state.RT > 64:
            self.dancing = "none"
            if state.Y:
                self.armSt = "action1"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.X:
                self.armSt = "action2"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.A:
                self.armSt = "action3"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.B:
                self.armSt = "action4"
                if self.armStatus == "done":
                    self.do_again = True
        elif state.RB:
            if state.Y:
                self.armSt = "good"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.X:
                self.armSt = "ye"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.A:
                self.dancing = "action1"
                if self.armStatus == "done":
                    self.do_again = True
        else:
            if state.Y:
                self.armSt = "init"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.X:
                self.armSt = "hello"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.A:
                self.armSt = "handshake"
                self.dancing = "none"
                if self.armStatus == "done":
                    self.do_again = True

        if self.dancing == "action1":
            self.armSt = "action1"
            if self.armSc == "action1" and self.armStatus == "done":
                self.dancing = "action2"
        elif self.dancing == "action2":
            self.armSt = "action2"
            if self.armSc == "action2" and self.armStatus == "done":
                self.dancing = "action3"
        elif self.dancing == "action3":
            self.armSt = "action3"
            if self.armSc == "action3" and self.armStatus == "done":
                self.dancing = "action4"
        elif self.dancing == "action4":
            self.armSt = "action4"

        if self.armSc != self.armSt or self.do_again: #状态更新，重新初始化动作的行数
            self.armSc = self.armSt
            self.tt = 0.
            self.Nt = 0
            self.Nc = -1
            self.armStatus = "doing"
            self.do_again = False

    def do_arm_task(self):

        if self.armSc == "init":
            self.do_action(self.action_lib["init"].copy())
            self.setFinger(self.finger_lib["init"][0])
        elif self.armSc == "hello":
            self.do_action(self.action_lib["hello"].copy())
            self.setFinger(self.finger_lib["hello"][0])
        elif self.armSc == "handshake":
            self.do_action(self.action_lib["handshake"].copy())
            self.setFinger(self.finger_lib["handshake"][0])
        elif self.armSc == "good":
            self.do_action(self.action_lib["good"].copy())
            self.setFinger(self.finger_lib["good"][0])
        elif self.armSc == "ye":
            self.do_action(self.action_lib["ye"].copy())
            self.setFinger(self.finger_lib["ye"][0])
        elif self.armSc == "jiayou":
            self.do_action(self.action_lib["jiayou"].copy())
            self.setFinger(self.finger_lib["jiayou"][0])
        elif self.armSc == "guzhang":
            self.do_action(self.action_lib["guzhang"].copy())
            self.setFinger(self.finger_lib["guzhang"][0])

        elif self.armSc == "action1":
            if self.time_cnt%2 == 0:
                self.do_traj(self.traj_lib["action1"].copy())
            self.setFinger(self.finger_lib["init"][0])
        elif self.armSc == "action2":
            if self.time_cnt%2 == 0:
                self.do_traj(self.traj_lib["action2"].copy())
            self.setFinger(self.finger_lib["init"][0])
        elif self.armSc == "action3":
            if self.time_cnt%2 == 0:
                self.do_traj(self.traj_lib["action3"].copy())
            self.setFinger(self.finger_lib["init"][0])
        elif self.armSc == "action4":
            self.do_action(self.traj_lib["action4"].copy())
            self.setFinger(self.finger_lib["init"][0])
        self.time_cnt += 1

    def walk_arm_swing(self, legState):
        leg_hip_left = legState.position[2]
        leg_hip_right = legState.position[2+5]
        hip_pos_error = (leg_hip_left - leg_hip_right) * 0.5
        arm_hip_left = -hip_pos_error + self.action_lib["init"][0][1]
        arm_hip_right =  hip_pos_error + self.action_lib["init"][0][1+5]
        self.armCommand.position[0] = arm_hip_left
        self.armCommand.position[5] = arm_hip_right

    def walk_arm_swing_6dof(self, legState):
        leg_hip_left = legState.position[2]
        leg_hip_right = legState.position[2+6]
        hip_pos_error = (leg_hip_left - leg_hip_right) * 0.5
        arm_hip_left = -hip_pos_error + self.action_lib["init"][0][1]
        arm_hip_right =  hip_pos_error + self.action_lib["init"][0][1+5]
        self.armCommand.position[0] = arm_hip_left
        self.armCommand.position[5] = arm_hip_right

    def open_file(self):
        self.file = open('/home/liangzhiyuan/RL/X02/x02-sim2real_new_grpc/base/arm_action_lib/5dof/ye_1.csv', mode='a', newline='')
        self.writer = csv.writer(self.file)

    # def write_file(self):
    #     joint_positions = self.armCommand.position[:]  # 获取关节位置目标值
    #     formatted_positions = [f"{pos:.4f}".center(10) for pos in joint_positions]
    #     self.writer.writerow(formatted_positions)  # 写入一行数据
    #     self.file.flush()  # 确保数据写入磁盘

    def write_file(self):
        joint_positions = np.zeros(11)
        for i in range(self.action_lib["ye"].shape[0]):
            joint_positions[:] = self.action_lib["ye"][i]  # 获取关节位置目标值
            joint_positions[1:] *= np.pi/180.
            formatted_positions = [f"{pos:.4f}".center(10) for pos in joint_positions]
            self.writer.writerow(formatted_positions)  # 写入一行数据
            self.file.flush()  # 确保数据写入磁盘

if __name__ == '__main__':
    gBot = ArmBase(Config)
    gBot.testArm()


