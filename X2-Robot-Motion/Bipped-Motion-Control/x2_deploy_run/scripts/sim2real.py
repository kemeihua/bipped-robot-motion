import os
import sys
# 获取当前脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的绝对路径
project_root = os.path.dirname(script_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import math
import time
from time import sleep

import numpy as np
from tqdm import tqdm
from Config import Config
from base.SimBase import SimBase
from base.SimBase import NanoSleep
from base.DroidGrpcClient import DroidGrpcClient
from utils.Gamepad import GamepadHandler
from utils.LPF import LPF
import onnxruntime as ort

class Sim2Real(SimBase, DroidGrpcClient):
    def __init__(self, _cfg, _policy):
        SimBase.__init__(self, _cfg, _policy)
        DroidGrpcClient.__init__(self, _cfg)
        self.set_joint_ctrl_mode()
        self.pos0 = self.cfg.robot_config.pos0
        self.game_rc = GamepadHandler()
        self.user_cmd = np.zeros(3)
        self.lpf_cmd = LPF(alpha=0.05)
        # self.run_state = 0 # 0: stop、  1: walk、 2：stand and walk
        self.run_state = "stop"  # 0: stop、  1: walk、 2：stand_and_walk
        self.zero_cmd_time = 2.
        self.zero_cmd_enable = 1
        self.is_fall = False
        self.arm_task_cmd = "init"

    def check_lowlevel_init(self):
        while(self.legState.lowlevel_init != 1):
            self.get_robot_state()
            print("waiting low_level init ......", self.legState.lowlevel_init)
            sleep(1)

    def set_joint_ctrl_mode(self):
        self.legCommand.cmd_enable = 2  # joint control mode
        for i in range(self.legActions):
            self.legCommand.kp[i] = self.cfg.robot_config.kps[i]
            self.legCommand.kd[i] = self.cfg.robot_config.kds[i]
            self.legCommand.max_torque[i] = self.cfg.robot_config.tau_limit[i]

    def set_stand_pd(self):
        for idx in range(self.legActions):
            self.legCommand.kp[idx] = self.cfg.robot_config.kpStand[idx]
            self.legCommand.kd[idx] = self.cfg.robot_config.kdStand[idx]

    def set_walk_pd(self):
        for idx in range(self.legActions):
            self.legCommand.kp[idx] = self.cfg.robot_config.kps[idx]
            self.legCommand.kd[idx] = self.cfg.robot_config.kds[idx]

    def set_fall_pd(self):
        for idx in range(self.legActions):
            self.legCommand.kp[idx] = self.cfg.robot_config.kpFall[idx]
            self.legCommand.kd[idx] = self.cfg.robot_config.kdFall[idx]

    def init_robot(self):
        self.check_lowlevel_init()
        self.set_stand_pd()
        self.joint_plan(1, self.cfg.robot_config.stand_pos0)
        timer = NanoSleep(self.cfg.control.decimation)  # 创建一个10毫秒的NanoSleep对象
        self.get_robot_state()
        print("LT, RT同时按下进入run")
        temp_tic = self.legState.system_tic
        while (self.game_rc.state.LT <= 250 or self.game_rc.state.RT <= 250):
            if self.legState.system_tic - temp_tic > 1000:
                temp_tic = self.legState.system_tic
                print(f"IMU: {self.legState.imu_euler[0]:.3f}, {self.legState.imu_euler[1]:.3f}, {self.legState.imu_euler[2]:.3f}")
            start_time = time.perf_counter()
            self.get_robot_state()
            self._update_run_state()
            self.arm_task_str_with_dance(self.arm_task_cmd)
            self.do_arm_task()
            self.set_arm_command()
            timer.waiting(start_time)
        print("init ok")
        self.run_state = "stop"
        self.set_walk_pd()

    def _update_cmd(self):
        en_vx = self.game_rc.state.LT > 64
        en_yaw = self.game_rc.state.RT > 64
        cmd = np.zeros(3)

        self.cfg.robot_config.euler0[1] = -0.5 * np.pi / 180. * self.game_rc.state.LEFT_Y if self.game_rc.state.LEFT_Y > 0 else 0.0
        vx_gain = 1.0 if self.game_rc.state.LEFT_Y < 0 else 3.0
        cmd[0] = np.clip(self.game_rc.state.LEFT_Y * vx_gain * en_vx, -1.0, 3.0)
        cmd[1] = np.clip(self.game_rc.state.RIGHT_X * 0.5, -0.5, 0.5)
        cmd[2] = np.clip(self.game_rc.state.LEFT_X, -1.0, 1.0)
        self.user_cmd = self.lpf_cmd.filter(cmd)

        # if self.phase_ms == 0.0:
        #     if np.abs(self.user_cmd[0]) < 1.0:
        #         self.cycle_time = self.cfg.control.cycle_time
        #     elif 1.0 <= np.abs(self.user_cmd[0]) < 1.5:
        #         self.cycle_time = 0.7
        #     elif 1.5 <= np.abs(self.user_cmd[0]):
        #         self.cycle_time = 0.64


    def _update_run_state(self):
        if self.game_rc.state.LT > 64 and self.game_rc.state.LB:
            self.run_state = "stop"
            self.arm_task_cmd = "init"
        if self.game_rc.state.LT > 64 and self.game_rc.state.START:
            self.run_state = "walk"
            self.arm_task_cmd = "init"
        if self.game_rc.state.LT > 64 and self.game_rc.state.BACK:
            self.run_state = "walk_and_stand"
            self.arm_task_cmd = "init"
        self._get_arm_cmd_rc()

    def _get_arm_cmd_rc(self):
        state = self.game_rc.state
        if state.RT > 64:
            if state.Y:
                self.arm_task_cmd = "action1"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.X:
                self.arm_task_cmd = "action2"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.A:
                self.arm_task_cmd = "action3"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.B:
                self.arm_task_cmd = "action4"
                if self.armStatus == "done":
                    self.do_again = True
        elif state.RB:
            if state.Y:
                self.arm_task_cmd = "good"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.X:
                self.arm_task_cmd = "ye"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.A:
                self.arm_task_cmd = "dance"
                self.dancing = "action1"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.B:
                self.arm_task_cmd = "jiayou"
                if self.armStatus == "done":
                    self.do_again = True
        else:
            if state.Y:
                self.arm_task_cmd = "init"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.X:
                self.arm_task_cmd = "hello"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.A:
                self.arm_task_cmd = "handshake"
                if self.armStatus == "done":
                    self.do_again = True
            elif state.BACK == False:
                if state.B:
                    self.arm_task_cmd = "guzhang"
                    if self.armStatus == "done":
                        self.do_again = True

    def do_task(self):
        if self.run_state == "stop":  # stop
            self._stop_task()
        elif self.run_state == "walk":  # walk
            self._walk_task()
        elif self.run_state == "walk_and_stand":  # walk_and_stand
            self._walk_and_stand_task()

    def _stop_task(self):
        self.phase_ms = np.floor(self.cycle_time * 0.25 / 0.001)
        self.walk_mode = 0
        self.action[:] = 0.
        self.set_stand_pd()
        self.legCommand.position[:] = self.cfg.robot_config.stand_pos0[:]
        self.arm_task_str_with_dance(self.arm_task_cmd)
        self.do_arm_task()

    def _walk_task(self):
        self.walk_mode = 1
        self.set_walk_pd()
        self.legCommand.position[:] = self.target_q[:]
        self.arm_task_str_with_dance(self.arm_task_cmd)
        if self.armSc == "init" and self.armStatus == "done":
            self.walk_arm_swing(self.legState)
        else:
            self.do_arm_task()

    def _walk_and_stand_task(self):
        cmd_vel_norm = np.sum(np.abs(self.user_cmd))
        est_vel_norm = np.sqrt(self.est_vel[0] ** 2 + self.est_vel[1] ** 2)
        if cmd_vel_norm < 0.10: # 速度cmd为0，等待2步进入stand
            self.zero_cmd_time += self.cfg.control.decimation * 0.001
            if self.zero_cmd_time > 1.5 * self.cycle_time:
                if 0.24 < self.phase < 0.26 or 0.74 < self.phase < 0.76:
                    self.zero_cmd_enable = 1
            else:
                self.zero_cmd_enable = 0
        else:
            self.zero_cmd_time = 0.
            self.zero_cmd_enable = 0

        if cmd_vel_norm < 0.10 and est_vel_norm < 0.15 and self.zero_cmd_enable: # stand
            self.walk_mode = 0
            if est_vel_norm > 0.1:
                self._chose_first_leg("stand")
            else:
                self._chose_first_leg("walk")
            if cmd_vel_norm > 0.02:
                self._chose_first_leg("walk")

            self.arm_task_str(self.arm_task_cmd)
            self.do_arm_task()
        else: # walk
            self.walk_mode = 1
            self.arm_task_str(self.arm_task_cmd)
            if self.armSc == "init" and self.armStatus == "done":
                self.walk_arm_swing(self.legState)
            else:
                self.do_arm_task()
        self.set_walk_pd()
        self.legCommand.position[:] = self.target_q[:]

    def _chose_first_leg(self, mode:str):
        if mode=="stand":
            if self.est_vel[1] > 0:
                self.phase_ms = np.floor(self.cycle_time * 0.5 / 0.001)
            else:
                self.phase_ms = 0
        elif mode=="walk":
            if self.est_vel[1] > 0:
                self.phase_ms = np.floor(self.cycle_time * 0.75 / 0.001)
            else:
                self.phase_ms = np.floor(self.cycle_time * 0.25 / 0.001)

    def safety_for_fall(self):
        if np.abs(self.euler[0]) > 0.6 or np.abs(self.euler[1]) > 0.6:
            self.is_fall = True

        if self.is_fall:
            self.phase_ms = 0
            self.walk_mode = 0
            self.action[:] = 0.
            self.set_fall_pd()
            self.legCommand.position[:] = self.legState.position[:]

    def get_obs(self):
        self.q = np.array(self.legState.position)
        self.dq = np.array(self.legState.velocity)
        self.gyro = np.array(self.legState.imu_gyro)
        euler = np.array(self.legState.imu_euler)[:2]
        euler[euler > math.pi] -= 2 * math.pi
        self.euler = euler - self.cfg.robot_config.euler0

        obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
        _phase = 2 * math.pi * self.phase
        user_cmd =  (self.user_cmd + self.cfg.robot_config.cmd_bias) * self.walk_mode
        obs[0, 0] = self.walk_mode * math.sin(_phase)
        obs[0, 1] = self.walk_mode * math.cos(_phase)
        obs[0, 2] = user_cmd[0] * self.cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = user_cmd[1] * self.cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = user_cmd[2] * self.cfg.normalization.obs_scales.ang_vel
        obs[0, 5:15] = (self.q - self.pos0) * self.cfg.normalization.obs_scales.dof_pos
        obs[0, 15:25] = self.dq * self.cfg.normalization.obs_scales.dof_vel
        obs[0, 25:35] = self.action
        obs[0, 35:38] = self.gyro  * self.cfg.normalization.obs_scales.ang_vel
        obs[0, 38:40] = self.euler * self.cfg.normalization.obs_scales.quat
        obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return obs

    def get_action(self, obs):
        # policy in out
        self.hist_obs.append(obs)
        self.hist_obs.popleft()
        policy_input = np.zeros([1, self.cfg.env.num_observations], dtype=np.float32)
        for i in range(self.cfg.env.frame_stack):
            policy_input[0, i * self.cfg.env.num_single_obs: (i + 1) * self.cfg.env.num_single_obs] = self.hist_obs[i][0, :]
        # onnx
        # action, est = self.policy.run(output_names=["action", "est"], input_feed={'input': policy_input})
        # action = action[0]  # 对 action 张量调用 detach()
        # self.est_vel = est[0] / self.cfg.normalization.obs_scales.lin_vel

        action = self.policy.run(output_names=["action"], input_feed={'input': policy_input})
        action = action[0]  # 对 action 张量调用 detach()
        self.est_vel = np.ones(3, dtype=np.double)

        action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        self.action[:] = action[:]

        target_qt = self.action * self.cfg.control.action_scale + self.pos0 + self.cfg.robot_config.pos0_bias
        # target_qt = np.clip(target_qt, self.cfg.robot_config.clip_actions_lower, self.cfg.robot_config.clip_actions_upper)
        return target_qt

    def run(self):
        self.set_walk_pd()
        pre_tic = 0
        timer = NanoSleep(self.cfg.control.decimation)  # 创建一个10毫秒的NanoSleep对象
        pbar = tqdm(range(int(self.cfg.env.run_duration / (self.cfg.control.decimation * 0.001))),
                    desc="x02 running...")  # x * 0.001, ms -> s
        start = time.perf_counter()
        for _ in pbar:
            start_time = time.perf_counter()
            # 获取机器人状态
            self.get_robot_state()
            # 根据遥控器值，更新user_cmd、run_state
            self._update_cmd()
            self._update_run_state()
            # 更新策略obs、action
            obs= self.get_obs()
            self.target_q = self.get_action(obs)
            # 根据run_state，执行对应的task
            self.do_task()
            # 摔倒保护
            self.safety_for_fall()
            # 写机器人位置
            self.set_robot_command()

            pbar.set_postfix(
                realCycle=f"{self.legState.system_tic - pre_tic}ms",  # 实际循环周期，单位毫秒
                calculateTime=f"{(time.perf_counter() - start_time) * 1000:.3f}ms",  # 计算用时，单位毫秒
                runTime=f"{(time.perf_counter() - start):.3f}s"  # 运行时间，单位秒
            )
            pre_tic = self.legState.system_tic
            timer.waiting(start_time)
            self.phase_ms += self.cfg.control.decimation
            if self.phase_ms >= 1000 * self.cycle_time:
                self.phase_ms = 0
            self.phase = (self.phase_ms * 0.001 / self.cycle_time) % 1

        self.joint_plan(1, self.cfg.robot_config.pos0)


if __name__ == '__main__':
    mode_path = Config.robot_config.mode_path
    print("load mode = ", mode_path)
    # onnx
    policy = ort.InferenceSession(mode_path)
    mybot = Sim2Real(Config_run, policy)
    mybot.init_robot()
    mybot.run()
