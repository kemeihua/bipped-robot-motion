import math
import time
import torch
import numpy as np
from collections import deque
from datetime import datetime
import pickle


class NanoSleep:
    def __init__(self, ms):
        self.duration_sec = ms * 0.001  # 转化为单位秒

    def waiting(self, _start_time):
        while True:
            current_time = time.perf_counter()
            elapsed_time = current_time - _start_time
            if elapsed_time >= self.duration_sec:
                break


class SimBase(object):
    def __init__(self, _cfg, _policy):
        self.cfg = _cfg
        self.policy = _policy

        self.timestep = self.cfg.control.timestep
        self.timestep_ms = self.cfg.control.timestep_ms # step time
        self.timetotal = 0. #  run total time (sec)
        self.timetotal_ms = 0 # run total time (ms)
        self.walk_mode = 1
        # joint target
        self.target_q = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.action = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.action_filter = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.target_joint_pos_scale = self.cfg.robot_config.target_joint_pos_scale
        self.pos0 = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        self.PO_q = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        # obs
        self.q = np.zeros(self.cfg.env.num_actions, dtype=np.float32)
        self.dq = np.zeros(self.cfg.env.num_actions, dtype=np.float32)
        self.euler = np.zeros(2, dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)
        self.hist_obs = deque()
        self.phase_ms = np.floor(self.cfg.control.cycle_time * 0.25 / 0.001)
        self.phase = 0. # [0 - 1]
        self.est_vel = np.zeros(3,dtype=np.double)
        self.cycle_time = self.cfg.control.cycle_time
        for _ in range(self.cfg.env.frame_stack):
            self.hist_obs.append(np.zeros([1, self.cfg.env.num_single_obs], dtype=np.double))

    def ref_trajectory(self, total_ms):
        leg_l = math.sin(2 * math.pi * total_ms * 0.001 / self.cfg.control.cycle_time)  # x * 0.001, ms -> s
        leg_r = math.sin(2 * math.pi * total_ms * 0.001 / self.cfg.control.cycle_time)  # x * 0.001, ms -> s
        scale1 = self.target_joint_pos_scale
        scale2 = 2.*self.target_joint_pos_scale
        ref_dof_pos = np.zeros(self.cfg.env.num_actions, dtype=np.float32)
        ref_dof_pos[2] = self.cfg.robot_config.pos0[2] + leg_l * scale1
        ref_dof_pos[3] = self.cfg.robot_config.pos0[3] - leg_l * scale2
        ref_dof_pos[4] = self.cfg.robot_config.pos0[4] + leg_l * scale1

        ref_dof_pos[7] = self.cfg.robot_config.pos0[7] + leg_r * scale1
        ref_dof_pos[8] = self.cfg.robot_config.pos0[8] - leg_r * scale2
        ref_dof_pos[9] = self.cfg.robot_config.pos0[9] + leg_r * scale1
        return ref_dof_pos

    def gen_traj(self, total_ms):
        t = total_ms * 0.001
        f = 2. * np.pi * 5.
        singe = (math.sin(f*t) + 1.2 * math.sin(f*2*t) + 0.2 * math.sin(f*3*t))  # 0.5*(sin(t) + sin(2t))
        ref_dof_pos = self.cfg.robot_config.q0 + singe * self.cfg.robot_config.q_scale
        return ref_dof_pos

    def run(self):
        pass

    def get_action(self, obs):
        # policy in out
        self.hist_obs.append(obs)
        self.hist_obs.popleft()
        policy_input = np.zeros([1, self.cfg.env.num_observations], dtype=np.float32)
        for i in range(self.cfg.env.frame_stack):
            policy_input[0, i * self.cfg.env.num_single_obs: (i + 1) * self.cfg.env.num_single_obs] = self.hist_obs[i][0, :]
        action = self.policy(torch.tensor(policy_input))[0].detach().numpy()
        action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        # action、qt filter
        if self.cfg.filter_params.add_filter:
            self.action_filter = (1 - self.cfg.filter_params.delay_qt) * self.action_filter + self.cfg.filter_params.delay_qt * action
            self.action = (1 - self.cfg.filter_params.delay_action) * self.action + self.cfg.filter_params.delay_action * action
        else:
            self.action_filter[:] = action[:]
            self.action[:] = action[:]
        # cal target_qt for PD ctrl
        # target_qt = self.action_filter * self.cfg.control.action_scale + self.cfg.robot_config.pos0
        target_qt = self.action_filter * self.cfg.control.action_scale + self.pos0
        return target_qt

    def state_filter(self, q, dq, euler, gyro):
        filter = self.cfg.filter_params
        if filter.add_filter:
            self.q     = (1. - filter.delay_q)     * self.q     + filter.delay_q     * q
            self.dq    = (1. - filter.delay_dq)    * self.dq    + filter.delay_dq    * dq
            self.euler = (1. - filter.delay_euler) * self.euler + filter.delay_euler * euler
            self.gyro  = (1. - filter.delay_gyro)  * self.gyro  + filter.delay_gyro  * gyro
        else:
            self.q[:] = q[:]
            self.dq[:] = dq[:]
            self.euler[:] = euler[:]
            self.gyro[:] = gyro[:]

    def init_robot(self):
        pass

