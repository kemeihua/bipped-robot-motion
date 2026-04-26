import math
import time
import mujoco
import mujoco_viewer
import numpy as np
from tqdm import tqdm
from Config import Config
from base.SimBase import SimBase
from base.SimBase import NanoSleep
import glfw
import onnxruntime as ort

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y])  # , yaw_z

def quat_rotate_inverse(q, v):
    """
    q: length-4 array [x, y, z, w]
    v: length-3 array [vx, vy, vz]
    return: rotated vector (length-3)
    """
    q_vec = q[:3]        # [x, y, z]
    q_w   = q[3]         # w

    a = v * (2.0 * q_w**2 - 1.0)
    b = 2.0 * q_w * np.cross(q_vec, v)
    c = 2.0 * q_vec * np.dot(q_vec, v)

    return a - b + c

class Sim2Sim(SimBase):
    def __init__(self, _cfg, _policy):
        super().__init__(_cfg, _policy)
        self.model = mujoco.MjModel.from_xml_path(self.cfg.sim_config.mujoco_model_path)
        self.model.opt.timestep = self.timestep
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.pos0 = self.cfg.robot_config.pos0_sim
        self.kp = self.cfg.robot_config.kpSim[:]
        self.kd = self.cfg.robot_config.kdSim[:]
        self.zero_cmd_time = 2.
        self.zero_cmd_enable = 1
        # 设置默认视角
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # 自由视角
        # 设置自定义视角
        self.viewer.cam.lookat = [0, 0, 1]  # 注视点位置
        self.viewer.cam.distance = 3.0  # 相机距离
        self.viewer.cam.azimuth = 90  # 方位角
        self.viewer.cam.elevation = -20  # 仰角

    def get_obs(self):
        self.q = self.data.qpos.astype(np.double)[-self.cfg.env.num_actions:]
        self.dq = self.data.qvel.astype(np.double)[-self.cfg.env.num_actions:]
        quat = self.data.qpos[3:7].astype(np.double)
        quat[:] = quat[[1, 2, 3, 0]]
        self.gyro = self.data.qvel[3:6].astype(np.double)
        euler = quaternion_to_euler_array(quat)
        euler[euler > math.pi] -= 2 * math.pi
        self.euler = euler
        # add noise
        if self.cfg.noise_scales.add_noise:
            self.q += (2.*np.random.rand(self.cfg.env.num_actions)-1.) * self.cfg.noise_scales.dof_pos
            self.dq += (2.*np.random.rand(self.cfg.env.num_actions)-1.) * self.cfg.noise_scales.dof_vel
            self.euler += (2.*np.random.rand(2)-1.) * self.cfg.noise_scales.euler
            self.gyro += (2.*np.random.rand(3)-1.) * self.cfg.noise_scales.ang_vel

        obs = np.zeros([1, self.cfg.env.num_single_obs], dtype=np.float32)
        _phase = 2 * math.pi * self.phase
        obs[0, 0] = self.walk_mode * math.sin(_phase)  # x * 0.001, ms -> s
        obs[0, 1] = self.walk_mode * math.cos(_phase)  # x * 0.001, ms -> s
        obs[0, 2] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = self.cfg.cmd.yaw * self.cfg.normalization.obs_scales.ang_vel
        obs[0, 5:15] = (self.q - self.pos0[:10]) * self.cfg.normalization.obs_scales.dof_pos
        obs[0, 15:25] = self.dq * self.cfg.normalization.obs_scales.dof_vel
        obs[0, 25:35] = self.action
        obs[0, 35:38] = self.gyro * self.cfg.normalization.obs_scales.ang_vel
        obs[0, 38:40] = self.euler
        obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return obs

    def set_sim_target(self, target_q):
        tau = (target_q - self.q) * self.kp - self.dq * self.kd
        self.data.ctrl = np.clip(tau, -self.cfg.robot_config.tau_limit,
                      self.cfg.robot_config.tau_limit)  # Clamp torques
        mujoco.mj_step(self.model, self.data)
        self.viewer.render()

    def run(self):
        self.start_key()
        self.timetotal_ms = 0
        # pbar = tqdm(range(int(self.cfg.env.run_duration / 0.001)),
        #             desc="x02 Simulating...")  # x * 0.001, ms -> s
        pbar = range(int(self.cfg.env.run_duration / 0.001))
        start = time.perf_counter()
        for _ in pbar:
            start_time = time.perf_counter()
            # Obtain an observation
            obs= self.get_obs()
            # 1000hz -> 100hz
            if self.timetotal_ms % self.cfg.control.decimation == 0:
                self.PO_q = self.get_action(obs)  # 策略推理

                if self.cfg.cmd.stand == 1:
                    self.walk_mode = 1
                    self.kp = self.cfg.robot_config.kpSim[:]
                    self.kd = self.cfg.robot_config.kdSim[:]
                    self.target_q = self.PO_q[:]
                elif self.cfg.cmd.stand == 2:
                    vel_norm = np.sqrt(self.cfg.cmd.vx ** 2 + self.cfg.cmd.vy ** 2 + self.cfg.cmd.yaw ** 2)
                    est_vel_norm = np.sqrt(self.est_vel[0] ** 2 + self.est_vel[1] ** 2)
                    if vel_norm < 0.10:
                        self.zero_cmd_time += self.cfg.control.decimation * 0.001
                        if self.zero_cmd_time > 1.5 * self.cycle_time:
                            if 0.24 < self.phase < 0.26 or 0.74 < self.phase < 0.76:
                                self.zero_cmd_enable = 1
                        else:
                            self.zero_cmd_enable = 0
                    else:
                        self.zero_cmd_time = 0.
                        self.zero_cmd_enable = 0

                    if vel_norm < 0.10 and est_vel_norm < 10.15 and self.zero_cmd_enable:
                        self.walk_mode = 0
                        if self.est_vel[1] > 0:
                            self.phase_ms = np.floor(self.cycle_time/2./0.001)
                        else:
                            self.phase_ms = 0
                    else:
                        self.walk_mode = 1
                    self.kp = self.cfg.robot_config.kpSim[:]
                    self.kd = self.cfg.robot_config.kdSim[:]
                    self.target_q = self.PO_q[:]

                    now = time.perf_counter()
                    # pbar.set_postfix(
                    #     calculateTime=f"{(now - start_time) * 1000:.3f}ms",  # 计算用时，单位毫秒
                    #     runTime=f"{(now - start):.3f}s"  # 运行时间，单位秒
                    # )

            self.set_sim_target(self.target_q)
            self.timetotal_ms += self.timestep_ms
            self.phase_ms += self.timestep_ms
            if self.phase_ms >= 1000 * self.cycle_time:
                self.phase_ms = 0
            self.phase = (self.phase_ms * 0.001 / self.cycle_time) % 1

            # show
            quat = self.data.qpos[3:7].astype(np.double)
            quat[:] = quat[[1, 2, 3, 0]]
            vel_world = self.data.qvel[0:3].astype(np.double)
            vel_base = quat_rotate_inverse(quat, vel_world)
            print(f"\r time: {self.timetotal_ms/1000.:5.2f} | cmd_vx: {self.cfg.cmd.vx:5.2f}| vel: {vel_base[0]:5.2f} | "
                  f"dq_L: {self.dq[1]:5.2f} {self.dq[2]:5.2f} {self.dq[3]:5.2f} {self.dq[4]:5.2f} | dq_R: {self.dq[6]:5.2f} {self.dq[7]:5.2f} {self.dq[8]:5.2f} {self.dq[9]:5.2f} | "
                  f"q_knee: {self.q[3]:5.2f} {self.q[8]:5.2f}"
                  ,end="")

        self.viewer.close()

    def joint_plan(self, T, qd):
        s0, s1, st = 0.0, 0.0, 0.0
        tt = 0.0
        dt = 0.002
        q0 = self.data.qpos.astype(np.double)[-self.cfg.env.num_actions:]
        timer = NanoSleep(1)  # 创建一个1毫秒的NanoSleep对象
        while tt < T + dt / 2.0:
            start_time = time.perf_counter()
            st = min(tt / T, 1.0)
            s0 = 0.5 * (1.0 + math.cos(math.pi * st))
            s1 = 1 - s0
            for idx in range(self.cfg.env.num_actions):
                self.target_q[idx] = s0 * q0[idx] + s1 * qd[idx]
            self.q = self.data.qpos.astype(np.double)[-self.cfg.env.num_actions:]
            self.dq = self.data.qvel.astype(np.double)[-self.cfg.env.num_actions:]
            self.set_sim_target(self.target_q)
            tt += dt
            timer.waiting(start_time)  # 等待下一个时间步长

    def init_robot(self):
        final_goal = self.cfg.robot_config.stand_pos0_sim
        self.joint_plan(1, final_goal)
        for idx in range(self.cfg.env.num_actions):
            self.target_q[idx] = final_goal[idx]

    def show(self):
        m = self.model
        name = m.names.decode('utf-8').split('\x00')
        print("\033[32m>>The robot: %s with %d dof, DofProperties information as follow:\033[0m" % (name[0], m.njnt))
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        print(
            "\033[33m|     Joint names    | type | idx |   default_pos  | damping | stiffness |   limits_lower   |   limits_upper   | margin |\033[0m")
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        for i in range(0, m.njnt):
            jointName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(
                "| %-19s|  %2d  |  %2d | %5.2f(%7.2f) | %5.1f   | %6.1f    | %7.4f(%7.2f) | %7.4f(%7.2f) |  %5.2f |" % (
                    jointName, m.jnt_type[i], i,
                    m.qpos0[i], np.rad2deg(m.qpos0[i]),
                    m.dof_damping[i],
                    m.jnt_stiffness[i],
                    m.jnt_range[i][0], np.rad2deg(m.jnt_range[i][0]),
                    m.jnt_range[i][1], np.rad2deg(m.jnt_range[i][1]),
                    m.jnt_margin[i]))
        print(
            "\033[32m+--------------------+------+-----+----------------+---------+-----------+------------------+------------------+--------+\033[0m")
        print("\033[32m>>The robot: %s with %d actuators/controls(ctrl) informations:\033[0m" % (name[0], m.nu))
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        print("\033[33m|     Joint names    | actuator | id | limit | c_lower | c_upper |\033[0m")
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        for i in range(0, m.nu):
            joint_id = m.actuator_trnid[i]  # 获取关节 ID
            jointName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id[0])  # 获取关节名称
            actuatorName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print("| %-19s|   %-6s | %2d |   %d   | %7.2f | %7.2f |" % (
                jointName, actuatorName, i,
                m.actuator_ctrllimited[i],
                m.actuator_ctrlrange[i][0],
                m.actuator_ctrlrange[i][1]))
        print("\033[32m+--------------------+----------+----+-------+---------+---------+\033[0m")
        print("\033[32m>>The robot: %s with %d sensors informations:\033[0m" % (name[0], m.nsensor))
        print("\033[32m+--------+----+-----+---------+\033[0m")
        print("\033[33m| sensor | id | dim | address |\033[0m")
        print("\033[32m+--------+----+-----+---------+\033[0m")
        for j in range(0, m.nsensor):
            SensorName = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, j)
            print("| %-6s | %2d |   %d |    %2d   |" % (
                SensorName, j,
                m.sensor_dim[j],
                m.sensor_adr[j]))
        print("\033[32m+--------+----+-----+---------+\033[0m")
        print("\033[32m>> the end of joint and sensor information !\033[0m")

    def start_key(self):
        window = glfw.create_window(400, 400, "key_callback", None, None)
        glfw.set_key_callback(window, self.key_callback)

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_W:
            self.cfg.cmd.vx += 0.05
        elif key == glfw.KEY_S:
            self.cfg.cmd.vx -= 0.05
        elif key == glfw.KEY_A:
            self.cfg.cmd.vy += 0.05
        elif key == glfw.KEY_D:
            self.cfg.cmd.vy -= 0.05
        elif key == glfw.KEY_J:
            self.cfg.cmd.yaw += 0.05
        elif key == glfw.KEY_L:
            self.cfg.cmd.yaw -= 0.05
        elif key == glfw.KEY_SPACE:
            self.cfg.cmd.vx = self.cfg.cmd.vy = self.cfg.cmd.yaw = 0
        elif key == glfw.KEY_1:
            self.cfg.cmd.stand = 1
        elif key == glfw.KEY_2:
            self.cfg.cmd.stand = 2
        elif key == glfw.KEY_3:
            self.cfg.cmd.stand = 3
        self.cfg.cmd.vx = np.clip(self.cfg.cmd.vx, -1.0, 3.0)
        self.cfg.cmd.vy = np.clip(self.cfg.cmd.vy, -1.0, 1.0)
        self.cfg.cmd.yaw = np.clip(self.cfg.cmd.yaw, -1.0, 1.0)

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
        self.est_vel = np.zeros(3, dtype=np.float32)

        action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        self.action[:] = action[:]

        target_qt = self.action * self.cfg.control.action_scale + self.pos0
        target_qt = np.clip(target_qt, self.cfg.robot_config.clip_actions_lower, self.cfg.robot_config.clip_actions_upper)
        return target_qt



if __name__ == '__main__':
    mode_path = Config.robot_config.mode_path
    print("load mode = ", mode_path)
    policy = ort.InferenceSession(mode_path)
    mybot = Sim2Sim(Config, policy)
    mybot.show()
    mybot.init_robot()
    mybot.run()