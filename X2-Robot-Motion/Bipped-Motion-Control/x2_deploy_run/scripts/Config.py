import numpy as np


class Config:
    class env:
        # change the observation dim
        frame_stack = 15
        num_single_obs = 40
        num_observations = int(frame_stack * num_single_obs)
        num_actions = 10
        run_duration = 100000.0  # 单位s
        num_arm_actions = 10
        num_leg_actions = 10
        # grpc_channel = '192.168.254.100'  # r6s
        grpc_channel = '192.168.55.201'

    class cmd:
        vx = 0.0
        vy = 0.0
        yaw = 0.0
        stand = 1

    class sim_config:
        mujoco_model_path = f'../robots/x2_real_10dof_new/scene.xml'

    class control:
        action_scale = 0.25
        decimation = 10  # 100hz(10ms)
        cycle_time = 0.64 # sec
        timestep = 0.001# sec   sim step 1KHz(1ms)
        timestep_ms = 1     # ms

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 0.25
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
        clip_observations = 18.
        clip_actions = 18.

    class noise_scales:
        add_noise = False
        dof_pos = 0.001
        dof_vel = 0.500
        euler   = 0.100
        ang_vel = 0.200

    class filter_params:
        add_filter = True
        delay_action = 1.0
        delay_qt     = 1.0
        delay_q      = 1.0
        delay_euler  = 1.0
        delay_dq     = 1.0
        delay_gyro   = 1.0

    class save_data:
        sim_path = '../data/simdata/'
        real_path = '../data/realdata/'

    class robot_config:
        # ================== Robot Static Paras ==================

        clip_actions_upper = np.array([ 30.,  30, 100,    0,  60,  30.,  30, 100,    0,  60], dtype=np.double) * np.pi / 180.*3
        clip_actions_lower = np.array([-30., -30, -30, -100, -60, -30., -30, -30, -100, -60], dtype=np.double) * np.pi / 180.*3
        kpFall = np.array([ 0, 0, 0, 0, 0,  0, 0, 0, 0, 0], dtype=np.double)
        kdFall = np.array([ 1, 5, 5, 5, 5,  1, 5, 5, 5, 5], dtype=np.double)
        target_joint_pos_scale = 0.17  # rad
        # tau_limit = 200. * np.ones(10, dtype=np.double)
        tau_limit = np.array([30., 45., 100., 100., 100., 30., 45., 100., 100., 100.], dtype=np.double) * 1.2
        # tau_limit = np.array([30., 45., 40., 70., 70., 30., 45., 40., 70., 70.], dtype=np.double) * 1.2
        # tau_limit = np.array([30., 45., 70., 70., 70., 30., 45., 70., 70., 70.], dtype=np.double) * 1.4
        # tau_limit = np.array([40., 40., 70., 100., 70., 40., 40., 70., 100., 70.], dtype=np.double)
        # tau_limit = np.array([30., 45., 70., 100., 30., 30., 45., 70., 100., 30.], dtype=np.double)
        # tau_limit = np.array([30., 70., 70., 100., 30., 30., 70., 70., 100., 30.], dtype=np.double) * 1.2
        # ================== traj test =======================
        q0      = np.array([0,    0,    0,    0,    0,       0,    0,    0,    0,    0],dtype=np.double) * np.pi / 180.
        q_scale = np.array([0,    0,    0,    0,    0,       0,    0,    0,    0,    0],dtype=np.double) * np.pi / 180.
        # ================== current policy pos0 and pd ===============
        stand_pos0 = np.array([0, 0.02, 0.30,  -0.6, 0.30,   0, 0.02, 0.30,  -0.6, 0.30], dtype=np.double)
        pos0       = np.array([0, 0.00, 0.30,  -0.6, 0.30,   0, 0.00, 0.30,  -0.6, 0.30], dtype=np.double)
        pos0_bias  = np.array([0, 0.00, 0.00,  -0.0, 0.00,   0, 0.00, 0.00,  -0.0, 0.00], dtype=np.double)
        kpStand = np.array([ 50, 100, 100, 100,  50,  50, 100, 100, 100,  50], dtype=np.double)
        kdStand = np.array([  1,   2,   2,   2,   2,   1,   2,   2,   2,   2], dtype=np.double)
        kps = np.array([ 50, 200, 200, 200, 30,   50, 200, 200, 200, 30], dtype=np.double)
        kds = np.array([  1,   5,   5,   5,  2,    1,   5,   5,   5,  2], dtype=np.double)

        stand_pos0_sim = np.array([0, 0.0, 0.3, -0.6, 0.3, 0, 0.0, 0.3, -0.6, 0.3], dtype=np.double)
        pos0_sim = np.array([0, 0.00, 0.3, -0.6, 0.3, 0, 0.00, 0.3, -0.6, 0.3], dtype=np.double)
        kpSim = np.array([100,  200, 200, 200, 30,    100, 200, 200, 200, 30], dtype=np.double)
        kdSim = np.array([  2,    5,   5,   5,  2,      2,   5,   5,   5,  2], dtype=np.double)
        euler0 = np.array([ 0.0, 0.0], dtype=np.double) * np.pi / 180.
        cmd_bias = np.array([ 0.00, 0.0, 0.0], dtype=np.double)
        # ======================  est stand =============================
        # mode_path = 'policies/stand/2025-04-27_09-08-55_mlp_est_10000.onnx' # x2-08可以稳定行走站立，0.64
        # ======================  run  =============================
        # mode_path = 'policies/run/2025-04-02_16-56-02_x2_real_2000.onnx' # ***** 北京马拉松 cmd=2m/s，1.8m/s，0.64
        # mode_path = 'policies/run/2025-08-08_19-42-22_mlp_2000.onnx'  # **** 马拉松策略优化转向，触地变重，cmd=3m/s，2.2m/s，0.64
        mode_path = 'policies/run/2025-08-09_16-47-15_mlp_2000.onnx'  # ***** 侧向稳定 cmd=3m/s，2.2m/s，0.64，resume 2025-08-08_19-42-22_mlp，2000轮:B09、06，1000轮:B08
        # mode_path = 'policies/run/2025-08-12_12-19-29_mlp_4000.onnx'  # **** 马拉松策略优化转向 cmd=2m/s，2m/s，0.64

        # log_dir = '/home/liangzhiyuan/RL/X02/x2-gym-run/'
        # mode_path = log_dir + "logs/X2_Real/2025-07-31_22-44-45_mlp/policies/2025-07-31_22-44-45_mlp_2000.onnx" # 马拉松策略复现,*****

        # mode_path = log_dir + 'logs/x2/2025-08-06_01-24-47_mlp/policies/2025-08-06_01-24-47_mlp_2000.onnx' # 马拉松策略复现,*****
        # mode_path = log_dir + 'logs/x2/2025-08-06_02-39-53_mlp/policies/2025-08-06_02-39-53_mlp_1000.onnx'  # resume 2025-08-06_01-48-48_mlp, 3.0m/s ：1000轮*****
        # mode_path = log_dir + 'logs/x2/2025-08-06_09-25-42_mlp/policies/2025-08-06_09-25-42_mlp_2000.onnx'  # resume 2025-08-06_01-48-48_mlp, 3.0m/s  :2000轮，3.0->2.5 *****
        #==============================================噪声错误==============================================
        # mode_path = log_dir + 'logs/x2/2025-08-07_16-36-34_mlp/policies/2025-08-07_16-36-34_mlp_1500.onnx' # *****
        # mode_path = log_dir + 'logs/x2/2025-08-07_16-53-58_mlp/policies/2025-08-07_16-53-58_mlp_1500.onnx'  # ******** 0.64\resume 2025-08-07_15-29-09_mlp, 3.0m/s  B09急速转弯
        # mode_path = log_dir + 'logs/x2/2025-08-07_20-36-32_mlp/policies/2025-08-07_20-36-32_mlp_1000.onnx'  # **** 0.64\resume 2025-08-07_15-29-09_mlp, 3.0m/s 实物突破 2.4m/s，侧向差，注意部署扭矩
        # mode_path = log_dir + 'logs/x2/2025-08-08_19-42-22_mlp/policies/2025-08-08_19-42-22_mlp_2000.onnx'  # ****** 0.64 马拉松，转向优化 3m/s B09 B06
        # mode_path = log_dir + 'logs/x2/2025-08-08_23-13-06_mlp/policies/2025-08-08_23-13-06_mlp_500.onnx'  # ***** 0.58\resume 2025-08-08_19-42-22_mlp stand_radio=0.40
        # mode_path = log_dir + 'logs/x2/2025-08-09_16-47-15_mlp/policies/2025-08-09_16-47-15_mlp_2000.onnx'  # ******** 0.64\resume 2025-08-08_19-42-22_mlp stand_radio=0.40 侧向稳定  2000:B09;1000:B08
        # mode_path = log_dir + 'logs/x2/2025-08-10_19-19-40_mlp/policies/2025-08-10_19-19-40_mlp_1000.onnx'  # **** 0.64\resume 2025-08-08_19-42-22_mlp stand_radio=0.50 跑动时侧向晃 侧向不稳，3.0会摔
        # mode_path = log_dir + 'logs/x2/2025-08-10_19-43-35_mlp/policies/2025-08-10_19-43-35_mlp_2000.onnx'  # **** 0.64\resume 2025-08-08_19-42-22_mlp stand_radio=0.50 跑动时侧向晃 侧向不稳
        # mode_path = log_dir + 'logs/x2/2025-08-11_22-26-22_mlp/policies/2025-08-11_22-26-22_mlp_2000.onnx'  # 0.64\resume 2025-08-08_19-42-22_mlp stand_radio=0.40 2000:B08 *****
        # mode_path = log_dir + 'logs/x2/2025-08-11_23-30-03_mlp/policies/2025-08-11_23-30-03_mlp_1000.onnx'  # 0.64\resume 2025-08-08_19-42-22_mlp stand_radio=0.50 1000:B08*****
        # mode_path = log_dir + 'logs/x2/2025-08-12_12-19-29_mlp/policies/2025-08-12_12-19-29_mlp_4000.onnx'  # 0.64 stand=0.55 马拉松稳定 ******
        # mode_path = log_dir + 'logs/x2/2025-08-12_15-23-38_mlp/policies/2025-08-12_15-23-38_mlp_2000.onnx'  # 0.0.64\resume 2025-08-12_12-19-29_mlp,stand_radio=0.50 仿真3.0,实物侧向不稳定

