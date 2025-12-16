  
  
  强化学习要素：
  环境/智能体/状态/奖励/动作
  
    # -------------------- 初始状态 --------------------
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.88]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
           'L_hip_yaw': 0.,
           'L_hip_roll': 0.00,
           'L_hip_pitch': 0.3,
           'L_knee_pitch': -0.6,
           'L_ankle_pitch': 0.3,
           'R_hip_yaw': 0.,
           'R_hip_roll': 0.00,
           'R_hip_pitch': 0.3,
           'R_knee_pitch': -0.6,
           'R_ankle_pitch': 0.3,
        }
        
        机器人的坐标体系RPY：roll轴、pitch轴和yaw轴
     roll：翻滚角ϕ \phiϕ（视角旋转）。向右滚为正，反之为负。
     pitch：俯仰角θ \thetaθ（往上往下）。抬头为正，反之为负。
      yaw：侧航角ψ \psiψ（往左往右）。右偏航为正，反之为负。
———————————————
        10个自由度，左右各5自由度
        
        
               class ranges:
            lin_vel_x = [-1.0, 1.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]


    get_observations()

机器人输入输出（状态/动作）：
车辆输入输出：传感器数据-车辆油门刹车转向脚

机器人运动学和动力学

修改环境参数提高机器人行走速度：
 static_friction = 0.2    #0.6
 dynamic_friction = 0.2   #0.6
 
         damping = {  'hip_yaw':      1.,   #2
                     'hip_roll':     1.,   #5
                     'hip_pitch':    1.,   #5
                     'knee_pitch':   1.,   #5
                     'ankle_pitch':  1.,   #2
                     
修改机器人初始状态：
                     


  
X2工程
  
    # -------------------- 初始状态 --------------------
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.88]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
           'L_hip_yaw': 0.,
           'L_hip_roll': 0.00,
           'L_hip_pitch': 0.3,
           'L_knee_pitch': -0.6,
           'L_ankle_pitch': 0.3,
           'R_hip_yaw': 0.,
           'R_hip_roll': 0.00,
           'R_hip_pitch': 0.3,
           'R_knee_pitch': -0.6,
           'R_ankle_pitch': 0.3,
        }
        10个自由度，左右各5自由度
        机器人的坐标体系RPY：roll轴、pitch轴和yaw轴
     roll：翻滚角ϕ \phiϕ（视角旋转）。向右滚为正，反之为负。
     pitch：俯仰角θ \thetaθ（往上往下）。抬头为正，反之为负。
      yaw：侧航角ψ \psiψ（往左往右）。右偏航为正，反之为负。
      
     欧拉角，四元素(坐标转换)
     欧拉角是用三个角度来表示旋转的，通常标记为（滚转、俯仰、偏航），也被称为（roll, pitch, yaw）。每个角度对应于相对于一个坐标轴的旋转。
     四元数是一种更为优雅且数学上稳健的表示旋转的方法，通常写为 q=w+xi+yj+zkq=w+xi+yj+zk，其中 ww 是标量部分，x,y,zx,y,z 是虚数部分。
     
     姿态
     线速度：XYZ
     角速度：rpy
     
———————————————

    机器人运动学和动力学           

DOF POsitions
描述机器人状态   机器人独立可动的关节数量
dof_vel
dof_acc


机器人输入输出（状态/动作）：
状态：位置pos,关节角度，速度?-
动作：转矩，电机-


车辆输入输出：
传感器数据，定位/地图/导航/感知预测障碍物- 
车辆速度（油门刹车转向脚）-

【模型简化与步态规划】
采用人形机器人全身动力学做规划十分复杂，且存在动力学参数辨识等问题，因此上层通常会将高维全身动力学模型简化为低维模型进行步态规划，下层考虑更复杂的模型进行跟踪，最终产生符合机器人动力学特性的运动。常见的简化模型有倒立摆（ASIMO、Walker、HRP）、弹簧倒立摆模型（Cassie、Digit）、单刚体（四足、青龙）等等。


工程常用API
mujoco
Gym
Pytorch

其他第三方库
numpy
cv2
tqdm(进度条库)
glfw(开源的多平台库)


###X2调参优化###
环境修改维度：
-地形复杂性，平坦/坡度/崎岖/障碍物
-地面材质，水泥，草地，湿滑
-外部干扰，风力/光照强弱
-能量消耗，高动态行走（跨步，爬坡），动静步态切换
-传感器精度
-算法鲁棒性
a.地形修改：

 static_friction = 0.2    #0.6
 dynamic_friction = 0.2   #0.6

b.控制修改：（刚度？）
         damping = {  'hip_yaw':      1.,   #2
                     'hip_roll':     1.,   #5
                     'hip_pitch':    1.,   #5
                     'knee_pitch':   1.,   #5
                     'ankle_pitch':  1.,   #2
                     }  # [N*m/rad]  # [N*m*s/rad]
                     

智能体算法修改：
https://zhuanlan.zhihu.com/p/1911068257554793562  PPO算法的调参步骤和技巧

 -超参数调优：如 
 的选择，直接影响策略更新的稳定性和收敛速度。
 ###1.修改学习率
learning_rate = 1e-3  1e-1
result:机器人行走速度缓慢
- 网络结构选择：对于复杂任务，可能需要更深或更复杂的网络架构来捕捉状态与动作间的非线性关系。（X2ppo网络结构未修改，代码）
       ### actor_hidden_dims = [512, 256, 128]
       ##critic_hidden_dims = [512, 256, 128]
   max_iterations = 5000 #3000     

-奖励函数修改：
提高机器人脚部高度
def _reward_feet_clearance(self)
self.cfg.rewards.feet_height 
###修改速度匹配
 def _reward_vel_mismatch_exp(self):
 def _reward_track_vel_hard(self):
 ###平滑性
 允许机器人加速
def _reward_base_acc(self):
rew = torch.exp(torch.norm(root_acc, dim=1) *10）
现象：修改加速奖励后机器人前进速变缓（或电脑卡顿）
修改机器人摆动


论文
feet rgulation reward
足部归一化奖励
def _reward_feet_regularization(self):
参考论文添加速度

feet distance penalty function
两脚距离
def _reward_knee_distance(self):
工程
        min_dist = 0.2 # 初始姿态下两脚间距离
        max_dist = 0.4
 修改距离奖励
 现象：仿真显示正常行进



修改状态：
加大速度（之前速度调参值在2左右），需要相应修改策略
修改机器人速度command
        class ranges:
            lin_vel_x = [-1.0, 1.5]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

            # lin_vel_x = [-1.0, 5.0]  # min max [m/s]
            # lin_vel_y = [-0.5, 3.5]   # min max [m/s]
            # ang_vel_yaw = [-1.0, 2.0]    # min max [rad/s]
            # heading = [-3.14, 3.14]
 速度超3之后机器人无法跑动（表现为原地踏步）
 同步修改奖励def _reward_low_speed(self):提高速度奖励无效果（表现为原地踏步）
 reward[speed_too_high] = 2.0   #0.
 提高迭代次数5000,速度仍缓慢
 max_iterations = 5000
 
 
 11.21Log:
 1.  ###平滑性l
 允许机器人加速
def _reward_base_acc(self):
rew = torch.exp(torch.norm(root_acc, dim=1) *3）
将机器人加速奖励调回原值，运动速度恢复正常。
2. 机器人抬脚高度恢复原来值，当前版本机器人速度变缓；第二版本继续抬高脚，目前未训练完成（电脑卡机）。
版本一（正常离地高度）：机器人正常抬脚高度1000步和5000步差异不大，正常运行。//gym可视化3000步，正常行进，5000步缓慢前进;mujoco可视化3000步，正常行进，5000步低速缓慢前进。
版本二（抬高脚）：机器人抬高角度训练1000步正常前进，5000步原地？//gym可视化3000步，缓慢行进，5000步原地踏步;mujoco可视化3000步，向后倒退；5000步原地踏步。
-todo 实物调试
11.25Log:
版本一（正常离地高度）：3000步，机器前行速度缓慢，可正常往后退；5000步，几乎无法前进。
部署代码仿真修改路径后可正常运行，实物存在通信问题。
关于路径问题，代码文件结构影响文件导入和加载，可尝试python-m 模块方式代替直接运行脚本。
远程部署和在线部署
部署启动操作：
1.执行启动脚本
2.启动遥控
3.机器初始状态摆正
4.遥控切换策略
5.遥控运行机器
6.自主执行策略？（遥操可以适应运控速度变化，观测接口）



###实物部署代码###
x2_deploy_run
机器人远程root@192.168.55.202

工程X02lite
在scripts/policies新建策略文件kmh添加对应策略.onnx
修改sim2real_mult.py run_path1模型路经

cat /etc/init.d/rl.sh
systemctl restart rl
systemctl stop rl
执行脚本
 . /etc/init.d/rl.sh
 
/home/x02lite/x2_deploy_release_lzy/scripts#  . /etc/init.d/rl.sh

zero-shot transfer from simulation to the real-world environment
仿真和现实差距问题
-动力学模型优化
-添加噪声干扰
-执行器/控制算法优化（视觉/深度相机/激光雷达，无毫米波雷达，传感输入端到端网络）
-增加算法鲁棒性
-增加传感器提高感知能力
-减小仿真和现实训练差异

实际差异：
添加噪声不能解决实际干扰
机器人本体结构影响算法开发
机器人系统设计
模型和控制器统一化

contact dynamic
Depth Sensors


                     
                     