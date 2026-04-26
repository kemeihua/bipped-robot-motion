
奖励函数reward
奖励设计=“进度+平衡+节能+姿态惩罚” or 平衡/前进/能耗/平滑
采用“教师-学生”蒸馏或循环策略，可把多个动作片段整合进一个网络，实现行为平滑切换
安全策略

### 从位置/速度/能耗等角度配置对应奖励

  # ================================= Rewards reference motion tracking========================== #
'''生存奖励：每个时间步都给予奖励'''
def _reward_survival(self):
'''关节位置跟踪奖励'''
def _reward_joint_pos(self):
'''足部离地高度奖励'''
def _reward_feet_clearance(self):
'''足部离地高度惩罚'''
def _reward_penalty_feet_clearance(self):
'''足部高度惩罚'''
def _reward_penalty_feet_height(self):
def _reward_feet_swing_height(self):
def _reward_booster_feet_swing(self):
def _reward_feet_distance(self):
def _reward_knee_distance(self):
def _reward_penalty_close_feet_xy(self):
def _reward_foot_slip(self):
def _reward_feet_air_time(self):
def _reward_osu_feet_air_time(self):
def _reward_epsh_feet_air_time(self):
def _reward_feet_contact_number(self):
def _reward_contact(self):
def _reward_feet_contact_forces(self):
def _reward_feet_contact_forces_zj(self):
def _reward_feet_stance(self):
def _reward_feet_swing(self):
# ==================================== base pos  =========================================== #
def _reward_lin_vel_z(self):
def _reward_epsh_orientation(self):
def _reward_epsh_lin_vel_z(self):
def _reward_epsh_ang_vel_xy(self):
def _reward_epsh_ang_vel_z(self):
def _reward_osu_default_joint_pos(self):
def _reward_osu_hip_pos(self):
def _reward_hip_pos(self):
def _reward_default_joint_pos(self):
def _reward_orientation(self):
def _reward_base_height(self):
def _reward_penalty_base_height(self):
def _reward_base_acc(self):
def _reward_yaw_roll_pos(self):
def _reward_ankle_pos(self):
def _reward_joint_kinematics(self):
def _reward_penalty_feet_ori(self): 
def _reward_stand_still(self):
def _reward_stand_clearance(self):
def _reward_stand_still_force(self):
def _reward_joint_regularization(self):
def  _reward_feet_regularization(self):
def _reward_foot_mirror_up(self):
# ==================================== vel tracking  =========================================== #
def _reward_vel_mismatch_exp(self):
def _reward_track_vel_hard(self):
def _reward_tracking_lin_vel(self):
def _reward_tracking_ang_vel(self):
def _reward_low_speed(self):
# ==================================== energy  =========================================== #
'''扭矩使用惩罚（能量效率）'''
def _reward_torques(self):
def _reward_dof_vel(self):
def _reward_dof_acc(self):
'''动作平滑性奖励'''
def _reward_action_smoothness(self):
def _reward_joint_power(self):
def _reward_power_distribution(self):
def _reward_collision(self):
def _reward_epsh_energy(self):
def _reward_epsh_action_rate(self):
def _reward_epsh_feet_force_symm(self):
# ==================================== energy  =========================================== #
'''关节位置限制惩罚'''
def _reward_dof_pos_limits(self):
def _reward_torque_limits(self):
def _reward_dof_vel_limits(self):
def _reward_termination(self):
def _reward_stumble(self):  
def _reward_contact_no_vel(self):



######奖励函数修改：
提高机器人脚部高度
def _reward_feet_clearance(self)
self.cfg.rewards.feet_height 

修改机器人速度
同步修改奖励def _reward_low_speed(self):提高速度奖励无效果（表现为原地踏步）
 
 
######contact 机器人触地奖励

函数名	奖励依据	奖励方式	特点
_reward_feet_contact_number	接触状态与步态一致性	一致：+1，不一致：-0.3	强调步态协调性
_reward_contact	前两个足部接触状态与步态一致性	一致：+1，不一致：0	只关注前腿，简单有效
_reward_feet_contact_forces	接触力大小	超过阈值部分惩罚	防止重踏
_reward_feet_contact_forces_zj	接触力变化	变化越小奖励越高（指数形式）	鼓励平滑接触


######不同周期步态对应奖励：
不同步态周期如：stance phase/swing phase 对应奖励策略 
self._get_gait_phase()

  
  
  
 强化学习算法优化： 
  -网络结构（暂不优化）
  max_iterations = 3000
  actor_hidden_dims = [64, 64, 64]
  critic_hidden_dims = [64, 64, 64]
  -超参数（统一化）
  -奖励

  
ppo 

  状态空间 def get_observations(self) def get_privileged_observations(self)
  动作空间def act(self, obs, critic_obs)
  
强化学习跟随机器人参考状态实现如控制（电机扭矩）
该方法计算机器人参考状态：相位phase和自由度位置dof_pos
 def compute_ref_state(self):
 self.ref_dof_pos = torch.zeros_like(self.dof_pos)
 
观测空间/状态空间
构建观察缓冲区：
        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            q,  #self.dof_pos 
            dq, #self.dof_vel
            self.actions,  # 10 ; 6 for virtual leg
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2  机器人的前两个欧拉角（例如滚转和俯仰）与某个缩放因子相乘实现了对观测值的缩放

            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            # self.rand_push_force[:, :2],  # 2
            # self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 12.,  # 1   /30
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)
        通过连接多个特征（如命令输入、关节位置、速度、动作、基本角速度等）来创建所谓的特权观察缓冲区（privileged_obs_buf），该缓冲区包含了更多的状态信息。

        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,  # 10D
            dq,  # 10D
            self.actions,  # 10D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz[:, :2] * self.obs_scales.quat,  # 2
        ), dim=-1)
        
        该方法用于从给定的动作计算出关节扭矩，是控制代理运动的核心部分。 # pd controller
	def _compute_torques(self, actions):
	

观测空间：关节位置/速度/扭矩/角速度
动作空间：网络模型输出动作actions和值values


                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    
该方法的主要作用是在给定当前观察的情况下，通过 actor_critic 网络计算出代理应当采取的动作以及与之相关的各种信息，如价值、策略的对数概率以及动作的分布参数等。         
        def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
        
        
        
        



                     
 
 
 


