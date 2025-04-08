# 固定坐标没问题了
# 还差：
# 1.加上末端执行器，加上基座
# 2.40个箱子
# 3.加上碰撞检测
# 4.一个逆运动学找不到，就用另一个拟合
# bug,只需要一步的，在test里会卡死
import argparse
import math
import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import List
from lib.ros_kinematics import RobotKinematics
from scipy.optimize import minimize

class ArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super(ArmEnv, self).__init__()
        
        # 初始化MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path("./config/aubo_with_box.xml")
        self.data = mujoco.MjData(self.model)
        # 渲染相关
        self.render_mode = render_mode
        self.viewer = None
        
        # 动作空间：x,y,z三个方向的位移
        self.action_space = spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,),  # [current_pos(3), target_pos(3), rel_pos(3), velocity(3)]
            dtype=np.float32
        )
        
        # 环境参数
        self.max_steps = 1000
        self.current_step = 0
        self.target_pos = np.zeros(3)
        self.pos_threshold = 0.05
        
        # 箱子位置信息
        self.boxes = []
        self._init_boxes()
        
        # 添加运动学求解器
        self.kinematics = RobotKinematics()
        # 当前关节角度
        self.current_joints = np.zeros(6)
        
        # 移除不需要的PD控制参数
        # self.kp = 800   # 位置增益
        # self.kv = 50    # 速度增益
        
        self.target_site_id = self.model.site("target_site").id

    def _init_boxes(self):
        """初始化所有箱子的位置信息"""
        for i in range(8):  # 8排
            for j in range(5):  # 5列
                x = 1.018 - j * 0.509  # 从右到左

                if i < 3:  # 前3排
                    y = 0.9  # 固定y坐标
                    z = 2.440 - i * 0.327  # 从上到下
                else:  # 后5排
                    # y = 1.618
                    y = 0.9
                    # z = 1.461 - (i-3) * 0.327 + 0.6 #0.6是连杆的偏置
                    z = 1.461 - (i-3) * 0.327 
                self.boxes.append([x, y, z])
    # def _init_boxes(self):
    #     """初始化前三层箱子的位置信息"""
    #     for i in range(3):  # 仅前三排
    #         for j in range(5):  # 5列
    #             x = 1.018 - j * 0.509  # 从右到左
    #             y = 0.8  # 固定y坐标
    #             z = 2.440 - i * 0.327  # 从上到下
    #             self.boxes.append([x, y, z])

    def _get_endeffector_pos(self) -> np.ndarray:
        """获取末端执行器位置"""
        return self.data.body("wrist3_Link").xpos.copy()

    def _get_endeffector_vel(self) -> np.ndarray:
        """获取末端执行器速度"""
        return self.data.body("wrist3_Link").cvel[:3].copy()

    def _set_target_position(self):
        """随机设置目标位置"""
        box_pos = self.boxes[np.random.randint(len(self.boxes))]
        # box_pos = np.array( [1.018, 0.8, 2.244])
        self.target_pos = box_pos + np.array([0, 0, 0])  # Y坐标减1米
        print(self.target_pos)
        mujoco.mj_resetData(self.model, self.data)  # 重置时保持目标位置可见

    def reset(self, seed=None, options=None):
        """环境重置"""
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        
        # 设置初始关节位置
        self.data.qpos = np.zeros(self.model.nq)
        mujoco.mj_forward(self.model, self.data)
        
        # 设置新目标
        self._set_target_position()
        
        # 更新目标点标记的位置
        self.data.site_xpos[self.target_site_id] = self.target_pos

        # 前向仿真一步使状态更新
        mujoco.mj_step(self.model, self.data)
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """获取观测值"""
        current_pos = self._get_endeffector_pos()
        target_pos = self.target_pos
        rel_pos = target_pos - current_pos
        velocity = self._get_endeffector_vel()
        
        return np.concatenate([current_pos, target_pos, rel_pos, velocity])

    def step(self, action):
        # 应用动作（末端位移）
        current_pos = self._get_endeffector_pos()
        # print(f"---进行一步前的current_pos: {current_pos}")
        target_delta = action * 0.05 # 缩放动作幅度
        target_pos = current_pos + target_delta
        # print(f"原始的target_pos: {target_pos}")

        # print(f"修改后target_pos: {target_pos}")
        # target_pos = np.array([1.018, 1, 2.44])

        # 使用PD控制移动到目标位置
        success = self._apply_ros_control(target_pos.copy())
        # print(f"target_pos: {target_pos}")
        # if success == False:
            # self._apply_pd_control(target_pos.copy())
        # 进行仿真
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # 获取观测
        obs = self._get_obs()

        current_pos = self._get_endeffector_pos()  # 这一步不能删掉
        # print(f"进行一步后的current_pos: {current_pos}")
        # 计算奖励
        distance = np.linalg.norm(obs[6:9])
        reward = -distance  # 主奖励：负距离
        if distance < self.pos_threshold:  # 成功奖励
            reward += 10
        
        # 终止条件
        done = distance < self.pos_threshold or self.current_step >= self.max_steps
        if done:
            print(f"完成！在{self.current_step}步时，距离为{distance}，奖励为{reward}")
        # 修改终止条件判断
        terminated = distance < self.pos_threshold
        truncated = self.current_step >= self.max_steps  # 新增截断判断
        # 确保目标点标记位置保持更新
        self.data.site_xpos[self.target_site_id] = self.target_pos
        # 渲染（如果需要）
        if self.render_mode == "human":
            self.render()
        # 返回5个值（原为4个）
        return obs, reward, terminated, truncated, {}  # 添加truncated参数

    def _apply_ros_control(self, target_pos):
        # print("使用ROS控制器")
        # 计算当前姿态（这里简化为固定姿态）
        roll, pitch, yaw = math.pi/2, 0, 0
        # if target_pos[2] > 1.5:
        #     roll, pitch, yaw = math.pi/2, 0, 0
        # else:
        #     roll, pitch, yaw = math.pi, 0, math.pi
        
        target_pos[0] = -target_pos[0]
        target_pos[1] = -target_pos[1]
        target_pos[2] = target_pos[2] - 0.99
        """使用逆运动学计算关节角度"""


        # print(f"目标位置: {target_pos}")
        # print(f"目标姿态: {roll}, {pitch}, {yaw}")
        # 计算逆运动学
        solutions = self.kinematics.inverse_kinematics(
            target_pos[0], target_pos[1], target_pos[2],
            roll, pitch, yaw
        )
        
        # 选择最佳解
        best_joints = None
        for i in range(8):
            if solutions['state'][i] == 1:
                joint_angles = solutions['joint'][i]
                if not any(np.isnan(angle) for angle in joint_angles):
                    best_joints = joint_angles
                    break
        
        # 检查是否找到有效解
        if best_joints is not None:
            # 添加偏移量
            best_joints = best_joints + np.array([math.pi, math.pi/2, 0, math.pi/2, 0, 0])
            # 更新关节角度
            self.current_joints = best_joints
            # 设置关节位置
            self.data.qpos[:6] = best_joints
            mujoco.mj_forward(self.model, self.data)
            return True
        else:
            # print("警告：未找到有效的逆运动学解")
            # 保持当前姿态不变
            self.data.qpos[:6] = self.current_joints
            return False

    def _apply_pd_control(self, target_pos):
        """使用优化器计算并移动到目标位置"""
        # print("使用PD控制器")
        # 设置固定的目标姿态（与原代码保持一致）
        roll, pitch, yaw = -math.pi/2, 0, 0
        target_quat = self.euler_to_quat(roll, pitch, yaw)
        
        # 保存初始关节角度
        initial_q = self.data.qpos[:6].copy()
        
        # 使用优化求解器找到目标关节角度
        bounds = [(-np.pi, np.pi) for _ in range(6)]
        result = minimize(
            self.pose_cost_function,
            initial_q,
            args=(target_pos, target_quat),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        if result.success:
            # 直接设置到目标关节角度（简化版，不做平滑过渡）
            self.data.qpos[:6] = result.x
            mujoco.mj_forward(self.model, self.data)
        else:
            print("警告：优化求解失败")

    def pose_cost_function(self, q, target_pos, target_quat):
        """计算位置和姿态误差的代价函数"""
        # 设置关节角度
        for i, angle in enumerate(q):
            self.data.qpos[i] = angle
        
        # 更新物理引擎
        mujoco.mj_forward(self.model, self.data)
        
        # 获取当前末端执行器位置和姿态
        curr_pos = self._get_endeffector_pos()
        curr_rot = self.data.xquat[self.model.body("wrist3_Link").id].copy()
        
        # 计算位置误差
        pos_error = np.sum((curr_pos - target_pos)**2)
        
        # 计算四元数误差
        quat_error = np.sum((curr_rot - target_quat)**2)
        
        # 总代价
        cost = pos_error + quat_error
        return cost

    def euler_to_quat(self, roll, pitch, yaw):
        """将欧拉角转换为四元数"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])

    def render(self):
        """渲染环境"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        if self.viewer.is_running():
            self.viewer.sync()
        
        return None
    
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--model', type=str, default='models/train5_model', help='模型路径')
    args = parser.parse_args()

    # 创建环境
    # env = ArmEnv(render_mode="human")
    env = ArmEnv(render_mode=None)
    vec_env = DummyVecEnv([lambda: env])

    if args.train:
        # 训练配置
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='./logs/',
            name_prefix='sac_model'
        )
        
        # SAC参数配置
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=5e-4,
            buffer_size=200000,
            batch_size=512,
            gamma=0.95,
            tau=0.005,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=dict(pi=[256,256], qf=[256,256])
            )
        )
        try:
            # 开始训练
            model.learn(
                total_timesteps=200000,
                callback=checkpoint_callback,
                tb_log_name="sac_arm"
            )
            model.save(args.model)
            print("模型保存到:", args.model)
        except KeyboardInterrupt:
            print("\n训练被手动中断")
        finally:
            # 保存最终模型
            model.save("models/train5_model_cut")
            print("被中断，模型保存到:", "train5_model_cut")
            # 关闭环境
            env.close()

    elif args.test:
        # 加载测试环境
        test_env = ArmEnv(render_mode="human")
        
        # 加载模型
        model = SAC.load(args.model)
        
        # 测试10个回合
        for episode in range(10):
            obs, _ = test_env.reset()
            total_reward = 0
            terminated = truncated = False
            
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = test_env.step(action)
                total_reward += reward
                test_env.render()
                time.sleep(0.02)
            
            print(f"回合 {episode+1}: 总奖励 = {total_reward:.2f}")

if __name__ == "__main__":
    main()