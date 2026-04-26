import os
import sys
# 获取当前脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的绝对路径
project_root = os.path.dirname(script_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

import asyncio
import websockets
import json

class WebSocketClient:
    def __init__(self, server_host="127.0.0.1", server_port=8765):
        self.server_uri = f"ws://{server_host}:{server_port}"
        self.connection = None
        self.loco_state = "loco_stop"
        self.arm_state = "init"
        self.robot_cmd = {
            "vel_x": 0.,
            "vel_y": 0.,
            "vel_yaw": 0.,
            "base_roll": 0.,
            "base_pitch": 0.,
            "base_yaw": 0.,
            "base_z": 0.,
            "loco_mode": "loco_stop", # loco_stop, loco_walk, loco_walk_and_stand
            "arm_mode": "init"}  # init, hello, shake
        self.robot_status={
            "bus_voltage": 0.,
            "bus_current": 0.,
            "bus_energy": 0.,
            "vel_x": 0.,
            "vel_y": 0.,
            "vel_z": 0.,
            "joint_q": [0.]*10,
            "joint_dq": [0.]*10,
            "loco_mode": "none",# loco_stop, loco_walk, loco_walk_and_stand, loco_fall
            "arm_mode": "none", # init, hello, shake
            "arm_action": "none", # doing, done
        }

    async def connect(self):
        """连接到 WebSocket 服务器"""
        try:
            self.connection = await websockets.connect(self.server_uri)
            print(f"已连接到服务端: {self.server_uri}")
            # 发送初始握手消息
            await self.connection.send("droidpad")
            response = await self.connection.recv()
            if response == "droidup":
                print("握手成功！")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    async def send_control_data(self):
        """发送控制器数据并接收响应"""
        if not self.connection:
            print("未连接到服务端！")
            return

        try:
            self.loco_state = "loco_stop"
            self.arm_state = "init"
            # json消息
            self.robot_cmd["vel_x"] = 0.5
            self.robot_cmd["vel_y"] = 0.
            self.robot_cmd["vel_yaw"] = 0.
            self.robot_cmd["loco_mode"] = self.loco_state
            self.robot_cmd["arm_mode"] = self.arm_state
            # 发送 JSON 数据
            await self.connection.send(json.dumps(self.robot_cmd))
            # print(f"已发送数据: {control_data}")

            # 接收服务端响应
            response = await self.connection.recv()
            data = json.loads(response)
            self.robot_status = data
            print(f"收到响应: {self.robot_status}")

        except websockets.ConnectionClosed:
            print("连接已关闭！")
        except Exception as e:
            print(f"发送数据时出错: {e}")

    async def close(self):
        """关闭连接"""
        if self.connection:
            await self.connection.close()
            print("连接已关闭")

async def send_continuous_data(client, interval=0.01):
    """每隔 interval 秒发送一次数据"""
    while True:
        await client.send_control_data()
        await asyncio.sleep(interval)

async def main():
    client = WebSocketClient("127.0.0.1", 8765)
    if await client.connect():
        try:
            await send_continuous_data(client)
        except KeyboardInterrupt:
            await client.close()

if __name__ == "__main__":
    asyncio.run(main())