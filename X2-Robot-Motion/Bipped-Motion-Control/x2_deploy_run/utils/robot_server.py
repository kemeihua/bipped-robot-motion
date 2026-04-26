import json
import asyncio
import threading
import websockets

class WebSocketDataServer:
    def __init__(self):
        self.thread = None
        self.port = 8765
        self.host = "127.0.0.1"
        self.DataServer_RunFlag = None
        self.server = None  # 用于存储 WebSocket 服务器实例
        self.robot_cmd = {
            "vel_x": 0.,
            "vel_y": 0.,
            "vel_yaw": 0.,
            "base_roll": 0.,
            "base_pitch": 0.,
            "base_yaw": 0.,
            "base_z": 0.,
            "loco_mode": "loco_walk_and_stand", # loco_stop, loco_walk, loco_walk_and_stand
            "arm_mode": "init"}
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

    def set_robot_status(self, robot_status):
        self.robot_status = robot_status

    def start_server(self, host, port):
        self.host = host
        self.port = port
        self.DataServer_RunFlag = True  # 控制服务器运行状态的标志位
        self.thread = threading.Thread(target=self.sync_server)
        self.thread.start()

    async def data_server(self, websocket):
        try:
            async for message in websocket:
                if message == 'droidpad':  # 握手消息
                    print(f"Received message: {message}, Shake hands successfully")
                    await websocket.send(f"droidup")
                else:
                    try:
                        # 尝试解析 JSON 格式的消息
                        data = json.loads(message)
                        self.robot_cmd = data
                        print(f"Parsed message: {self.robot_cmd}")

                        # 发送 JSON 格式的消息
                        response = self.robot_status
                        await websocket.send(json.dumps(response))
                    except json.JSONDecodeError:
                        # 如果消息不是有效的 JSON 格式，发送一个简单的状态消息
                        await websocket.send('status:OK')
        except websockets.ConnectionClosed as e:
            # 处理客户端断开连接的情况
            print(f"Client disconnected: {e}")
        finally:
            # 确保关闭 WebSocket 连接
            await websocket.close()

    async def run_server(self):
        # 启动 WebSocket 服务器
        self.server = await websockets.serve(self.data_server, self.host, self.port)
        print(f"WebSocket server started on ws://{self.host}:{self.port}. Waiting for connections...")

        try:
            while self.DataServer_RunFlag:
                await asyncio.sleep(1)  # 每秒检查一次
        finally:
            # 如果标志位为 False，关闭服务器
            print("Server is shutting down...")
            self.server.close()  # 关闭 WebSocket 服务器
            await self.server.wait_closed()  # 等待服务器关闭
            print("Server has been shut down.")

    def sync_server(self):
        asyncio.run(self.run_server())

    def stop_server(self):
        # 设置标志位为 False，触发服务器关闭
        self.DataServer_RunFlag = False
        if self.thread:
            self.thread.join()  # 等待线程结束
        print("Server shutdown requested.")

# 创建并运行 WebSocket 服务器
if __name__ == "__main__":
    server = WebSocketDataServer()
    server.start_server(host="127.0.0.1", port=8765)
    asyncio.run(asyncio.sleep(100))  # 等待 100 秒
    server.stop_server()