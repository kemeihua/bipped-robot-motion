import json
import asyncio
import threading
import websockets
from time import sleep

class WebSocketDataServer:
    def __init__(self):
        self.thread = None
        self.port = 8765
        self.host = "127.0.0.1"
        self.voice_cmd = {"arm_mode": "init"}
        self.DataServer_RunFlag = None
        self.server = None  # 用于存储 WebSocket 服务器实例
        self.robot_status={"arm_mode": "none"} # init, hello, shake

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
                        # print(f"Parsed message: {data}")
                        # 将当前遥控器状态存储到实例变量中
                        self.voice_cmd = {"arm_mode":  data['arm_mode']}

                        # 打印提取的字段
                        # print(f"arm_mode: {self.voice_cmd['arm_mode']}")
                        response = {"arm_mode": self.robot_status["arm_mode"]}
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
    server.start_server(host="192.168.55.95", port=8765)
    asyncio.run(asyncio.sleep(10000))  # 等待 100 秒
    server.stop_server()