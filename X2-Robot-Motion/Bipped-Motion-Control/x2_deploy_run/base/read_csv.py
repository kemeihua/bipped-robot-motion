import os
import numpy as np
import glob


def load_action_sequences(folder_path='action'):
    """
    加载 action 文件夹下的所有 CSV 文件到二维 NumPy 数组（float32类型）

    参数:
        folder_path (str): 包含 CSV 文件的文件夹路径，默认为 'action'

    返回:
        dict: 键为文件名（不含扩展名），值为对应的二维 NumPy 数组（float32类型）
    """
    action_arrays = {}

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    # 获取文件夹下所有 CSV 文件
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    if not csv_files:
        print(f"警告: 在 {folder_path} 中未找到任何CSV文件")
        return action_arrays

    for file_path in csv_files:
        # 提取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        try:
            # 读取 CSV 文件到 NumPy 数组，并指定数据类型为 float32
            data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)

            # 强制转换为二维数组
            if data.ndim == 1:
                data = data.reshape(1, -1)  # 转为 (1, n)
                print(f"调整维度: {file_name}.csv -> 从 {data.shape} 转为 (1, {data.shape[0]})")

            action_arrays[file_name] = data
            print(f"成功加载: {file_name}.csv -> 形状 {data.shape}, 数据类型 {data.dtype}")

        except ValueError as e:
            print(f"数据格式错误 {file_name}.csv: {e}")
        except Exception as e:
            print(f"加载 {file_name}.csv 失败: {str(e)}")

    return action_arrays


# 使用示例
if __name__ == "__main__":
    # 加载 action 文件夹下的所有 CSV 文件
    actions = load_action_sequences(f"/home/liangzhiyuan/RL/X02/x02-sim2real_new_grpc/base/arm_action_lib/4dof")
    # print(actions['init'])
    # print(actions['hello'])
    # 打印加载的数组（示例）
    for name, array in actions.items():
        print(f"\n数组 '{name}':")
        print(array)