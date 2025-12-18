
###数据准备
import numpy as np
import pandas as pd

# 示例数据准备
# 生成一些假设的状态（state）和动作（action）
num_samples = 1000
state_dim = 4  # 状态维度
action_dim = 2  # 动作维度（例如：左/右）

# 随机生成状态和对应的专家动作
states = np.random.rand(num_samples, state_dim)
actions = np.random.randint(0, action_dim, size=(num_samples,))

# 将数据保存为 DataFrame（方便后续处理）
data = pd.DataFrame(np.hstack((states, actions.reshape(-1, 1))), columns=[f'feature_{i}' for i in range(state_dim)] + ['action'])
data.to_csv('expert_data.csv', index=False)


###模型构建
from keras.models import Sequential
from keras.layers import Dense

# 定义行为克隆模型
def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))  # 输出动作的概率分布
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 创建模型
model = create_model(state_dim, action_dim)


###训练模型
# 从 CSV 中加载专家数据
data = pd.read_csv('expert_data.csv')
X = data.iloc[:, :-1].values  # 状态
y = data['action'].values      # 动作

# 训练模型
model.fit(X, y, epochs=20, batch_size=32)



