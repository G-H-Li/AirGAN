import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# TODO demo
# 生成示例数据
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

# 创建SVR模型
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# 训练模型
svr_model.fit(X, y)

# 预测
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_pred = svr_model.predict(X_test)

# 绘制结果
plt.scatter(X, y, label='Data', color='darkorange')
plt.plot(X_test, y_pred, color='navy', label='SVR (RBF Kernel)')
plt.title('Support Vector Regression')
plt.xlabel('Data')
plt.ylabel('Target')
plt.legend()
plt.show()
