import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# TODO demo
# 生成示例时间序列数据
np.random.seed(42)
time = np.arange(1, 101)
data = np.random.normal(size=100) + np.sin(np.linspace(0, 4 * np.pi, 100))

# 将数据转为pandas DataFrame
df = pd.DataFrame({'value': data})

# 绘制时间序列图
plt.figure(figsize=(10, 4))
plt.plot(time, df['value'])
plt.title('Example Time Series Data')
plt.show()

# 绘制自相关图和偏自相关图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['value'], ax=ax1, lags=20)
plot_pacf(df['value'], ax=ax2, lags=20)
plt.show()

# 拟合ARIMA模型
order = (1, 1, 1)  # (p, d, q) 参数
model = ARIMA(df['value'], order=order)
result = model.fit()

# 预测未来一步
forecast_steps = 10
forecast = result.get_forecast(steps=forecast_steps)

# 获取置信区间
forecast_ci = forecast.conf_int()

# 绘制原始数据和预测结果
plt.figure(figsize=(12, 6))
plt.plot(time, df['value'], label='Original Data')
plt.plot(np.arange(101, 101 + forecast_steps), forecast.predicted_mean, color='red', label='Forecast')
plt.fill_between(np.arange(101, 101 + forecast_steps), forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3, label='Confidence Interval')
plt.title('ARIMA Forecasting')
plt.legend()
plt.show()
