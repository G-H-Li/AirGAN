import os

import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import Config

plt.rcParams['figure.dpi'] = 300


def get_mae(predict, label):
    return np.mean(np.mean(np.abs(predict - label), axis=-1), axis=0)


def get_rmse(predict, label):
    return np.mean(np.sqrt(np.mean(np.square(predict - label), axis=-1)), axis=0)


def get_mse(predict, label):
    return np.mean(np.mean(np.square(predict - label), axis=-1), axis=0)


def get_mape(predict, label):
    y_true_nonzero = np.where(label == 0, 1e-10, label)
    return np.mean(np.mean(np.abs((predict - y_true_nonzero) / y_true_nonzero), axis=-1)*100, axis=0)


def get_mre(predict, label):
    y_true_nonzero = np.where(label == 0, 1e-10, label)
    return np.mean(np.mean(np.abs((predict - label) / y_true_nonzero), axis=-1), axis=0)


def read_predict_res(config, model_name: str, hist_len: int, pred_len: int, predict_mode: str = "group"):
    predict = np.load(os.path.join(config.results_dir, f"{model_name}_predict_{hist_len}_{pred_len}.npy"))
    label = np.load(os.path.join(config.results_dir, f"{model_name}_label_{hist_len}_{pred_len}.npy"))
    if predict_mode == "group":
        predict = predict[:, :, :, 0].transpose((0, 2, 1))
        predict = predict.reshape(-1, predict.shape[-1], 1)
        label = label[:, :, :, 0].transpose((0, 2, 1))
        label = label.reshape(-1, label.shape[-1], 1)

    mae = get_mae(predict, label)
    rmse = get_rmse(predict, label)
    mse = get_mse(predict, label)
    mre = get_mre(predict, label)

    x = np.arange(3, 3 * (mae.shape[-1] + 1), 3)
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(x, mae, label="MAE", color="blue")
    axs[0, 0].set(title="MAE")
    axs[0, 1].plot(x, rmse, label="RMSE", color="red")
    axs[0, 1].set(title="RMSE")
    axs[1, 0].plot(x, mse, label="MSE", color="black")
    axs[1, 0].set(title="MSE")
    axs[1, 1].plot(x, mre, label="MRE", color="green")
    axs[1, 1].set(title="MRE")

    plt.suptitle(f'{model_name} {hist_len}_{pred_len} prediction error')
    plt.tight_layout()

    plt.show()


def plt_trend_and_pm25(config):
    pm25 = np.load(os.path.join(config.dataset_dir, 'KnowAir_PM25.npy'))
    pm25_trend = np.load(os.path.join(config.dataset_dir, 'KnowAir_PM25_trend.npy'))

    case = pm25[:100, 1].squeeze()
    trend = pm25_trend[:100, 1].squeeze()
    plt.plot(case)
    plt.plot(trend)

    plt.show()


def plt_multi_method_results(config, model_list: list, hist_len: int, pred_len: int):
    for model_name in model_list:
        predict = np.load(os.path.join(config.results_dir, f"{model_name}_predict_{hist_len}_{pred_len}.npy"))
        label = np.load(os.path.join(config.results_dir, f"{model_name}_label_{hist_len}_{pred_len}.npy"))
        if model_name != "SimST":
            predict = predict[:, :, :, 0].transpose((0, 2, 1))
            predict = predict.reshape(-1, predict.shape[-1], 1)
            label = label[:, :, :, 0].transpose((0, 2, 1))
            label = label.reshape(-1, label.shape[-1], 1)

        mae = get_mae(predict, label)
        x = np.arange(3, 3 * (mae.shape[-1] + 1), 3)
        # x = ["1-24h", "24-48h", "48-72h"]
        plt.plot(x, mae, label=model_name)

    from matplotlib.font_manager import FontProperties

    # 设置我们需要用到的中文字体（字体文件地址）
    my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)
    plt.legend()
    plt.xlabel("预测小时数(h)", fontproperties=my_font)
    plt.ylabel("平均绝对误差MAE", fontproperties=my_font)
    plt.show()


if __name__ == "__main__":
    config = Config()
    # read_predict_res(config, "SimST", 8, 24, "city")
    # read_predict_res(config, "PM25_GNN", 8, 24, "group")
    plt_multi_method_results(config, ["MLP", "LSTM", "PM25_GNN", "AirFormer", "SimST"], 8, 24)
    # plt_trend_and_pm25(config)
