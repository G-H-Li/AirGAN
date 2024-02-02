import os

import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import Config

plt.rcParams['figure.dpi'] = 300


def read_predict_res(config, model_name: str, hist_len: int, pred_len: int, predict_mode: str = "group"):
    predict = np.load(os.path.join(config.results_dir, f"{model_name}_predict_{hist_len}_{pred_len}.npy"))
    label = np.load(os.path.join(config.results_dir, f"{model_name}_label_{hist_len}_{pred_len}.npy"))
    if predict_mode == "group":
        predict = predict[:, :, :, 0].transpose((0, 2, 1))
        predict = predict.reshape(-1, predict.shape[-1], 1)
        label = label[:, :, :, 0].transpose((0, 2, 1))
        label = label.reshape(-1, label.shape[-1], 1)

    mae = np.mean(np.mean(np.abs(predict - label), axis=-1), axis=0)
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=-1)), axis=0)
    mse = np.mean(np.mean(np.square(predict - label), axis=-1), axis=0)
    y_true_nonzero = np.where(label == 0, 1e-10, label)
    mre = np.mean(np.mean(np.abs((predict - label) / y_true_nonzero), axis=-1), axis=0)

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


if __name__ == "__main__":
    config = Config()
    read_predict_res(config, "SimST", 8, 24, "city")
    # read_predict_res(config, "PM25_GNN", 8, 24, "group")
    # plt_trend_and_pm25(config)
