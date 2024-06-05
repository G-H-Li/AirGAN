import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

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


def get_r2(predict, label):
    mean = np.mean(label, axis=1).reshape(-1, 1)
    a = np.sum(np.square(label - mean))
    b = np.sum(np.square(predict - label))
    return 1 - np.sum(b / a)


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
        predict = np.load(os.path.join(config.results_dir, "for", "self-regression", f"{model_name}_predict_{hist_len}_{pred_len}.npy"))
        label = np.load(os.path.join(config.results_dir, "for", "self-regression", f"{model_name}_label_{hist_len}_{pred_len}.npy"))
        if model_name != "STNN-AirT":
            predict = predict[:, :, :, 0].transpose((0, 2, 1))
            predict = predict.reshape(-1, predict.shape[-1], 1)
            label = label[:, :, :, 0].transpose((0, 2, 1))
            label = label.reshape(-1, label.shape[-1], 1)

        mae = get_mae(predict, label)
        # mae = mae.reshape(3, -1).mean(axis=1)
        # print(f"{model_name} MAE: {mae}")
        x = np.arange(3, 3 * (mae.shape[-1] + 1), 3)
        # x = ["1-24h", "24-48h", "48-72h"]
        plt.plot(x, mae, label=model_name)

    from matplotlib.font_manager import FontProperties

    # 设置我们需要用到的中文字体（字体文件地址）
    my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=18)
    plt.legend(fontsize=12)
    plt.xlabel("预测小时数(h)", fontproperties=my_font)
    plt.ylabel("均方根误差(RMSE)", fontproperties=my_font)
    plt.show()


def plt_city_results(config, folder_names: list):
    for i in folder_names:
        r2 = []
        for j in range(3):
            predict = np.load(os.path.join(config.records_dir, i, f"exp_{j}", "predict.npy"))
            label = np.load(os.path.join(config.records_dir, i, f"exp_{j}", "label.npy"))
            # predict = np.float16(predict)
            # label = np.float16(label[:, 0])
            r2.append(get_r2(predict[..., 0], label[..., 0]))
        r2 = np.array(r2).mean(axis=0)
        print(f"{i} R2: {r2}")


def convert_npy_to_csv(config, filename: str):
    loc = np.load(os.path.join(config.dataset_dir, filename))
    loc = loc[:, :2]
    np.savetxt(os.path.join(config.results_dir, f"{filename}.csv"),
               loc.reshape(-1, loc.shape[-1]), delimiter=',', fmt='%f')


def plt_pm25(config, filename: str):
    pm25 = np.load(os.path.join(config.dataset_dir, filename))
    liangxiangPM25 = pm25[:, 3]
    liangxiangPM25 = liangxiangPM25.reshape(-1, liangxiangPM25.shape[-1])
    start_date = datetime(2014, 5, 1, 0 )
    end_date = datetime(2015, 5, 1, 0)
    sequence_length = liangxiangPM25.shape[0]
    dates = generate_dates(start_date, end_date, sequence_length)
    plt.figure(figsize=(12, 8))
    plt.xticks(rotation=30)
    # plt.plot(dates[1104: 1152], liangxiangPM25[1104: 1152])     # 6-16 to 6-18
    # plt.plot(dates[4992: 5040], liangxiangPM25[4992: 5040])   # 11-25 to 11-27
    plt.plot(dates, liangxiangPM25)
    my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=18)
    plt.legend(fontsize=12)
    plt.xlabel("时间", fontproperties=my_font)
    plt.ylabel("PM2.5浓度", fontproperties=my_font)
    plt.title("房山良乡监测站点2014年5月-2015年5月PM2.5浓度曲线", fontproperties=my_font)
    plt.show()


def plt_PM25_results(config, folder_name: str, model_names: list, idx: list):
    label1 = np.load(os.path.join(config.results_dir, "fusion", folder_name, f"label_{idx[0]}.npy"))
    label2 = np.load(os.path.join(config.results_dir, "fusion", folder_name, f"label_{idx[1]}.npy"))
    predict1 = np.load(os.path.join(config.results_dir, "fusion", folder_name, f"{model_names[0]}_predict_{idx[0]}.npy"))
    predict2 = np.load(os.path.join(config.results_dir, "fusion", folder_name, f"{model_names[0]}_predict_{idx[1]}.npy"))
    predict3 = np.load(os.path.join(config.results_dir, "fusion", folder_name, f"{model_names[1]}_predict_{idx[0]}.npy"))
    predict4 = np.load(os.path.join(config.results_dir, "fusion", folder_name, f"{model_names[1]}_predict_{idx[1]}.npy"))

    label1 = label1.reshape(24)
    label2 = label2.reshape(24)
    predict1 = predict1.reshape(24)
    predict2 = predict2.reshape(24)
    predict3 = predict3.reshape(24)
    predict4 = predict4.reshape(24)

    label = np.concatenate((label1, label2), axis=0)
    predict_no_pre = np.concatenate((predict1, predict2), axis=0)
    predict = np.concatenate((predict3, predict4), axis=0)

    start_date = datetime(2014, 6, 16, 0)
    end_date = datetime(2014, 6, 18, 0)
    dates = generate_dates(start_date, end_date, 48)
    plt.xticks(rotation=30)
    plt.figure(figsize=(10, 6))
    my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=18)
    plt.legend(fontsize=12)
    plt.xlabel("时间", fontproperties=my_font)
    plt.ylabel("PM2.5浓度", fontproperties=my_font)
    plt.plot(dates, label)
    plt.plot(dates, predict_no_pre)
    plt.plot(dates, predict)
    plt.legend(["Ground-Truth", "STNN-Air w/o Pre Station", "STNN-Air"])
    plt.show()


def generate_dates(start_date, end_date, sequence_length):
    dates = []
    current_date = start_date
    delta = (end_date - start_date) / sequence_length
    for i in range(sequence_length):
        dates.append(current_date)
        current_date += delta
    return dates


if __name__ == "__main__":
    config = Config()
    # read_predict_res(config, "SimST", 8, 24, "city")
    # read_predict_res(config, "PM25_GNN", 8, 24, "group")
    # plt_multi_method_results(config, ["MLP", "LSTM", "GRU", "GC-LSTM", "PM25-GNN", "AirFormer", "STNN-AirT"], 8, 24)
    # plt_trend_and_pm25(config)
    # plt_pm25(config, "UrbanAir_pm25.npy")
    plt_PM25_results(config, "winter", ["NBST", "ADAIN"], [4992, 5016])