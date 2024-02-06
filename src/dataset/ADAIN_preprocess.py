import os
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

from src.utils.config import Config


class ADAINPreprocess:
    def __init__(self, config):
        dataset_path = os.path.join(config.dataset_dir, "UrbanAir_processed.csv.gz")
        self.data_path = config.dataset_dir
        self.time_window = 24
        self.raw_data = pd.read_csv(dataset_path)
        self.raw_data = pd.get_dummies(self.raw_data, columns=['weather', 'wind_direction'])
        self.train_dataset, self.test_dataset, self.train_stations_list, self.test_stations_list = self.split_data()
        cols_to_scale = ['temperature', 'latitude', 'longitude', 'wind_speed', 'humidity']
        i = 0
        self.train_data = self.train_dataset[i]
        self.test_data = self.test_dataset[i]
        self.train_stations = self.train_stations_list[i]
        self.test_stations = self.test_stations_list[i]
        scaler = RobustScaler().fit(self.train_data[cols_to_scale])
        self.train_data[cols_to_scale] = scaler.transform(self.train_data[cols_to_scale])
        self.test_data[cols_to_scale] = scaler.transform(self.test_data[cols_to_scale])
        self.time_range = self.train_data.index.unique()

        train_combos = []
        test_combos = []
        for j in range(len(self.time_range)-24+1):
            train_combos.append(self.get_train_features(j))
            test_combos.append(self.get_test_features(j))

        for combo_data, name in zip([train_combos, test_combos], ['train', 'test']):
            station_metaq_data = np.concatenate([combo[0] for combo in combo_data])
            station_dist_data = np.concatenate([combo[1] for combo in combo_data])
            local_met_data = np.concatenate([combo[2] for combo in combo_data])
            local_aq_data = np.concatenate([combo[3] for combo in combo_data])
            local_stations = np.concatenate([combo[4] for combo in combo_data])
            np.save(os.path.join(self.data_path, f"UrbanAir_fold_{i}_{name}_station_metaq_data.npy"),
                    station_metaq_data)
            np.save(os.path.join(self.data_path, f"UrbanAir_fold_{i}_{name}_station_dist_data.npy"), station_dist_data)
            np.save(os.path.join(self.data_path, f"UrbanAir_fold_{i}_{name}_local_met_data.npy"), local_met_data)
            np.save(os.path.join(self.data_path, f"UrbanAir_fold_{i}_{name}_local_aq_data.npy"), local_aq_data)
            np.save(os.path.join(self.data_path, f"UrbanAir_fold_{i}_{name}_local_stationids.npy"), local_stations)

    def split_data(self):
        all_stations = self.raw_data.station_id.unique()
        splitter = KFold(n_splits=3, shuffle=True, random_state=1234)

        train_dataset, test_dataset, train_stations_list, test_stations_list = [], [], [], []
        for f_i, (train, test) in enumerate(splitter.split(all_stations)):
            train_stations = all_stations[train]
            test_stations = all_stations[test]
            train_data = self.raw_data[self.raw_data.station_id.isin(train_stations)]
            test_data = self.raw_data[self.raw_data.station_id.isin(test_stations)]
            train_data['time'] = pd.to_datetime(train_data['time'])
            test_data['time'] = pd.to_datetime(test_data['time'])
            train_data = train_data.set_index('time').sort_values(['time', 'station_id'])
            test_data = test_data.set_index('time').sort_values(['time', 'station_id'])
            train_dataset.append(train_data)
            test_dataset.append(test_data)
            train_stations_list.append(train_stations)
            test_stations_list.append(test_stations)
        return train_dataset, test_dataset, train_stations_list,test_stations_list

    def get_train_features(self, i):
        # For train data
        station_metaq_data = []
        station_dist_data = []
        local_met_data = []
        local_aq_data = []
        local_stations = []

        tmp_df = self.train_data.loc[self.time_range[i:i+self.time_window]]
        for station in self.train_stations:
            # Station side
            station_side = tmp_df[tmp_df.station_id != station]
            station_met_aq = station_side.drop(columns=['station_id', 'longitude', 'latitude', 'filled'])
            station_met_aq2 = np.array(np.split(station_met_aq.values, self.time_window, axis=0))
            station_met_aq2 = station_met_aq2.swapaxes(0, 1).swapaxes(1, 2)[np.newaxis, :]
            station_metaq_data.append(station_met_aq2)

            # Local side
            local_side = tmp_df[tmp_df.station_id == station]
            local_stations.append(local_side['station_id'].values[-1].reshape(-1, 1))
            local_met = local_side.drop(
                columns=['station_id', 'longitude', 'latitude', 'PM25_Concentration', 'filled']).values.swapaxes(0, 1)[
                        np.newaxis, :]
            local_met_data.append(local_met)
            local_aq = local_side['PM25_Concentration'].values[-1].reshape(-1, 1)
            local_aq_data.append(local_aq)

            station_dist = (station_side.drop_duplicates('station_id')[['longitude', 'latitude']].values - \
                            local_side.drop_duplicates('station_id')[['longitude', 'latitude']].values)[np.newaxis, :]
            station_dist_data.append(station_dist)
        return [np.concatenate(station_metaq_data),
                np.concatenate(station_dist_data),
                np.concatenate(local_met_data),
                np.concatenate(local_aq_data),
                np.concatenate(local_stations)]

    def get_test_features(self, i):
        # For test data
        station_metaq_data = []
        station_dist_data = []
        local_met_data = []
        local_aq_data = []
        local_stations = []

        tmp_df_tst = self.test_data.loc[self.time_range[i:i + self.time_window]]

        station_side = self.train_data.loc[self.time_range[i:i + self.time_window]]
        station_met_aq = station_side.drop(columns=['station_id', 'longitude', 'latitude', 'filled'])
        station_met_aq2 = np.array(np.split(station_met_aq.values, self.time_window, axis=0)).swapaxes(0, 1).swapaxes(1, 2)[
                          np.newaxis, :]   # 1*7300()*30(feature_num)*24(time_win)

        for station in self.test_stations:
            station_metaq_data.append(station_met_aq2)

            # Local side
            local_side = tmp_df_tst[tmp_df_tst.station_id == station]
            local_stations.append(local_side['station_id'].values[-1].reshape(-1, 1))
            local_met = local_side.drop(columns=['station_id', 'longitude', 'latitude',
                                                 'PM25_Concentration', 'filled']).values.swapaxes(0, 1)[np.newaxis, :]
            local_met_data.append(local_met)  # 1*29(feature_num)*8760
            #         print('local_features', local_met.columns)
            local_aq = local_side['PM25_Concentration'].values[-1].reshape(-1, 1)
            local_aq_data.append(local_aq)

            station_dist = (station_side.drop_duplicates('station_id')[['longitude', 'latitude']].values - \
                            local_side.drop_duplicates('station_id')[['longitude', 'latitude']].values)[np.newaxis, :]
            station_dist_data.append(station_dist)  # 1*20*2

        return [np.concatenate(station_metaq_data),
                np.concatenate(station_dist_data),
                np.concatenate(local_met_data),
                np.concatenate(local_aq_data),
                np.concatenate(local_stations)]

def get_train_features(i):
    global train_data
    global time_window
    global time_range
    # For train data
    station_metaq_data = []
    station_dist_data = []
    local_met_data = []
    local_aq_data = []
    local_stations = []

    tmp_df = train_data.loc[time_range[i:i + time_window]]
    for station in train_stations:
        # Station side
        station_side = tmp_df[tmp_df.station_id != station]
        station_met_aq = station_side.drop(columns=['station_id', 'longitude', 'latitude', 'filled'])
        #         clear_output(wait=True)
        #         print('station_features', station_met_aq.columns)
        station_met_aq2 = np.array(np.split(station_met_aq.values, time_window, axis=0), dtype=np.float32).swapaxes(0, 1).swapaxes(1, 2)[
                          np.newaxis, :]
        station_metaq_data.append(station_met_aq2)

        # Local side
        local_side = tmp_df[tmp_df.station_id == station]
        local_stations.append(local_side['station_id'].values[-1].reshape(-1, 1))
        local_met = local_side.drop(
            columns=['station_id', 'longitude', 'latitude', 'PM25_Concentration', 'filled']).values.swapaxes(0, 1)[
                    np.newaxis, :]
        local_met_data.append(local_met)
        local_aq = local_side['PM25_Concentration'].values[-1].reshape(-1, 1)
        local_aq_data.append(local_aq)

        station_dist = (station_side.drop_duplicates('station_id')[['longitude', 'latitude']].values - \
                        local_side.drop_duplicates('station_id')[['longitude', 'latitude']].values)[np.newaxis, :]
        station_dist_data.append(station_dist)
    return [np.concatenate(station_metaq_data),
            np.concatenate(station_dist_data),
            np.concatenate(local_met_data),
            np.concatenate(local_aq_data),
            np.concatenate(local_stations)]


def get_test_features(i):
    global train_data
    global test_data
    global time_window
    global time_range
    # For test data
    station_metaq_data = []
    station_dist_data = []
    local_met_data = []
    local_aq_data = []
    local_stations = []

    tmp_df_tst = test_data.loc[time_range[i:i + time_window]]

    station_side = train_data.loc[time_range[i:i + time_window]]
    station_met_aq = station_side.drop(columns=['station_id', 'longitude', 'latitude', 'filled'])
    #     clear_output(wait=True)
    #     print('station_features', station_met_aq.columns)
    station_met_aq2 = np.array(np.split(station_met_aq.values, time_window, axis=0), dtype=np.float32).swapaxes(0, 1).swapaxes(1, 2)[
                      np.newaxis, :]

    for station in test_stations:
        station_metaq_data.append(station_met_aq2)

        # Local side
        local_side = tmp_df_tst[tmp_df_tst.station_id == station]
        local_stations.append(local_side['station_id'].values[-1].reshape(-1, 1))
        local_met = local_side.drop(columns=['station_id', 'longitude', 'latitude',
                                             'PM25_Concentration', 'filled']).values.swapaxes(0, 1)[np.newaxis, :]
        local_met_data.append(local_met)
        #         print('local_features', local_met.columns)
        local_aq = local_side['PM25_Concentration'].values[-1].reshape(-1, 1)
        local_aq_data.append(local_aq)

        station_dist = (station_side.drop_duplicates('station_id')[['longitude', 'latitude']].values - \
                        local_side.drop_duplicates('station_id')[['longitude', 'latitude']].values)[np.newaxis, :]
        station_dist_data.append(station_dist)

    return [np.concatenate(station_metaq_data),
            np.concatenate(station_dist_data),
            np.concatenate(local_met_data),
            np.concatenate(local_aq_data),
            np.concatenate(local_stations)]


config = Config()
fold = 0  # not using for loop to avoid ram overflow

train_data = pd.read_csv(os.path.join(config.dataset_dir, f'UrbanAir_fold_{fold}_train_mar.csv.gz'))
train_data['time'] = pd.to_datetime(train_data['time'])
train_data = train_data.set_index('time').sort_values(['time', 'station_id'])
test_data = pd.read_csv(os.path.join(config.dataset_dir, f'UrbanAir_fold_{fold}_test_mar.csv.gz'))
test_data['time'] = pd.to_datetime(test_data['time'])
test_data = test_data.set_index('time').sort_values(['time', 'station_id'])
time_window = 24
time_range = train_data.index.unique()
train_stations = train_data.station_id.unique()
test_stations = test_data.station_id.unique()

if __name__ == "__main__":
    cols_to_scale = ['temperature', 'latitude', 'longitude', 'wind_speed', 'humidity']

    scaler = RobustScaler().fit(train_data[cols_to_scale])
    train_data[cols_to_scale] = scaler.transform(train_data[cols_to_scale])
    test_data[cols_to_scale] = scaler.transform(test_data[cols_to_scale])

    workers = Pool(12)
    train_combo = workers.map(get_train_features, range(len(time_range) - 24 + 1))
    print('train finished')
    test_combo = workers.map(get_test_features, range(len(time_range) - 24 + 1))
    print('test finished')
    workers.close()

    for combo_data, name in zip([train_combo, test_combo], ['train', 'test']):
        station_metaq_data = np.concatenate([combo[0] for combo in combo_data])
        station_dist_data = np.concatenate([combo[1] for combo in combo_data])
        local_met_data = np.concatenate([combo[2] for combo in combo_data])
        local_aq_data = np.concatenate([combo[3] for combo in combo_data])
        local_stations = np.concatenate([combo[4] for combo in combo_data])
        np.save(os.path.join(config.dataset_dir, f"UrbanAir_fold_{fold}_{name}_station_metaq_data.npy"),
                station_metaq_data)
        np.save(os.path.join(config.dataset_dir, f"UrbanAir_fold_{fold}_{name}_station_dist_data.npy"), station_dist_data)
        np.save(os.path.join(config.dataset_dir, f"UrbanAir_fold_{fold}_{name}_local_met_data.npy"), local_met_data)
        np.save(os.path.join(config.dataset_dir, f"UrbanAir_fold_{fold}_{name}_local_aq_data.npy"), local_aq_data)
        np.save(os.path.join(config.dataset_dir, f"UrbanAir_fold_{fold}_{name}_local_stationids.npy"), local_stations)
