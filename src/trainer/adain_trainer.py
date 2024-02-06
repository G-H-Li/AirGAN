import os
import shutil

import numpy as np
import tensorflow as tf
import arrow

from src.reference_model.ADAIN import ADAIN
from src.utils.config import Config
from src.utils.logger import TrainLogger


class AdainTrainer:
    def __init__(self):
        self.config = Config(config_filename='ADAIN_config.yaml')
        self.mode = 'train'
        # log setting
        self._create_records()
        self.logger = TrainLogger(os.path.join(self.record_dir, 'progress.log')).logger
        # self._get_device()
        # self._set_seed()
        self._load_data()
        self.time_window = 24
        met = 29
        dist = 2
        aq = 1
        self.model = ADAIN(met, dist, aq, self.time_window, self.config.dropout)

    def _set_seed(self):
        if self.config.seed != 0:
            tf.random.set_seed(self.config.seed)

    def _get_device(self):
        gpus = tf.config.list_physical_devices()
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    def _create_records(self):
        """
        Create the records directory for experiments
        :return:
        """
        exp_datetime = arrow.now().format('YYYYMMDDHHmmss')
        self.record_dir = os.path.join(self.config.records_dir, f'{self.config.model_name}_{self.mode}_{exp_datetime}')
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

    def _load_data(self):
        self.train_station_metaq = np.load(os.path.join(self.config.dataset_dir,
                                                        'UrbanAir_fold_0_train_station_metaq_data.npy'))
        self.train_station_dist = np.load(os.path.join(self.config.dataset_dir,
                                                       'UrbanAir_fold_0_train_station_dist_data.npy'))
        self.train_local_met = np.load(os.path.join(self.config.dataset_dir,
                                                    'UrbanAir_fold_0_train_local_met_data.npy'))
        self.train_local_aq = np.load(os.path.join(self.config.dataset_dir,
                                                   'UrbanAir_fold_0_train_local_aq_data.npy'))
        self.test_station_metaq = np.load(os.path.join(self.config.dataset_dir,
                                                       'UrbanAir_fold_0_test_station_metaq_data.npy'))
        self.test_station_dist = np.load(os.path.join(self.config.dataset_dir,
                                                      'UrbanAir_fold_0_test_station_dist_data.npy'))
        self.test_local_met = np.load(os.path.join(self.config.dataset_dir,
                                                   'UrbanAir_fold_0_test_local_met_data.npy'))
        self.test_local_aq = np.load(os.path.join(self.config.dataset_dir,
                                                  'UrbanAir_fold_0_test_local_aq_data.npy'))

    def _train(self):
        self.model.compile(loss='mse',
                           optimizer='adam',
                           metrics=[tf.keras.metrics.RootMeanSquaredError()])

        history = self.model.fit(x=[self.train_local_met, self.train_station_dist, self.train_station_metaq],
                                 y=self.train_local_aq, batch_size=self.config.batch_size,
                                 validation_split=0.1, epochs=self.config.epochs, verbose=1)
        return history

    def run(self):
        try:
            shutil.copy(self.config.model_config_path, os.path.join(self.record_dir,
                                                                    f'{self.config.model_name}_config.yaml'))
            self.logger.debug('config.yaml copied')
        except IOError as e:
            self.logger.error(f'Error copying config file: {e}')
        self.logger.debug('Start experiment...')
        for exp in range(self.config.exp_times):
            self.logger.info(f'Current experiment : {exp}')
            exp_dir = os.path.join(self.record_dir, f'exp_{exp}')
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            history = self._train()
            output_data_np = np.array(history)
            # 将 NumPy 数组保存到文件中
            np.save(os.path.join(exp_dir, 'output_data.npy'), output_data_np)
            self.model.save(os.path.join(exp_dir, f'model_{self.config.model_name}.h5'), save_format='h5')


if __name__ == '__main__':
    trainer = AdainTrainer()
    trainer.run()
