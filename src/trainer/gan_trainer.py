import os.path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.CW_GAN import CW_GAN
from src.trainer.trainer import Trainer
from src.utils.metrics import get_metrics
from src.utils.utils import get_mean_std


class GAN_Trainer(Trainer):
    """
    Trainer class
    General purpose training:
    1. load setting
    2. load dataset
    3. train model
    4. test model
    """
    def __init__(self):
        super().__init__()
        # model setting
        self.model = self._get_model()

    def _get_model(self):
        """
        construct model
        :return: model object
        """
        if self.config.model_name == 'CW_GAN':
            return CW_GAN(self.config, self.device, self.edge_index,
                          self.edge_attr, self.city_loc, self.train_dataset.feature.shape[-1], self.logger)
        else:
            self.logger.error('Unsupported model name')
            raise Exception('Wrong model name')

    def _train(self, train_loader):
        """
        Train model
        :return: train loss
        """
        train_loss_d = 0
        train_loss_g = 0
        for batch_idx, data in tqdm(enumerate(train_loader)):
            pm25, feature, em_feature = data
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            feature_hist = feature[:, :self.config.hist_len]
            loss_d, loss_g = self.model.batch_train(pm25_hist, feature_hist, pm25_label)
            train_loss_d += loss_d.item()
            train_loss_g += loss_g.item()

        train_loss_g /= len(train_loader) + 1
        train_loss_d /= len(train_loader) + 1
        return train_loss_d, train_loss_g

    def _valid(self, valid_loader):
        """
        Validate model
        :return: validation loss
        """
        val_loss = 0
        for batch_idx, data in tqdm(enumerate(valid_loader)):
            pm25, feature, em_feature = data
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            feature_hist = feature[:, :self.config.hist_len]
            loss = self.model.batch_valid(pm25_hist, feature_hist, pm25_label)
            val_loss += loss.item()

        val_loss /= len(valid_loader) + 1
        return val_loss

    def _test(self, test_loader):
        """
        Test model
        :return: test loss
        """
        predict_list = []
        label_list = []
        test_loss = 0
        for batch_idx, data in tqdm(enumerate(test_loader)):
            pm25, feature, em_feature = data
            pm25_label = pm25[:, self.config.hist_len:]
            pm25_hist = pm25[:, :self.config.hist_len]
            feature_hist = feature[:, :self.config.hist_len]
            loss, pm25_pred, pm25_label = self.model.batch_test(pm25_hist, feature_hist, pm25_label)
            test_loss += loss.item()

            pm25_pred_val = self.test_dataset.pm25_scaler.denormalize(pm25_pred.cpu().detach().numpy())
            pm25_label_val = self.test_dataset.pm25_scaler.denormalize(pm25_label.cpu().detach().numpy())
            predict_list.append(pm25_pred_val)
            label_list.append(pm25_label_val)

        test_loss /= len(test_loader) + 1

        predict_epoch = np.concatenate(predict_list, axis=0)
        label_epoch = np.concatenate(label_list, axis=0)
        predict_epoch[predict_epoch < 0] = 0
        return test_loss, predict_epoch, label_epoch

    def run(self):
        """
        do experiment training
        :return:
        """
        # save config file
        try:
            shutil.copy(self.config.config_path, os.path.join(self.record_dir, 'base_config.yaml'))
            shutil.copy(self.config.model_config_path, os.path.join(self.record_dir,
                                                                    f'{self.config.model_name}_config.yaml'))
            self.logger.debug('base_config.yaml copied')
        except IOError as e:
            self.logger.error(f'Error copying config file: {e}')
        self.logger.debug('Start experiment...')
        for exp in range(self.config.exp_times):
            self.logger.info(f'Current experiment : {exp}')
            exp_dir = os.path.join(self.record_dir, f'exp_{exp}')
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            # create data loader
            train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                      drop_last=True)
            valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.batch_size, shuffle=False,
                                      drop_last=True)
            test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                     drop_last=True)
            # epoch variants
            best_epoch = 0
            self.train_loss_list = []
            self.test_loss_list = []
            self.valid_loss_list = []
            self.rmse_list = []
            self.mae_list = []
            self.csi_list = []
            self.pod_list = []
            self.far_list = []

            for epoch in range(self.config.epochs):
                self.logger.debug(f'Experiment time :{exp}, Epoch time : {epoch}')
                train_loss_d, train_loss_g = self._train(train_loader)
                val_loss = self._valid(valid_loader)
                self.logger.info('train_loss_d: %.4f, train_loss_g: %.4f, val_loss: %.4f' %
                                 (train_loss_d, train_loss_g, val_loss))
                # End train without the best result in consecutive early_stop epochs
                if epoch - best_epoch > self.config.early_stop and self.config.is_early_stop:
                    self.logger.info('Early stop at epoch {}, best loss = {:.6f}'
                                     .format(epoch, np.min(self.valid_loss_list)))
                    break
                # update val loss
                best_val_loss = np.min(self.valid_loss_list) if len(self.valid_loss_list) > 0 else np.inf
                if val_loss < best_val_loss:
                    best_epoch = epoch
                    # test model
                    test_loss, predict_epoch, label_epoch = self._test(test_loader)
                    rmse, mae, csi, pod, far = get_metrics(predict_epoch, label_epoch)

                    self.logger.info('Epoch time: %d, test results: \n'
                                     'Train loss_d: %0.4f, Train loss_g: %0.4f, Val loss: %0.4f, Test loss: %0.4f \n'
                                     'RMSE: %0.2f, MAE: %0.2f \n'
                                     'CSI: %0.4f, POD: %0.4f, FAR: %0.4f'
                                     % (epoch, train_loss_d, train_loss_g, val_loss, test_loss, rmse, mae, csi, pod, far))
                    self.train_loss_list.append([train_loss_d, train_loss_g])
                    self.valid_loss_list.append(val_loss)
                    self.test_loss_list.append(test_loss)
                    self.rmse_list.append(rmse)
                    self.mae_list.append(mae)
                    self.csi_list.append(csi)
                    self.pod_list.append(pod)
                    self.far_list.append(far)
                    # save model
                    torch.save(self.model.discriminator.state_dict(),
                               os.path.join(exp_dir, f'model_{self.config.model_name}_d.pth'))
                    torch.save(self.model.generator.state_dict(),
                               os.path.join(exp_dir, f'model_{self.config.model_name}_g.pth'))
                    # save prediction and label
                    if self.config.save_npy:
                        np.save(os.path.join(exp_dir, f'predict.npy'), predict_epoch)
                        np.save(os.path.join(exp_dir, f'label.npy'), label_epoch)
                        self.logger.info(f'Save model and results at epoch {epoch}')
                    else:
                        self.logger.info(f'Save model at epoch {epoch}')

            self.logger.info('Experiment time: %d, test results: \n'
                             'Train loss_d: %0.4f, Train loss_g: %0.4f, Val loss: %0.4f, Test loss: %0.4f \n'
                             'RMSE: %0.2f, MAE: %0.2f \n'
                             'CSI: %0.4f, POD: %0.4f, FAR: %0.4f'
                             % (exp, self.train_loss_list[-1][0], self.train_loss_list[-1][1], self.valid_loss_list[-1], self.test_loss_list[-1],
                                self.rmse_list[-1], self.mae_list[-1], self.csi_list[-1],
                                self.pod_list[-1], self.far_list[-1]))
            self.exp_train_loss_list.append(self.train_loss_list[-1])
            self.exp_test_loss_list.append(self.test_loss_list[-1])
            self.exp_valid_loss_list.append(self.valid_loss_list[-1])
            self.exp_rmse_list.append(self.rmse_list[-1])
            self.exp_mae_list.append(self.mae_list[-1])
            self.exp_csi_list.append(self.csi_list[-1])
            self.exp_pod_list.append(self.pod_list[-1])
            self.exp_far_list.append(self.far_list[-1])

            # save metrics
            metrics_data = np.concatenate((np.array(self.train_loss_list), np.array(self.valid_loss_list),
                                           np.array(self.test_loss_list), np.array(self.rmse_list),
                                           np.array(self.mae_list), np.array(self.csi_list),
                                           np.array(self.pod_list), np.array(self.far_list)), axis=0)
            np.save(os.path.join(exp_dir, f'exp_{exp}_res.npy'), metrics_data)

        self.logger.info("Finished all experiments: \n"
                         'train_loss_d| mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_train_loss_list[:, 0])) +
                         'train_loss_g| mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_train_loss_list[:, 1])) +
                         'val_loss    | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_valid_loss_list)) +
                         'test_loss   | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_test_loss_list)) +
                         'RMSE        | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_rmse_list)) +
                         'MAE         | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_mae_list)) +
                         'CSI         | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_csi_list)) +
                         'POD         | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_pod_list)) +
                         'FAR         | mean: %0.4f std: %0.4f\n' % (get_mean_std(self.exp_far_list)))
        metrics_data = np.concatenate((np.array(self.exp_train_loss_list), np.array(self.exp_valid_loss_list),
                                       np.array(self.exp_test_loss_list), np.array(self.exp_rmse_list),
                                       np.array(self.exp_mae_list), np.array(self.exp_csi_list),
                                       np.array(self.exp_pod_list), np.array(self.exp_far_list)), axis=0)
        np.save(os.path.join(self.record_dir, 'all_exp_res.npy'), metrics_data)
        self.logger.debug('Experiments finished.')
