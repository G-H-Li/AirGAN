from src.dataset.parser import KnowAirDataset
from src.model.AECGAN import AECGAN
from src.trainer.trainer import Trainer


class TimeTrainer(Trainer):
    """
    Temporal model Trainer class
    1. load setting
    2. load dataset
    """

    def __init__(self):
        # read config
        super().__init__()
        # model setting
        self.model = self._get_model()

    def _get_model(self):
        self.in_dim = (self.train_dataset.feature.shape[-1] +
                       self.train_dataset.pm25.shape[-1])
        if self.config.model_name == "AECGAN":
            return AECGAN(self.config.hist_len,
                          self.config.pred_len,
                          self.in_dim,
                          self.config.batch_size,
                          self.device,
                          self.config.hidden_dim,
                          self.config.use_ec,
                          self.config.noise_type)
        else:
            raise NotImplementedError

    def _read_data(self):
        if self.config.dataset_name == "KnowAir":
            self.train_dataset = KnowAirDataset(config=self.config, mode='train')
            self.valid_dataset = KnowAirDataset(config=self.config, mode='valid')
            self.test_dataset = KnowAirDataset(config=self.config, mode='test')
            self.city_num = self.train_dataset.node_num

    def _test(self, test_loader):
        pass

    def _train(self, train_loader):
        pass

    def _valid(self, valid_loader):
        pass

    def run(self):
        pass
