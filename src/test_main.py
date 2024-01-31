import os.path

from src.trainer.gan_trainer import GAN_Trainer
from src.trainer.sim_trainer import SimTrainer
from src.trainer.st_trainer import STTrainer

if __name__ == '__main__':
    # trainer = GAN_Trainer(mode='test')
    # trainer = SimTrainer(mode='test')
    trainer = STTrainer(mode='test')
    trainer.run_test(os.path.join(trainer.config.results_dir, "model_PM25_GNN.pth"), 8, 24)
