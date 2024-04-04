import os.path

from src.trainer.sim_trainer import SimTrainer
from src.trainer.for_trainer import ForTrainer

if __name__ == '__main__':
    # trainer = GAN_Trainer(mode='test')
    # trainer = SimTrainer(mode='test')
    trainer = ForTrainer(mode='test')
    trainer.run_test(os.path.join(trainer.config.results_dir, f"model_{trainer.config.model_name}.pth"),
                     8, 24)
    # trainer.get_model_info()
