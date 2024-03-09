from src.trainer.gan_trainer import GAN_Trainer
from src.trainer.nbst_trainer import NBSTTrainer
from src.trainer.sim_trainer import SimTrainer
from src.trainer.st_trainer import STTrainer

if __name__ == '__main__':
    # trainer = GAN_Trainer(mode='train')
    # trainer = SimTrainer(mode='train')
    # trainer = STTrainer(mode='train')
    trainer = NBSTTrainer()
    trainer.run()
