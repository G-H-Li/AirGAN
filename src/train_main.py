from src.trainer.adain_trainer import AdainTrainer
from src.trainer.gan_trainer import GAN_Trainer
from src.trainer.nbst_trainer import NBSTTrainer
from src.trainer.reference_base_trainer import MLBaseTrainer
from src.trainer.sim_trainer import SimTrainer
from src.trainer.st_trainer import STTrainer

if __name__ == '__main__':
    # trainer = GAN_Trainer(mode='train')
    # trainer = SimTrainer(mode='train')
    # trainer = STTrainer(mode='train')
    # trainer = NBSTTrainer()
    # trainer = AdainTrainer()
    trainer = MLBaseTrainer(mode='train')
    trainer.run()
