from src.trainer.refer_trainer import ReferTrainer
from src.trainer.nbst_trainer import NBSTTrainer
from src.trainer.reference_base_trainer import MLBaseTrainer
from src.trainer.sim_trainer import SimTrainer
from src.trainer.for_trainer import ForTrainer

if __name__ == '__main__':
    trainer = SimTrainer(mode='train')
    # trainer = ForTrainer(mode='train')
    # trainer = NBSTTrainer()
    # trainer = ReferTrainer()
    # trainer = MLBaseTrainer(mode='train')
    trainer.run()
