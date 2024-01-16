from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

import signatory
import torch
from sklearn.linear_model import LinearRegression
from torch import optim

from tqdm import tqdm

from src.model.Cells_FNN import SimpleGenerator
from src.utils.metrics import get_standard_test_metrics
from src.utils.utils import to_numpy, sample_indices


@dataclass
class SignatureConfig:
    augmentations: Tuple
    depth: int
    basepoint: bool = False


def _apply_augmentation(x: torch.Tensor, y: torch.Tensor, augmentation) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    if type(augmentation).__name__ == 'Concat':
        return y, augmentation.apply(x, y)
    else:
        return y, augmentation.apply(y)


def apply_augmentations(x: torch.Tensor, augmentations: Tuple) -> torch.Tensor:
    y = x
    for augmentation in augmentations:
        x, y = _apply_augmentation(x, y, augmentation)
    return y


def augment_path_and_compute_signatures(x: torch.Tensor, config: SignatureConfig) -> torch.Tensor:
    y = apply_augmentations(x, config.augmentations)
    return signatory.signature(y, config.depth, basepoint=config.basepoint)


@dataclass
class SigCWGANConfig:
    mc_size: int
    sig_config_future: SignatureConfig
    sig_config_past: SignatureConfig

    def compute_sig_past(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_past)

    def compute_sig_future(self, x):
        return augment_path_and_compute_signatures(x, self.sig_config_future)


def calibrate_sigw1_metric(config, x_future, x_past):
    sigs_past = config.compute_sig_past(x_past)
    sigs_future = config.compute_sig_future(x_future)
    assert sigs_past.size(0) == sigs_future.size(0)
    X, Y = to_numpy(sigs_past), to_numpy(sigs_future)
    lm = LinearRegression()
    lm.fit(X, Y)
    sigs_pred = torch.from_numpy(lm.predict(X)).float().to(x_future.device)
    return sigs_pred


def sample_sig_fake(G, q, sig_config, x_past):
    x_past_mc = x_past.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    x_fake = G.sample(q, x_past_mc)
    sigs_fake_future = sig_config.compute_sig_future(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake


def sigcwgan_loss(sig_pred: torch.Tensor, sig_fake_conditional_expectation: torch.Tensor):
    return torch.norm(sig_pred - sig_fake_conditional_expectation, p=2, dim=1).mean()


class SigCWGAN:
    def __init__(
            self,
            base_config,
            config: SigCWGANConfig,
            x_real: torch.Tensor,
    ):
        self.base_config = base_config
        self.batch_size = base_config.batch_size
        self.hidden_dims = base_config.hidden_dims
        self.p, self.q = base_config.p, base_config.q
        self.total_steps = base_config.total_steps

        self.device = base_config.device

        self.x_real = x_real
        self.dim = self.latent_dim = x_real.shape[-1]

        self.G = SimpleGenerator(self.p * self.dim, self.dim, self.hidden_dims, self.latent_dim).to(self.device)

        self.sig_config = config
        self.mc_size = config.mc_size

        self.x_past = x_real[:, :self.p]
        x_future = x_real[:, self.p:]
        self.sigs_pred = calibrate_sigw1_metric(config, x_future, self.x_past)

        self.training_loss = defaultdict(list)
        self.test_metrics_list = get_standard_test_metrics(x_real)

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-2)
        self.G_scheduler = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=100, gamma=0.9)

    def sample_batch(self, ):
        random_indices = sample_indices(self.sigs_pred.shape[0], self.batch_size)  # sample indices
        # sample the least squares signature and the log-rtn condition
        sigs_pred = self.sigs_pred[random_indices].clone().to(self.device)
        x_past = self.x_past[random_indices].clone().to(self.device)
        return sigs_pred, x_past

    def step(self):
        self.G.train()
        self.G_optimizer.zero_grad()  # empty 'cache' of gradients
        sigs_pred, x_past = self.sample_batch()
        sigs_fake_ce, x_fake = sample_sig_fake(self.G, self.q, self.sig_config, x_past)
        loss = sigcwgan_loss(sigs_pred, sigs_fake_ce)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10)
        self.training_loss['loss'].append(loss.item())
        self.training_loss['total_norm'].append(total_norm)
        self.G_optimizer.step()
        self.G_scheduler.step()  # decaying learning rate slowly.
        self.evaluate(x_fake)

    def fit(self):
        if self.batch_size == -1:
            self.batch_size = self.x_real.shape[0]
        for _ in tqdm(range(self.total_steps), ncols=80):  # sig_pred, x_past, x_real
            self.step()

    def evaluate(self, x_fake):
        for test_metric in self.test_metrics_list:
            with torch.no_grad():
                test_metric(x_fake[:10000])
            self.training_loss[test_metric.name].append(
                to_numpy(test_metric.loss_componentwise)
            )
