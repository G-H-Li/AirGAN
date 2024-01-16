from collections import defaultdict
from typing import Tuple

import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

import numpy as np

from src.model.Cells_FNN import ResFNN, ResConv1d
from src.utils.utils import toggle_grad


class ECG(nn.Module):
    def __init__(self, dim: int, p: int, q: int, hidden_dims: Tuple[int], latent_dim: int):
        super().__init__()

        self.G = ResFNN(dim * p + latent_dim, dim, hidden_dims)
        self.EC = ResConv1d(dim, dim, hidden_dims)

        self.latent_dim = latent_dim
        self.dim = dim
        self.p = p
        self.q = q

    def generate(self, z, x_past, use_ec=False):
        x_generated = list()
        for t in range(z.shape[1]):
            z_t = z[:, t:t + 1]
            x_in = torch.cat([z_t, x_past.reshape(x_past.shape[0], 1, -1)], dim=-1)
            x_gen = self.G(x_in)
            x_past = torch.cat([x_past[:, 1:], x_gen], dim=1)
            x_generated.append(x_gen)

            # update conditions
            if t % self.p == 0 and use_ec:
                residual = 0
                for _ in range(1):
                    ec_residual = self.EC(x_past)
                    ec_residual = ec_residual.reshape(x_past.shape[0], self.p, self.dim)
                    x_past = x_past + ec_residual
                    residual = residual + ec_residual
        x_fake = torch.cat(x_generated, dim=1)
        return x_fake

    def forward(self, z, x_past):
        # update the x_past with error correction
        ec_input = x_past
        for _ in range(1):
            ec_residual = self.EC(ec_input)
            ec_residual = ec_residual.reshape(x_past.shape[0], self.p, self.dim)
            ec_input = ec_input + ec_residual

        # generate data with error correction
        x_fake_ec = self.generate(z, x_past=ec_input, use_ec=True)

        # generate data without error correction
        x_fake = self.generate(z, x_past=x_past, use_ec=False)

        return x_fake, x_fake_ec, ec_input

    def test(self, z, x_past):
        x_generated = list()
        x_ec = list()
        x_residual = list()
        for t in range(z.shape[1]):
            z_t = z[:, t:t + 1]
            x_in = torch.cat([z_t, x_past.reshape(x_past.shape[0], 1, -1)], dim=-1)
            x_gen = self.G(x_in)
            x_past = torch.cat([x_past[:, 1:], x_gen], dim=1)
            x_generated.append(x_gen)

            if t % self.p == 0:
                residual = 0
                for _ in range(1):
                    EC_residual = self.EC(x_past)
                    EC_residual = EC_residual.reshape(x_past.shape[0], self.p, self.dim)
                    x_past = x_past + EC_residual
                    residual = residual + EC_residual
                x_ec.append(x_past)
                x_residual.append(residual)

        x_fake = torch.cat(x_generated, dim=1)
        x_ec = torch.cat(x_ec, dim=1)
        x_residual = torch.cat(x_residual, dim=1)
        return x_fake, x_ec, x_residual

    def sample(self, steps, x_past):
        z = torch.randn(x_past.size(0), steps, self.latent_dim).to(x_past.device)
        if steps > self.q:
            # train_step
            return self.test(z, x_past)
        # test_step
        return self.forward(z, x_past)


def compute_loss(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    return torch.nn.functional.binary_cross_entropy_with_logits(d_out, targets)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


class AECGAN:
    def __init__(self, hist_len, pred_len, in_dim, batch_size, device, hidden_dims: int, use_ec, noise_type):
        self.batch_size = batch_size
        self.hidden_dims: Tuple[int] = 3 * (hidden_dims,)
        self.p, self.q = hist_len, pred_len
        self.noise_type = noise_type
        self.use_ec = use_ec
        self.device = device
        self.dim = self.latent_dim = in_dim

        # build_model
        self.D_steps_per_G_step = 2
        self.D = ResFNN(self.dim * (self.p + self.q), self.dim * self.q, self.hidden_dims, True).to(self.device)
        self.G = ECG(self.dim, self.p, self.q, self.hidden_dims, self.latent_dim).to(self.device)

        # set learning parameters
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0, 0.9))  # Using TTUR
        self.g_optimizer = torch.optim.Adam([
            {'params': self.G.G.parameters(), 'lr': 1e-4, 'betas': (0, 0.9)},
            {'params': self.G.EC.parameters(), 'lr': 1e-3, 'betas': (0.9, 0.999)},
        ])

        self.training_loss = defaultdict(list)

    def pgd(self, x_past, x_fake):
        """
        add shuffle
            condition: x_past
            output: x_fake
        """
        outputs = self.L2_attack(self.D, x_past, x_fake)
        return outputs

    def L2_attack(self, model, x_past, x_fake, max_norm=0.2, steps=10):
        delta = torch.zeros_like(x_past, requires_grad=True)
        zeros = torch.zeros_like(x_fake, requires_grad=False)

        max_norm = np.sqrt((self.p + self.q) * max_norm ** 2)

        if self.noise_type == 'gaussian':
            delta = 0.2 * torch.randn_like(x_past)
            delta.data.renorm_(p=2, dim=0, maxnorm=max_norm)
            return x_past + delta

        batch_size = x_past.shape[0]
        inputs = torch.cat([x_past, x_fake], dim=1)
        optimizer = torch.optim.SGD([delta], lr=max_norm / steps * 2)
        for i in range(steps):
            adv = inputs + torch.cat([delta, zeros], dim=1)
            dscore = model(adv)
            dloss = compute_loss(dscore, 0)

            if self.noise_type == 'min_adv':
                loss = dloss * (-1)
            elif self.noise_type == 'max_adv':
                loss = dloss
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            loss.backward()
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1))

            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer.step()

            delta.data.renorm_(p=2, dim=0, maxnorm=max_norm)
            # delta.data.clamp_(-max_norm, max_norm)

        return x_past + delta

    def calc_gradients(self, x_past, x_fake, x_real):
        """ calculate the gradient with respect to G and D
            @ x_past: realistic history data
            @ x_fake: generated data condition on the x_past

            returns:
            @ return1: \partial L_G / \partial \hat x
            @ return2: \partial G / \partial c
        """
        batch_size = x_past.shape[0]

        x_real = x_real[:, self.p: self.p + self.q]

        delta1 = Variable(x_fake, requires_grad=True)
        inputs = torch.cat([x_past, delta1], dim=1)
        output = torch.log(1 - torch.sigmoid(self.D(inputs)))
        output = torch.mean(output, dim=1, keepdim=True)
        output.backward(torch.ones_like(output))
        D_norm_l1 = delta1.grad.view(batch_size, -1).norm(p=1, dim=1)
        D_norm_l2 = delta1.grad.view(batch_size, -1).norm(p=2, dim=1)
        D_norm_l1 = D_norm_l1.mean()
        D_norm_l2 = D_norm_l2.mean()

        delta1 = Variable(x_fake, requires_grad=True)
        input1 = torch.cat([x_past, delta1], dim=1)
        output1 = torch.log(1 - torch.sigmoid(self.D(input1)))
        output1.backward(torch.ones_like(output1))
        D_norm2_l1 = delta1.grad.view(batch_size, -1).norm(p=1, dim=1)
        D_norm2_l2 = delta1.grad.view(batch_size, -1).norm(p=2, dim=1)
        D_norm2_l1 = D_norm2_l1.mean()
        D_norm2_l2 = D_norm2_l2.mean()

        delta2 = Variable(x_past, requires_grad=True)
        g_fake, _, _ = self.G.sample(self.q, delta2)
        g_fake.backward(torch.ones_like(g_fake))
        G_norm_l1 = delta2.grad.view(batch_size, -1).norm(p=1, dim=1)
        G_norm_l2 = delta2.grad.view(batch_size, -1).norm(p=2, dim=1)
        G_norm_l1 = G_norm_l1.mean()
        G_norm_l2 = G_norm_l2.mean()

        return D_norm_l1.item(), D_norm_l2.item(), D_norm2_l1.item(), D_norm2_l2.item(), G_norm_l1.item(), G_norm_l2.item()

    def step(self, data):
        # Discriminator step
        for i in range(self.D_steps_per_G_step):
            # generate x_fake
            x_past = data[:, :self.p].clone().to(self.device)
            with torch.no_grad():
                x_fake, x_fake_ec, residual = self.G.sample(self.q, x_past.clone())
            x_past_pgd = self.pgd(x_past, x_fake)
            with torch.no_grad():
                x_fake_pgd, x_fake_pgd_ec, residual_pgd = self.G.sample(self.q, x_past_pgd)

            x_fake = torch.cat([x_past, x_fake], dim=1)
            x_fake_ec = torch.cat([x_past, x_fake_ec], dim=1)
            x_fake_past_pgd = torch.cat([x_past_pgd, x_fake_pgd], dim=1)
            x_fake_past_pgd_ec = torch.cat([x_past_pgd, x_fake_pgd_ec], dim=1)

            D_loss_real, D_loss_fake, D_loss_fake_adv, reg = self.d_trainstep(
                [x_fake, x_fake_ec],
                [x_fake_past_pgd, x_fake_past_pgd_ec],
                data.to(self.device))
            if i == 0:
                self.training_loss['D_loss_fake'].append(D_loss_fake)
                self.training_loss['D_loss_fake_adv'].append(D_loss_fake_adv)
                self.training_loss['D_loss_real'].append(D_loss_real)
        torch.cuda.empty_cache()

        # Generator step
        x_past = data[:, :self.p].clone().to(self.device)

        x_fake, x_fake_ec, residual = self.G.sample(self.q, x_past)
        x_past_pgd = self.pgd(x_past.detach(), x_fake.detach())
        x_fake_pgd, x_fake_pgd_ec, residual_pgd = self.G.sample(self.q, x_past_pgd)

        x_fake_past = torch.cat([x_past, x_fake], dim=1)
        x_fake_past_ec = torch.cat([x_past, x_fake_ec], dim=1)
        x_fake_past_pgd = torch.cat([x_past_pgd, x_fake_pgd], dim=1)
        x_fake_past_pgd_ec = torch.cat([x_past_pgd, x_fake_pgd_ec], dim=1)

        G_loss, G_loss_adv, G_loss_res = self.g_trainstep(
            [x_fake_past, x_fake_past_ec, residual],
            [x_fake_past_pgd, x_fake_past_pgd_ec, residual_pgd],
            data.clone().to(self.device))
        self.training_loss['D_loss'].append(D_loss_fake + D_loss_fake_adv + D_loss_real)
        self.training_loss['G_loss'].append(G_loss + G_loss_adv + G_loss_res)
        self.training_loss['G_loss_adv'].append(G_loss_adv)
        self.training_loss['G_loss_res'].append(G_loss_res)
        # self.evaluate(x_fake)

        d_norm_l1, d_norm_l2, d_norm2_l1, d_norm2_l2, g_norm_l1, g_norm_l2 = (
            self.calc_gradients(x_past.detach(), x_fake.detach(), data.clone().to(self.device)))
        self.training_loss['G_norm_l1'].append(g_norm_l1)
        self.training_loss['G_norm_l2'].append(g_norm_l2)

        self.training_loss['D_norm_l1'].append(d_norm_l1)
        self.training_loss['D_norm_l2'].append(d_norm_l2)

        self.training_loss['D_norm2_l1'].append(d_norm2_l1)
        self.training_loss['D_norm2_l2'].append(d_norm2_l2)

    def g_trainstep(self, generated, generated_adv, x_real):
        toggle_grad(self.G, True)
        self.G.train()
        self.g_optimizer.zero_grad()

        x_real.requires_grad_()

        x_fake, x_fake_ec, residual = generated
        x_fake_pgd, x_fake_pgd_ec, residual_pgd = generated_adv
        d_fake = self.D(x_fake)
        d_fake_ec = self.D(x_fake_ec)
        d_fake_pgd = self.D(x_fake_pgd)
        d_fake_pgd_ec = self.D(x_fake_pgd_ec)

        gloss1 = compute_loss(d_fake, 1) + compute_loss(d_fake_ec, 1)
        gloss2 = compute_loss(d_fake_pgd, 1) + compute_loss(d_fake_pgd_ec, 1)
        gloss_res = torch.mean((residual - x_real[:, :self.p]) ** 2) + torch.mean(
            (residual_pgd - x_real[:, :self.p]) ** 2)

        if self.use_ec == 0:
            # Baseline Modle
            gloss1 = compute_loss(d_fake, 1)
            loss = gloss1
        elif self.use_ec == 1:
            # AEC-GAN w/o Aug
            gloss1 = compute_loss(d_fake, 1)
            loss = gloss1 + gloss_res
        elif self.use_ec == 2:
            # AEC-GAN
            loss = gloss1 + gloss2 + gloss_res
        else:
            raise NotImplementedError
        loss.backward()
        self.g_optimizer.step()

        return gloss1.item(), gloss2.item(), gloss_res.item()

    def d_trainstep(self, x_generated1, x_generated2, x_real):
        toggle_grad(self.D, True)
        self.D.train()
        self.d_optimizer.zero_grad()

        x_fake, x_fake_ec = x_generated1
        x_fake_past_pgd, x_fake_past_pgd_ec = x_generated2

        # On real data
        x_real.requires_grad_()
        d_real = self.D(x_real)
        dloss_real = compute_loss(d_real, 1)

        # On fake data
        x_fake.requires_grad_()
        x_fake_ec.requires_grad_()
        x_fake_past_pgd.requires_grad_()
        x_fake_past_pgd_ec.requires_grad_()

        dloss_fake = compute_loss(self.D(x_fake), 0) + compute_loss(self.D(x_fake_ec), 0)
        dloss_fake_adv = compute_loss(self.D(x_fake_past_pgd), 0) + compute_loss(self.D(x_fake_past_pgd_ec),
                                                                                           0)

        # Compute regularizer on fake/real
        if self.use_ec == 0:
            # Baseline
            dloss_fake = compute_loss(self.D(x_fake), 0)
            dloss = dloss_real + dloss_fake
        elif self.use_ec == 1:
            # AEC-GAN w/o Aug
            dloss_fake = compute_loss(self.D(x_fake), 0)
            dloss = dloss_real + dloss_fake
        else:
            # AEC-GAN
            dloss = dloss_real + dloss_fake + dloss_fake_adv

        dloss.backward()
        reg = torch.ones(1)
        # Step discriminator params
        self.d_optimizer.step()

        return dloss_real.item(), dloss_fake.item(), dloss_fake_adv.item(), reg.item()