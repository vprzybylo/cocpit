import torch
from cocpit import config as config
import torch.nn.functional as F


def kl_divergence(alpha):

    ones = torch.ones(
        [1, len(config.CLASS_NAMES)], dtype=torch.float32, device=config.DEVICE
    )
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    return first_term + second_term


def edl_loss(
    func,
    y,
    alpha,
    epoch_num,
    annealing_step,
):
    y = y.to(config.DEVICE)
    alpha = alpha.to(config.DEVICE)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha)
    return A + kl_div


def edl_log_loss(output, target, epoch_num, annealing_step):
    evidence = F.relu(output)
    alpha = evidence + 1
    return torch.mean(
        edl_loss(
            torch.log,
            target,
            alpha,
            epoch_num,
            annealing_step,
        )
    )


def edl_digamma_loss(output, target, epoch_num, annealing_step):
    evidence = F.relu(output)
    alpha = evidence + 1
    return torch.mean(
        edl_loss(torch.digamma, target, alpha, epoch_num, annealing_step)
    )
