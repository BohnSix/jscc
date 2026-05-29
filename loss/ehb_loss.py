import torch
import torch.nn.functional as F


def redundancy_cosine_loss(u_list, eps: float = 1e-8):
    """Pairwise cosine redundancy penalty between expert outputs.

    Encourages U1~U4 to carry complementary information.
    """
    pair_losses = []
    pair_names = []
    for i in range(len(u_list)):
        for j in range(i + 1, len(u_list)):
            f1 = F.normalize(u_list[i].flatten(1), p=2, dim=1, eps=eps)
            f2 = F.normalize(u_list[j].flatten(1), p=2, dim=1, eps=eps)
            sim = (f1 * f2).sum(dim=1)
            loss_ij = (sim ** 2).mean()
            pair_losses.append(loss_ij)
            pair_names.append(f"red_u{i+1}_u{j+1}")
    loss = torch.stack(pair_losses).mean()
    stats = {name: val.detach().item() for name, val in zip(pair_names, pair_losses)}
    return loss, stats


def incremental_gain_loss(distortions, gammas=None):
    """Hinge loss enforcing progressive reconstruction improvement.

    distortions: list of scalar distortion values [d1, d2, ..., dN]
    gammas: list of margin values (length N-1), default all zeros
    """
    n = len(distortions)
    if gammas is None:
        gammas = [0.0] * (n - 1)
    losses = []
    stats = {}
    for i in range(n):
        stats[f"d{i+1}"] = distortions[i].detach().item()
    for i in range(n - 1):
        gain = distortions[i] - distortions[i + 1]
        loss_i = torch.relu(distortions[i + 1] - distortions[i] + gammas[i])
        losses.append(loss_i)
        stats[f"gain{i+1}{i+2}"] = gain.detach().item()
        stats[f"inc_loss{i+2}"] = loss_i.detach().item()
    inc_loss = sum(losses)
    return inc_loss, stats
