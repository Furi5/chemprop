import math
import os
from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from tqdm import trange

import matplotlib
import matplotlib.pyplot as plt


def compute_pnorm(model: nn.Module) -> float:
    """
    Computes the norm of the parameters of a model.

    :param model: A PyTorch model.
    :return: The norm of the parameters of the model.
    """
    return math.sqrt(sum([p.norm().item() ** 2 for p in model.parameters()]))


def compute_gnorm(model: nn.Module) -> float:
    """
    Computes the norm of the gradients of a model.

    :param model: A PyTorch model.
    :return: The norm of the gradients of the model.
    """
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None]))


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def param_count_all(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An PyTorch model.
    :return: The number of trainable parameters in the model.
    """
    return sum(param.numel() for param in model.parameters())


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.

    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.

    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if not (
            len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs)
            == len(init_lr) == len(max_lr) == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates! "
                f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
                f"len(warmup_epochs)= {len(warmup_epochs)}, "
                f"len(total_epochs)= {len(total_epochs)}, "
                f"len(init_lr)= {len(init_lr)}, "
                f"len(max_lr)= {len(max_lr)}, "
                f"len(final_lr)= {len(final_lr)}"
            )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def activate_dropout(module: nn.Module, dropout_prob: float):
    """
    Set p of dropout layers and set to train mode during inference for uncertainty estimation.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param dropout_prob: A float on (0,1) indicating the dropout probability.
    """
    if isinstance(module, nn.Dropout):
        module.p = dropout_prob
        module.train()


def visualize_bond_attention(viz_dir: str,
                             mol_graph: None,
                             attention_weights: torch.FloatTensor,
                             depth: int):
    """
    Saves figures of attention maps between bonds.

    :param viz_dir: Directory in which to save attention map figures.
    :param mol_graph: BatchMolGraph containing a batch of molecular graphs.
    :param attention_weights: A num_bonds x num_bonds PyTorch FloatTensor containing attention weights.
    :param depth: The current depth (i.e. message passing step).
    """
    for i in trange(mol_graph.n_mols):
        smiles = mol_graph.smiles_batch[i]
        mol = Chem.MolFromSmiles(smiles)

        smiles_viz_dir = os.path.join(viz_dir, smiles)
        os.makedirs(smiles_viz_dir, exist_ok=True)

        a_start, a_size = mol_graph.a_scope[i]
        b_start, b_size = mol_graph.b_scope[i]
        atomSum_weights = np.zeros(a_size)
        for b in trange(b_start, b_start + b_size):

            a1, a2 = mol_graph.b2a[b].item() - a_start, mol_graph.b2a[mol_graph.b2revb[b]].item() - a_start


            b_weights = attention_weights[b]
            a2b = mol_graph.a2b[a_start:a_start + a_size]
            a_weights = index_select_ND(b_weights, a2b)
            a_weights = a_weights.sum(dim=1)
            a_weights = a_weights.cpu().data.numpy()
            atomSum_weights += a_weights
        Amean_weight = atomSum_weights / a_size
        nanMean=np.nanmean(Amean_weight)
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                     Amean_weight - nanMean,
                                                     colorMap=matplotlib.cm.bwr)


        save_path = os.path.join(smiles_viz_dir, f'bond_{b - b_start}_depth_{depth}.png')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)


def visualize_atom_attention(viz_dir: str,
                             mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
                             num_atoms: int,
                             attention_weights: torch.FloatTensor):
    """
    Saves figures of attention maps between atoms. Note: works on a single molecule, not in batch

    :param viz_dir: Directory in which to save attention map figures.
    :param smiles: Smiles string for molecule.
    :param num_atoms: The number of atoms in this molecule.
    :param attention_weights: A num_atoms x num_atoms PyTorch FloatTensor containing attention weights.
    """
    # mol = Chem.MolFromSmiles(smiles)

    smiles_viz_dir = os.path.join(viz_dir, f'{Chem.inchi.MolToInchiKey(mol)}_{num_atoms}')
    os.makedirs(smiles_viz_dir, exist_ok=True)
    atomSum_weights=np.zeros(num_atoms)
    for a in range(num_atoms):
        a_weights = attention_weights[a].cpu().data.numpy()
        atomSum_weights+=a_weights
    Amean_weight=atomSum_weights/num_atoms

    nanMean=np.nanmean(Amean_weight)

    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                         Amean_weight-nanMean,
                                                         colorMap=matplotlib.cm.bwr)
    save_path = os.path.join(smiles_viz_dir, f'atom_{a}.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)