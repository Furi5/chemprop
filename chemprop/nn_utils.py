from PIL import Image
import math
import os
from typing import List, Union, Tuple
import io
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
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm
from rdkit.Chem.Draw import rdMolDraw2D
import svgutils.compose as sc
from chemprop.Process_runner import ProcessRunner
from rdkit.Chem import Draw
from PIL import Image
import cairosvg


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
    # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    final_size = index_size + suffix_dim

    # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = source.index_select(dim=0, index=index.view(-1))
    # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target = target.view(final_size)

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
            len(optimizer.param_groups) == len(
                warmup_epochs) == len(total_epochs)
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
        self.warmup_steps = (self.warmup_epochs *
                             self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (
            self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (
            self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

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
                self.lr[i] = self.init_lr[i] + \
                    self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * \
                    (self.exponential_gamma[i] **
                     (self.current_step - self.warmup_steps[i]))
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

            a1, a2 = mol_graph.b2a[b].item(
            ) - a_start, mol_graph.b2a[mol_graph.b2revb[b]].item() - a_start

            b_weights = attention_weights[b]
            a2b = mol_graph.a2b[a_start:a_start + a_size]
            a_weights = index_select_ND(b_weights, a2b)
            a_weights = a_weights.sum(dim=1)
            a_weights = a_weights.cpu().data.numpy()
            atomSum_weights += a_weights
        Amean_weight = atomSum_weights / a_size
        nanMean = np.nanmean(Amean_weight)
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                         Amean_weight - nanMean,
                                                         colorMap=matplotlib.cm.bwr)

        save_path = os.path.join(
            smiles_viz_dir, f'bond_{b - b_start}_depth_{depth}.png')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)


def attention_tensor_np(
        num_atoms: int,
        attention_weights: torch.FloatTensor):
    atomSum_weights = np.zeros(num_atoms)
    for a in range(num_atoms):
        a_weights = attention_weights[a].cpu().data.numpy()
        atomSum_weights += a_weights
    Amean_weight = atomSum_weights/num_atoms

    nanMean = np.nanmean(Amean_weight)
    attention_weights = Amean_weight - nanMean
    return Amean_weight - nanMean


# def visualize_atom_attention(
#         smiles: str,
#         tox_name: str,
#         attention_weights: np.ndarray):
#     """
#     Saves figures of attention maps between atoms. Note: works on a single molecule, not in batch

#     :param viz_dir: Directory in which to save attention map figures.
#     :param smiles: Smiles string for molecule.
#     :param num_atoms: The number of atoms in this molecule.
#     :param attention_weights: A num_atoms x num_atoms PyTorch FloatTensor containing attention weights.
#     """
#     viz_dir = 'output.svg'
#     with ProcessRunner() as P:

#         mol = Chem.MolFromSmiles(smiles)
#         max_value = np.max(attention_weights)
#         norm = Normalize(vmin=-max_value, vmax=max_value)
#         PiYG_cmap = cm.get_cmap('PiYG_r', 2)
#         colorMap = LinearSegmentedColormap.from_list(
#             'PiYG_r', [PiYG_cmap(0), (1.0, 1.0, 1.0), PiYG_cmap(1)], N=255)
#         plt_colors = cm.ScalarMappable(norm=norm, cmap=colorMap)
#         fig = SimilarityMaps.GetSimilarityMapFromWeights(mol=mol,
#                                                          weights=attention_weights,
#                                                          colorMap=colorMap,
#                                                          size=(250, 250),
#                                                          norm=norm
#                                                          )
#         cbar = plt.colorbar(plt_colors, orientation='horizontal', pad=0.01)
#         cbar.set_label('Attention Weights')
#         ticks = np.linspace(-max_value, max_value, 10)
#         tick_values = np.round(np.linspace(-max_value, max_value, 10), 2)
#         cbar.set_ticks(ticks)  # Set the color bar ticks
#         cbar.set_ticklabels(tick_values)  # Set the color bar tick labels
#         fig.savefig(viz_dir, bbox_inches='tight')
#         plt.close(fig)
#         with open(viz_dir, 'rb') as f:
#             output = f.read()
#         return output

#       # # 添加色条
#       # max_value = max(abs(np.array(attention_weights)))
#       # ticks = np.linspace(-max_value, max_value, 10)
#       # tick_values = np.round(np.linspace(-max_value, max_value, 10), 4)

#       # cbar = plt.colorbar(plt_colors,
#       #                     orientation='horizontal',
#       #                     ticks=ticks,
#       #                     label='Attention Weights')
#       # cbar.set_ticklabels(tick_values)
#       # cbar.set_label('Attention Score', fontsize=13, labelpad=-60, y=0,)
#       # cbar.text(-0.01, 0.5, 'Non-toxicity', transform=cbar.ax.transAxes,
#       #           fontsize=13, verticalalignment='center', horizontalalignment='right')
#       # cbar.text(1.01, 0.5, 'Toxicity', transform=cbar.ax.transAxes,
#       #           fontsize=13, verticalalignment='center', horizontalalignment='left')

#       # # 添加标题和副标题
#       # title = f"\nAttention visualization of {tox_name}"
#       # subtitle = "\n".join([
#       #     "This visualization depicts the attention weights",
#       #     f"of each atom in a molecule towards the {tox_name} endpoint.",
#       #     "Purple indicates toxic substructures, green indicates non-toxic substructures.",
#       #     "Denser Solid contour lines indicate higher weights in toxic substructures,",
#       #     "Denser dashed contour lines indicate higher weights in non-toxic substructures."
#       # ])
#       # fig.text(0.01, 0.96, title, fontsize=20,
#       #          weight="bold", ha="left", va="baseline")
#       # fig.text(0.01, 0.94, subtitle, fontsize=14, ha="left", va="top")

#       # # 添加图例，并设置位置为左上角
#       # solid_line = plt.Line2D([], [], color='black', linestyle='-',
#       #                         label='Contour Line with Toxicity', linewidth=2)
#       # dashed_line = plt.Line2D([], [], color='black', linestyle='--',
#       #                          label='Contour Line with Non-toxicity', linewidth=2)
#       # fig.legend(handles=[solid_line, dashed_line], bbox_to_anchor=(
#       #     -0.01, 0.795), loc='upper left', frameon=False, fontsize=14)

def mol_attention(mol, attention_weights):
    d = Draw.MolDraw2DCairo(1000, 800)
    PiYG_cmap = cm.get_cmap('PRGn_r', 2)
    colorMap = LinearSegmentedColormap.from_list(
        'PRGn', [PiYG_cmap(0), (1.0, 1.0, 1.0), PiYG_cmap(1)], N=1000)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, list(attention_weights), draw2d=d, colorMap=colorMap)
    d.FinishDrawing()
    with open('mol.png', 'wb') as f:
        f.write(d.GetDrawingText())


def plot_colorbar(attention_weights: np.ndarray) -> None:
    desired_width_px = 1000
    desired_height_px = 800

    # Set the DPI (dots per inch)
    dpi = 150  # You can adjust the DPI to your preference

    # Calculate the figure size in inches
    fig_width_inch = desired_width_px / dpi
    fig_height_inch = desired_height_px / dpi
    max_value = np.max(np.abs(attention_weights))
    min_value = np.min(attention_weights)
    norm = Normalize(vmin=-min_value, vmax=max_value)
    PiYG_cmap = cm.get_cmap('PRGn_r', 2)
    colorMap = LinearSegmentedColormap.from_list(
        'PRGn', [PiYG_cmap(0), (1.0, 1.0, 1.0), PiYG_cmap(1)], N=1000)

    bounds = np.linspace(-max_value, max_value, 21)
    ticks = np.linspace(-max_value, max_value, 21)
    # bounds = ticks
    norm = BoundaryNorm(bounds, colorMap.N)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=colorMap)

    tick_values = np.round(np.linspace(-max_value, max_value, 21), 4)
    tick_labels = [f'{val:.4f}' if i %
                   2 == 0 else '' for i, val in enumerate(tick_values)]
    # tick_labels[-1] = f'{max_value:.4f}'

    # Create a new figure for the colorbar
    fig, cbar_ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
    cbar_ax.set_position([0.5, 0.5, 0.9, 0.05])
    cbar = plt.colorbar(plt_colors, cax=cbar_ax,
                        orientation='horizontal', boundaries=bounds, ticks=ticks)
    cbar.ax.set_xticklabels(tick_labels)
    cbar.set_label('Attention Score', fontsize=13, labelpad=-60, y=0,)
    cbar.ax.text(-0.01, 0.5, 'Non-toxicity', transform=cbar.ax.transAxes,
                 fontsize=13, verticalalignment='center', horizontalalignment='right')
    cbar.ax.text(1.01, 0.5, 'Toxicity', transform=cbar.ax.transAxes,
                 fontsize=13, verticalalignment='center', horizontalalignment='left')

    fig.savefig("colorbar.png", format='png',
                pad_inches=-0.01,
                bbox_inches='tight', transparent=False)
    plt.close()


def plot_node(tox_name):
    desired_width_px = 1000
    desired_height_px = 800
    dpi = 157

    # Calculate the figure size in inches
    fig_width_inch = desired_width_px / dpi
    fig_height_inch = desired_height_px / dpi

    # Create a new figure
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch), dpi=dpi)

    # Title and subtitle
    title = f"\nAttention visualization of {tox_name}"
    subtitle = "\n".join([
        "This picture shows the attention weights of each \n"
        f"atom in a molecule for the {tox_name} endpoint.\n\n"
        "Purple indicates toxic substructures, green indicates non-toxic substructures.",
        "Denser Solid contour lines indicate higher weights in toxic substructures,",
        "Denser dashed contour lines indicate higher weights in non-toxic substructures."
    ])

    fig.text(0.01, 1, title, fontsize=16,
             weight="bold", ha="left", va="baseline")
    fig.text(0.01, 0.96, subtitle, fontsize=11, ha="left", va="top")

    # Legend
    solid_line = plt.Line2D([], [], color='black', linestyle='-',
                            label='Contour Line with Toxicity', linewidth=1)
    dashed_line = plt.Line2D([], [], color='black', linestyle='--',
                             label='Contour Line with Non-toxicity', linewidth=1)
    fig.legend(handles=[solid_line, dashed_line], bbox_to_anchor=(-0.01,
               0.74), loc='upper left', frameon=False, fontsize=11)

    # Example plot (replace this with your actual plot)
    # Save the figure as a PNG file
    fig.savefig("node.png", format='png',
                bbox_inches='tight', transparent=False)
    plt.close()


def concatenate_images(image_path1='mol.png', image_path2='colorbar.png', image_path3='node.png', output_path='attention.png'):
    """
    Concatenate two images.

    :param image_path1: Path to the first image
    :param image_path2: Path to the second image
    :param output_path: Path to save the concatenated image
    :param direction: Direction to concatenate ('horizontal' or 'vertical')
    """
    # Open the images
    img1 = Image.open(image_path1)
    top_px_to_remove = int(120)
    box = (0, top_px_to_remove, img1.width, img1.height)
    img1 = img1.crop(box)
    # Crop the image

    img2 = Image.open(image_path2)
    img3 = Image.open(image_path3)
    # Get dimensions
    new_img = Image.new('RGBA', (img1.width, img1.height+img3.height))
    new_img.paste(img3, (0, 0))
    new_img.paste(img1, (0, img3.height))
    new_img.paste(img2, (100, img3.height+img1.height-100))
    # Save the new image
    new_img.save(output_path)


def visualize_atom_attention(smiles, tox_name, attention_weights):
    with ProcessRunner() as P:
        mol = Chem.MolFromSmiles(smiles)
        mol_attention(mol, attention_weights)
        plot_colorbar(attention_weights)
        plot_node(tox_name)
        concatenate_images()
        with open('attention.png', 'rb') as f:
            return f.read()
