import os
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

import common.utils.visualization as visualization_utils
from common.train.callbacks import Callback
from common.train.fit_output import FitOutput
from common.train.trainer import Trainer
from tensor_factorization.models import HierarchicalTensorFactorization


class HTWeightsPlotter(Callback):
    DEFAULT_FOLDER_NAME = "weights_plots"
    EPOCH_PLOTS_FOLDER_NAME_TEMPLATE = "epoch_{0}"
    WEIGHT_HEATMAP_PLOT_FILE_NAME_TEMPLATE = "l_{0}_f_{1}.png"
    WEIGHT_SORTED_HEATMAP_PLOT_FILE_NAME_TEMPLATE = "l_{0}_f_{1}_sort.png"

    def __init__(self, ht_model: HierarchicalTensorFactorization, output_dir: str, sort_weight_matrices_cols: bool = False,
                 num_top_cols: int = -1, folder_name: str = DEFAULT_FOLDER_NAME, create_dir: bool = True, create_plots_interval: int = 1,
                 num_saved: int = 10):
        self.ht_model = ht_model
        self.sort_weight_matrices_cols = sort_weight_matrices_cols
        self.num_top_cols = num_top_cols

        self.output_dir = output_dir
        self.folder_name = folder_name
        self.create_dir = create_dir
        self.create_plots_interval = create_plots_interval
        self.num_saved = num_saved

        self.plots_dir = os.path.join(self.output_dir, self.folder_name)
        self.epochs_saved_plots_for = []

    def on_fit_initialization(self, trainer):
        if self.create_dir and not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def on_epoch_end(self, trainer: Trainer):
        if (trainer.epoch + 1) % self.create_plots_interval == 0:
            self.__create_weights_heatmap_plots(trainer.epoch)

    def on_fit_end(self, trainer: Trainer, num_epochs_ran: int, fit_output: FitOutput):
        if self.epochs_saved_plots_for[-1] == trainer.epoch:
            return

        self.__create_weights_heatmap_plots(trainer.epoch)

    def __create_weights_heatmap_plots(self, epoch: int):
        epoch_plots_folder = os.path.join(self.plots_dir, HTWeightsPlotter.EPOCH_PLOTS_FOLDER_NAME_TEMPLATE.format(epoch))
        if not os.path.exists(epoch_plots_folder):
            os.makedirs(epoch_plots_folder)

        per_layer_per_factor_weights = self.__get_per_layer_per_factor_weights()
        for l, layer_factor_weights in enumerate(per_layer_per_factor_weights):
            for i, factor_weight_matrix in enumerate(layer_factor_weights):
                heat_plog_fig = self.__create_weight_matrix_heat_plot_fig(factor_weight_matrix, epoch=epoch, layer=l, factor_index=i)

                heat_plot_fig_filename = HTWeightsPlotter.WEIGHT_HEATMAP_PLOT_FILE_NAME_TEMPLATE.format(l, i)
                if self.sort_weight_matrices_cols:
                    heat_plot_fig_filename = HTWeightsPlotter.WEIGHT_SORTED_HEATMAP_PLOT_FILE_NAME_TEMPLATE.format(l, i)

                heat_plog_fig.savefig(os.path.join(epoch_plots_folder, heat_plot_fig_filename))

        self.epochs_saved_plots_for.append(epoch)
        if len(self.epochs_saved_plots_for) > self.num_saved:
            self.__delete_oldest_plots()

    def __get_per_layer_per_factor_weights(self):
        per_layer_per_factor_weights = []
        for layer_parameter_list in self.ht_model.per_hidden_layer_parameter_lists:
            layer_factor_weights = [parameter.data.detach() for parameter in layer_parameter_list]
            per_layer_per_factor_weights.append(layer_factor_weights)

        return per_layer_per_factor_weights

    def __create_weight_matrix_heat_plot_fig(self, weight_matrix: torch.Tensor, epoch: int, layer: int, factor_index: int):
        if self.sort_weight_matrices_cols:
            weight_matrix = self.__get_sorted_trimmed_cols_weight_matrix(weight_matrix)

        weight_matrix = weight_matrix.cpu().detach().numpy()
        fig = visualization_utils.create_garbage_collectable_figure()
        fig.set_size_inches(fig.get_size_inches() * 1.5)
        ax = fig.add_subplot(111)

        ax.imshow(weight_matrix, cmap='viridis')
        norm = mpl.colors.Normalize(vmin=weight_matrix.min(), vmax=weight_matrix.max())
        colorbar_ticks = np.linspace(weight_matrix.min(), weight_matrix.max(), num=5)
        fig.colorbar(ticks=colorbar_ticks, mappable=plt.cm.ScalarMappable(norm=norm, cmap="viridis"))
        ax.set_title(f"epoch {epoch} layer {layer} factor {factor_index}")

        return fig

    def __get_sorted_trimmed_cols_weight_matrix(self, weight_matrix):
        col_norms = weight_matrix.norm(dim=0)
        sorting_indices = torch.argsort(col_norms, descending=True)
        weight_matrix = weight_matrix[:, sorting_indices]

        if self.num_top_cols > 0:
            weight_matrix = weight_matrix[:, : self.num_top_cols]

        return weight_matrix

    def __delete_oldest_plots(self):
        oldest_epoch = self.epochs_saved_plots_for[0]
        epoch_plots_folder = os.path.join(self.plots_dir, HTWeightsPlotter.EPOCH_PLOTS_FOLDER_NAME_TEMPLATE.format(oldest_epoch))

        if os.path.exists(epoch_plots_folder):
            shutil.rmtree(epoch_plots_folder, ignore_errors=True)

        del self.epochs_saved_plots_for[0]
