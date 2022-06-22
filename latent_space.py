import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import fix_seed
from models import ConditionalVariationalAutoEncoder
from datasets import load_tfds


if __name__ == "__main__":
    log_dir = "./logs"
    seed = 42
    fix_seed(seed)
    x_dim = 28 * 28
    z_dim = 2
    batch_size = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, dataset_test = load_tfds("mnist", batch_size=batch_size, seed=seed, preprocess_fn=None)

    model = ConditionalVariationalAutoEncoder(x_dim, z_dim, 10, device)
    model.load_state_dict(torch.load("./models/checkpoint.pth"))
    model.eval()

    cm = plt.get_cmap("tab10")
    sns.set_style('ticks')

    for num_batch, data in enumerate(dataset_test):
        image = np.array(data['image'] / 255)
        label = np.array(data['label'])
        image = torch.from_numpy(np.array(image))
        label = torch.from_numpy(np.array(label))
        fig_plot, ax_plot = plt.subplots(figsize=(9, 9))
        fig_scatter, ax_scatter = plt.subplots(figsize=(9, 9))

        _, z, _ = model(image, label)
        z = z.detach().numpy()

        for k in range(10):
            cluster_indexes = np.where(label == k)[0]
            ax_plot.plot(z[cluster_indexes, 0], z[cluster_indexes, 1], "o", ms=4, color=cm(k), label=f"{k}")
            ax_scatter.scatter(z[cluster_indexes, 0], z[cluster_indexes, 1], marker=f"${k}$", color=cm(k), label=f"{k}")
        ax_plot.spines['right'].set_visible(True)
        fig_plot.legend(loc=(0.91, 0.74))
        fig_scatter.legend(loc=(0.91, 0.74))
        fig_plot.tight_layout()
        fig_scatter.tight_layout()
        fig_plot.savefig(f"./figure/latent_space_z_{z_dim}_{num_batch}_plot.png")
        fig_scatter.savefig(f"./figure/latent_space_z_{z_dim}_{num_batch}_scatter.png")
        plt.close(fig_plot)
        plt.close(fig_scatter)
        break
    
    l = 25
    x = np.linspace(-2, 2, l)
    y = np.linspace(-2, 2, l)
    t_class = 8
    target_input = torch.full((l*l,), t_class)
    z_x, z_y = np.meshgrid(x, y)
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.plot(z_x, z_y, "o", ms=4, color="k")
    fig.savefig("./figure/lattice_point.png")

    Z = torch.tensor(np.array([z_x, z_y]), dtype=torch.float).permute(1,2,0) 
    Z = Z.reshape((-1, 2))
    y = model.decoder(Z, target_input).cpu().detach().numpy().reshape(-1, 28, 28) 
    fig, axes = plt.subplots(l, l, figsize=(9, 9))

    for i in range(l):
        for j in range(l):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            axes[i][j].imshow(y[l * (l - 1 - i) + j], "gray")
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("./figure/from_lattice_point.png")
    
    # for i in range(0, 10):
    #     l = 25
    #     x = np.linspace(-2, 2, l)
    #     y = np.linspace(-2, 2, l)
    #     t_class = i
    #     target_input = torch.full((l*l,), t_class)
    #     z_x, z_y = np.meshgrid(x, y)
    #     fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    #     ax.plot(z_x, z_y, "o", ms=4, color="k")
    #     fig.savefig("./figure/lattice_point.png")

    #     Z = torch.tensor(np.array([z_x, z_y]), dtype=torch.float).permute(1,2,0) 
    #     Z = Z.reshape((-1, 2))
    #     y = model.decoder(Z, target_input).cpu().detach().numpy().reshape(-1, 28, 28) 
    #     fig, axes = plt.subplots(l, l, figsize=(9, 9))

    #     for i in range(l):
    #         for j in range(l):
    #             axes[i][j].set_xticks([])
    #             axes[i][j].set_yticks([])
    #             axes[i][j].imshow(y[l * (l - 1 - i) + j], "gray")
    #     fig.subplots_adjust(wspace=0, hspace=0)
    #     fig.savefig("./figure/from_lattice_point.png")
    #     fig.savefig(f"./figure/background_{t_class}.png")