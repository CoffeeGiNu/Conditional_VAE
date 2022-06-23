import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import fix_seed
from models import ConditionalVariationalAutoEncoder
from datasets import load_tfds


if __name__ == "__main__":
    log_dir = "./logs"
    seed = 42
    fix_seed(seed)
    x_dim = 28 * 28
    z_dim = 2
    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, dataset_test = load_tfds("mnist", batch_size=batch_size, seed=seed, preprocess_fn=None)
    
    model = ConditionalVariationalAutoEncoder(x_dim, z_dim, 10, device)
    model.load_state_dict(torch.load("./models/checkpoint.pth"))
    model.eval()

    for data in dataset_test:
        image = np.array(data['image'] / 255)
        label = np.array(data['label'])
        image = torch.from_numpy(np.array(image))
        label = torch.from_numpy(np.array(label))
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        # for a in axes:
        #     a.set_xticks([])
        #     a.set_yticks([])

        _, _, y = model(image, label)
        image_x = image.cpu().detach().numpy().reshape(-1, 28, 28)
        image_y = y.cpu().detach().numpy().reshape(-1, 28, 28)
        for j in range(batch_size):
            axes[0][j].imshow(image_x[j], "gray")
            axes[0][j].set_xticks([])
            axes[0][j].set_yticks([])
            axes[1][j].imshow(image_y[j], "gray")
            axes[1][j].set_xticks([])
            axes[1][j].set_yticks([])
        fig.savefig(f"./figure/reconstracttion_z{z_dim}.png")
        plt.close()
        break