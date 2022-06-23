import torch
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from models import ConditionalVariationalAutoEncoder


if __name__ == "__main__":
    step = 50
    x_dim = 28 * 28
    z_dim = 2
    t_class = 2
    target_input = torch.full((step,), t_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalVariationalAutoEncoder(x_dim, z_dim, 10, device)
    model.load_state_dict(torch.load("./models/checkpoint.pth"))
    model.eval()

    z11 = torch.tensor([-3, 0], dtype=torch.float)
    z12 = torch.tensor([3, 0], dtype=torch.float)
    z21 = torch.tensor([-3, 3], dtype=torch.float)
    z22 = torch.tensor([3, -3], dtype=torch.float)
    z31 = torch.tensor([0, 3], dtype=torch.float)
    z32 = torch.tensor([0, -3], dtype=torch.float)
    z41 = torch.tensor([3, 3], dtype=torch.float)
    z42 = torch.tensor([-3, -3], dtype=torch.float)

    z1_list = [z11, z21, z31, z41]
    z2_list = [z12, z22, z32, z42]

    z1_to_z2_list = []
    target_list = []

    y1_to_y2_list = []

    for z1, z2 in zip(z1_list, z2_list):
        z1_to_z2_list.append(torch.cat([((z1 * ((step - i) / step)) + (z2 * (i / step))) for i in range(step)]).reshape(step, z_dim))

    for z1_to_z2 in z1_to_z2_list:
        y1_to_y2_list.append(model.decoder(z1_to_z2, target_input).cpu().detach().numpy().reshape(-1, 28, 28))

    for n in range(len(y1_to_y2_list)):
        fig, ax = plt.subplots(1, 1, figsize=(9,9))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_xticks([])
        ax.set_yticks([])
        images = []
        for i, im in enumerate(y1_to_y2_list[n]):
            images.append([ax.imshow(im, "gray")])
        animation = ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=1000)
        animation.save(f"./figure/walk_through_{n}_{t_class}.gif", writer="pillow")