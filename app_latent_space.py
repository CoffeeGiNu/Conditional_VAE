import torch
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from utils import fix_seed
from models import VariationalAutoEncoder

# References: 
# https://matplotlib.org/3.1.1/api/backend_bases_api.html#matplotlib.backend_bases.MouseEvent
# https://matplotlib.org/3.1.0/gallery/misc/cursor_demo_sgskip.html

class App(object):
    def __init__(self, model, device='cpu', show_exsamples=True, clear=True):
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 9))
        self.x, self.y = 0.0, 0.0
        self.clear = clear
        self.model = model
        self.device = device
        self.show_examples = show_exsamples
        self.xlim = None
        self.ylim = None
        self.back_im = None
        self._init_plot()
    
    def _init_plot(self):
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.ax.grid()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_press_event', self.on_release)
        if self.show_examples:
            self._show_examples_plot()
        plt.show()
    
    def _show_examples_plot(self):
        # cm = plt.get_cmap("tab10")
        # sns.set_style('ticks')

        # for data in dataset_test:
        #     image = np.array(data['image'] / 255)
        #     label = np.array(data['label'])
        #     image = torch.from_numpy(np.array(image))

        #     _, z, _ = model(image)
        #     z = z.detach().numpy()

        #     for k in range(10):
        #         cluster_indexes = np.where(label == k)[0]
        #         plt.scatter(z[cluster_indexes, 0], z[cluster_indexes, 1], marker=f"${k}$", color=cm(k), label=f"{k}")
        #     break
        if not self.xlim:
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()
        if not self.back_im:
            self.back_im = Image.open("./figure/background.png")
        self.ax.imshow(self.back_im, extent=[*self.xlim, *self.ylim], alpha=0.8)
    
    def reconst_img(self):
        z = torch.tensor([self.x, self.y], dtype=torch.float).to(self.device)
        reconst = self.model.decoder(z).cpu().detach().numpy().reshape(28, 28)
        self.ax.imshow(reconst*255, "gray", extent=[2,4,2,4], alpha=1)

    def draw(self):
        # Clear
        if self.clear:
            self.ax.clear()
            self.ax.grid()
        if self.show_examples:
            self._show_examples_plot()
        
        # Your plottings
        self.ax.plot(self.x, self.y, "*", color="c", markersize=12)
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.reconst_img()
        
        # update
        self.ax.figure.canvas.draw()
    
    def on_press(self, event):
        if not event.inaxes:
            return
        print(f'on_press(): {event.button}, {event.xdata:1.2f}, {event.ydata:1.2f}', )
    
    def on_release(self, event):
        if not event.button == 1: return
        # Your events
        self.x, self.y = event.xdata, event.ydata
        self.draw()
        
    def on_key(self, event):
        # Your events
        if event.key == "left":
            self.x -= 0.15
            self.draw()
        elif event.key == "right":
            self.x += 0.15
            self.draw()
        elif event.key == "down":
            self.y -= 0.15
            self.draw()
        elif event.key == "up":
            self.y += 0.15
            self.draw()
        elif event.key == "c":
            self.ax.clear()
            self.ax.set_xlim(-4, 4)
            self.ax.set_ylim(-4, 4)
            self.ax.grid()
            if self.show_examples:
                self._show_examples_plot()
            self.ax.figure.canvas.draw() 
        print(f'on_key(): {event.key}, {self.x:1.2f}, {self.y:1.2f}', )


if __name__ == "__main__":
    log_dir = "./logs"
    seed = 42
    fix_seed(seed)
    x_dim = 28 * 28
    z_dim = 2
    batch_size = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # _, _, dataset_test = load_tfds("mnist", batch_size=batch_size, seed=seed, preprocess_fn=None)

    model = VariationalAutoEncoder(x_dim, z_dim, device)
    model.load_state_dict(torch.load("./models/checkpoint.pth"))
    model.eval()
    for param in model.parameters():
        param.grad = None
    model.to(device)

    cm = plt.get_cmap("tab10")
    sns.set_style('ticks')

    app = App(model, device)
