import torch

from utils import fix_seed
from models import VariationalAutoEncoder


log_dir = "./logs"
seed = 42
fix_seed(seed)
x_dim = 28 * 28
z_dim = 10
batch_size = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VariationalAutoEncoder(x_dim, z_dim, device)
model.load_state_dict(torch.load("./models/checkpoint_z10.pth"))
model.eval()

dummy_input = torch.randn(1, 1, 784, device="cpu")
torch.onnx.export(model, dummy_input, "./models/model.onnx", input_names=["input"], output_names=["KLD", "RC", "latent", "output"], verbose=True)