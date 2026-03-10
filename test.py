import torch, os, glob, cv2, random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
from Condition_Noise_Predictor.UNet import NoisePred

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--step_number", type=int, default=3)
parser.add_argument("--cs_ratio", type=float, default=0.5)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--testset_name", type=str, default="M3FD")
parser.add_argument("--result_dir", type=str, default="result")
args = parser.parse_args()

seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

epoch = args.epoch
T = args.step_number
B = args.block_size
ratio = args.cs_ratio
print("cs ratio =", ratio)

N = B * B
q = int(np.ceil(ratio * N))

U, S, V = torch.linalg.svd(torch.randn(N, N, device=device))
Phi = (U @ V)[:, :q]

model_channels = 64
num_res_blocks = 2
dropout = 0.1
time_embed_dim_mult = 4
down_sample_mult = [1,2,4,8]
unet = NoisePred(3, 3, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                  down_sample_mult)

model = Net(T, unet).to(device)

model_dir = "%s/models_M3FD_noise" % (args.model_dir)
model.load_state_dict(torch.load("./%s/net_params.pkl" % (model_dir)))

def load_and_process_mri_pet(mri_folder, pet_folder, index):
    # 读取图像
    mri_path = mri_folder[index]
    pet_path = pet_folder[index]

    img1 = Image.open(mri_path).convert('L')  # 'L' 表示灰度模式  # 单通道读取
    img1 = np.array(img1)
    img2 = Image.open(pet_path).convert('RGB')
    img2_ycbcr = img2.convert("YCbCr")
    img2 = np.array(img2_ycbcr)[:, :, 0]
    test_image = np.array(img2_ycbcr)
    # 转换为Tensor并自动归一化
    x1 = torch.from_numpy(img1) / 255.0
    x2 = torch.from_numpy(img2) / 255.0
    h, w = img1.shape
    h1, w1 = img2.shape
    x1 = torch.reshape(x1, [1, h, w])
    x2 = torch.reshape(x2, [1, h1, w1])

    return x1, x2, test_image

psz = 256
bsz = 1

# 使用示例
folder1_paths = sorted(glob.glob('./data/M3FD/test/ir_noisy/*.png'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
folder2_paths = sorted(glob.glob('./data/M3FD/test/vi_noisy/*.png'), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
print(folder1_paths)
print(folder2_paths)


with torch.no_grad():
    PSNR_list, SSIM_list = [], []
    result_dir = os.path.join(args.result_dir, args.testset_name)
    os.makedirs(result_dir, exist_ok=True)

    for i in range(len(folder1_paths)):
        print(i)
        x1, x2, test_image = load_and_process_mri_pet(folder1_paths, folder2_paths, i)
        x1 = x1.unsqueeze(1).to(device)
        x2 = x2.unsqueeze(1).to(device)

        # x = H(x1, random.randint(0, 7))
        # perm = torch.randperm(psz * psz, device=device)
        # perm_inv = torch.empty_like(perm)
        # perm_inv[perm] = torch.arange(perm.shape[0], device=device)
        # A = lambda z: (z.reshape(bsz, -1)[:, perm].reshape(bsz, -1, N) @ Phi)
        # AT = lambda z: (z @ Phi.t()).reshape(bsz, -1)[:, perm_inv].reshape(bsz, 1, psz, psz)
        # y = A(x)

        x_out = model(x1, x2, use_amp_=False)
        x_1, x_2, f = x_out.split(1, dim=1)
        f = (f.clamp(min=0.0, max=1.0) * 255.0).cpu().numpy().squeeze()
        # x_1 = (x_1.clamp(min=0.0, max=1.0) * 255.0).cpu().numpy().squeeze()
        test_image = cv2.resize(test_image, (1024,768), interpolation=cv2.INTER_LINEAR)
        test_image[:,:,0] = f
        test_image = cv2.cvtColor(test_image, cv2.COLOR_YCrCb2RGB).astype(np.uint8)
        # result_path = os.path.join(result_dir, path.split("/")[-1])
        cv2.imwrite("./result/%s/%05d.png" % (args.testset_name, i), test_image)
        # cv2.imwrite("%s_PSNR_%.2f_SSIM_%.4f.png" % (result_path, PSNR, SSIM), test_image)
    # print("Average PSNR: %.2f" % np.mean(PSNR_list))
    # print("Average SSIM: %.4f" % np.mean(SSIM_list))