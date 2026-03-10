import torch, os, glob, cv2, random
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm
from PIL import Image
import clip
import pytorch_ssim
from Condition_Noise_Predictor.UNet import NoisePred

# dist.init_process_group(backend="nccl", init_method="env://")
# rank = dist.get_rank()
rank = 1

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--step_number", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--cs_ratio", type=float, default=0.5)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--testset_name", type=str, default="Set11")

args = parser.parse_args()

seed = 2025 + rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

epoch = args.epoch
learning_rate = args.learning_rate
T = args.step_number
B = args.block_size
bsz = args.batch_size
psz = args.patch_size
ratio = args.cs_ratio

if rank == 0:
    print("cs ratio =", ratio)
    print("batch size per gpu =", bsz)
    print("patch size =", psz)

N = B * B
q = int(np.ceil(ratio * N))

U, S, V = torch.linalg.svd(torch.randn(N, N, device=device))
Phi = (U @ V)[:, :q]

# print("reading files...")
# start_time = time()
# training_image_paths = glob.glob(os.path.join(args.data_dir, "pristine_images") + "/*")
# print("training_image_num", len(training_image_paths), "read time", time() - start_time)

# 文本输入
# text_line = ["Denoise the T1ce modality before image fusion"]
# # 加载 CLIP 模型并放到 GPU（或 CPU）
# clip_model, _ = clip.load("ViT-B/32", device=device)
# clip_model = clip_model.float()
# # 冻结图像编码器参数（只训练文本部分）
# for p in clip_model.visual.parameters():
#     p.requires_grad = False
# # 将文本转为 token 并移动到同一设备
# text_tokens = clip.tokenize(text_line).to(device)  # 不要加 .float()
# print(clip_model.dtype)

# from diffusers import StableDiffusionPipeline
# pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5").to(device)
# from diffusers import UNet2DConditionModel
# 只取 UNet，16 位权重
# unet = UNet2DConditionModel.from_pretrained(
#     "sd-legacy/stable-diffusion-v1-5",
#     subfolder="unet",
#     torch_dtype=torch.float32
# ).to(device)
# unet = DiffusionUNet(
#         in_channels=4, out_channels=4,
#         base_ch=64, ch_mults=(2,2,8, 8), num_res_blocks=2,
#         attn_resolutions={4}, time_dim=256, dropout=0.0
#     ).to(device)
model_channels = 64
num_res_blocks = 2
dropout = 0.1
time_embed_dim_mult = 4
down_sample_mult = [1,2,4,8]
unet = NoisePred(3, 3, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                  down_sample_mult)

model =Net(T, unet).to(device)
# model = DDP(model, device_ids=[rank])
# model._set_static_graph()

model_dir = "%s/models" % (args.model_dir)
model.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (model_dir, 30)))


# text_line = []
# text_line.append("This work performs PET-MRI image fusion to integrate the functional information from PET with the anatomical details provided by MRI.")
# text = clip.tokenize(text_line).float().to(device)

if rank == 0:
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param.", param_cnt/1e6, "M")

class FusionDataset(Dataset):
    def __init__(self, folder1_paths, folder2_paths, folder3_paths, folder4_paths, psz=64, iter_num=1000, bsz=16):
        self.folder1_paths = folder1_paths  # 第一个文件夹的图像路径列表
        self.folder2_paths = folder2_paths  # 第二个文件夹的图像路径列表
        self.folder3_paths = folder3_paths
        self.folder4_paths = folder4_paths
        self.psz = psz  # 裁剪的patch大小
        self.iter_num = iter_num  # 迭代次数
        self.bsz = bsz  # batch size

        # 确保两个文件夹的图像数量相同且一一对应
        assert len(self.folder1_paths) == len(self.folder2_paths), \
            "两个文件夹的图像数量必须相同"

    def __getitem__(self, index):
        while True:
            # 获取对应的图像对
            idx = index % len(self.folder1_paths)  # 防止index超出范围
            path1 = self.folder1_paths[idx]
            path2 = self.folder2_paths[idx]
            path3 = self.folder3_paths[idx]
            path4 = self.folder4_paths[idx]

            # 读取图像并转换为YCrCb（假设需要Y通道）
            img1 = Image.open(path1).convert('L')  # 'L' 表示灰度模式  # 单通道读取
            img1 = np.array(img1)
            img2 = Image.open(path2).convert('RGB')
            img2_ycbcr = img2.convert("YCbCr")
            img2 = np.array(img2_ycbcr)[:, :, 0]
            img3 = Image.open(path3).convert('L')  # 'L' 表示灰度模式  # 单通道读取
            img3 = np.array(img3)
            img4 = Image.open(path4).convert('RGB')
            img4_ycbcr = img4.convert("YCbCr")
            img4 = np.array(img4_ycbcr)[:, :, 0]

            # 转换为Tensor并自动归一化
            x1 = torch.from_numpy(img1) / 255.0
            x2 = torch.from_numpy(img2) / 255.0
            x3 = torch.from_numpy(img3) / 255.0
            x4 = torch.from_numpy(img4) / 255.0

            # 检查图像尺寸是否足够大
            h1, w1 = x1.shape
            h2, w2 = x2.shape
            max_h = min(h1, h2) - self.psz
            max_w = min(w1, w2) - self.psz

            if max_h < 0 or max_w < 0:
                continue  # 如果图像太小，跳过这对图像

            # 随机裁剪相同位置的patch
            start_h = random.randint(0, max_h)
            start_w = random.randint(0, max_w)

            patch1 = x1[start_h:start_h + self.psz, start_w:start_w + self.psz]
            patch2 = x2[start_h:start_h + self.psz, start_w:start_w + self.psz]
            patch3 = x3[start_h:start_h + self.psz, start_w:start_w + self.psz]
            patch4 = x4[start_h:start_h + self.psz, start_w:start_w + self.psz]

            return patch1, patch2, patch3, patch4 #x1, x2, x3, x4

    def __len__(self):
        return len(folder1_paths)
iter_num = 1000

class FusionMaxLoss(nn.Module):
    """
    融合损失 = w_int * || F - max(I, V) ||_1  +  w_grad * || G(F) - max(G(I), G(V)) ||_1
    其中 G(.) 为梯度幅值（默认 |Gx|+|Gy|，可选 sqrt(Gx^2+Gy^2)）。
    适配输入形状: (B, C, H, W)，若 C=3 会先转灰度再计算梯度与像素最大值。
    """
    def __init__(self, grad_mode: str = "l1", reduction: str = "mean", eps: float = 1e-12):
        super().__init__()
        assert grad_mode in ("l1", "l2"), "grad_mode 必须是 'l1' 或 'l2'"
        assert reduction in ("mean", "sum", "none"), "reduction 必须是 'mean'/'sum'/'none'"
        self.grad_mode = grad_mode
        self.reduction = reduction
        self.eps = eps

        # Sobel 核 (1,1,3,3)
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.],
                           [ 0.,  0.,  0.],
                           [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_kx", kx)
        self.register_buffer("sobel_ky", ky)

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) → (B,1,H,W)
        if x.dim() != 4:
            raise ValueError(f"期望 4D 张量 (B,C,H,W)，收到 {x.shape}")
        if x.size(1) == 1:
            return x
        # 按常用亮度加权将 RGB 转灰度；若 C>3，用均值近似
        if x.size(1) >= 3:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            return y
        else:
            return x.mean(dim=1, keepdim=True)

    def _grad_mag(self, x_gray: torch.Tensor) -> torch.Tensor:
        # 计算梯度幅值：|Gx|+|Gy| 或 sqrt(Gx^2+Gy^2)
        gx = F.conv2d(x_gray, self.sobel_kx, padding=1)
        gy = F.conv2d(x_gray, self.sobel_ky, padding=1)
        if self.grad_mode == "l1":
            g = torch.abs(gx) + torch.abs(gy)
        else:  # "l2"
            g = torch.sqrt(gx * gx + gy * gy + self.eps)
        return g

    def forward(self, Fused: torch.Tensor, Infra: torch.Tensor, Visible: torch.Tensor) -> torch.Tensor:
        if Fused.shape != Infra.shape or Fused.shape != Visible.shape:
            raise ValueError(f"Fused/Infra/Visible 形状需一致，收到 {Fused.shape}, {Infra.shape}, {Visible.shape}")

        # 像素域目标：逐像素最大值
        I_gray = self._to_gray(Infra)
        V_gray = self._to_gray(Visible)
        F_gray = self._to_gray(Fused)

        pixel_target = torch.maximum(I_gray, V_gray)          # (B,1,H,W)
        loss_int = torch.abs(F_gray - pixel_target)

        # 梯度域目标：梯度幅值的逐像素最大值
        gF = self._grad_mag(F_gray)
        gI = self._grad_mag(I_gray)
        gV = self._grad_mag(V_gray)
        grad_target = torch.maximum(gI, gV)
        loss_grad = torch.abs(gF - grad_target)

        # 组合

        if self.reduction == "mean":
            return loss_int.mean(), loss_grad.mean()
        elif self.reduction == "sum":
            return loss_int.sum(), loss_grad.sum()
        else:
            return loss_int, loss_grad  # (B,1,H,W)

criterion = FusionMaxLoss(grad_mode="l1", reduction="mean").to(device)

# 使用示例
folder1_paths = sorted(glob.glob('./data/M3FD/train/ir_noisy/*.png'))  # 第一个文件夹图像路径
folder2_paths = sorted(glob.glob('./data/M3FD/train/vi_noisy/*.png'))
folder3_paths = sorted(glob.glob('./data/M3FD/train/ir/*.png'))
folder4_paths = sorted(glob.glob('./data/M3FD/train/vi/*.png'))
dataset = FusionDataset(folder1_paths, folder2_paths, folder3_paths, folder4_paths, psz=psz, iter_num=1000, bsz=args.batch_size)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.5)
scaler = torch.cuda.amp.GradScaler()

model_dir = "./%s/models" % (args.model_dir)
log_path = "./%s/R_%.2f_T_%d_B_%d.txt" % (args.log_dir, ratio, T, B)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name, "*"))

print("start training...")
for epoch_i in range(0, epoch + 1):
    start_time = time()
    loss_avg = 0.0
    # dist.barrier()
    for x1, x2, x3, x4 in tqdm(dataloader):
        x1 = x1.unsqueeze(1).to(device)
        x2 = x2.unsqueeze(1).to(device)
        x3 = x3.unsqueeze(1).to(device)
        x4 = x4.unsqueeze(1).to(device)

        # x = H(x1, random.randint(0, 7))
        # perm = torch.randperm(psz * psz, device=device)
        # perm_inv = torch.empty_like(perm)
        # perm_inv[perm] = torch.arange(perm.shape[0], device=device)
        # A = lambda z: (z.reshape(bsz, -1)[:, perm].reshape(bsz, -1, N) @ Phi)
        # AT = lambda z: (z @ Phi.t()).reshape(bsz, -1)[:, perm_inv].reshape(bsz, 1, psz, psz)
        # y = A(x)

        y = model(x1, x2)
        x_1, x_2, f = y.split(1, dim=1)

        # loss_pixel = (f - x3).abs().mean() + (f - x4).abs().mean()
        # loss_ssim = (1 - pytorch_ssim.ssim(f, x3)) + (1 - pytorch_ssim.ssim(f, x4))
        loss_source =(x_1 - x3).abs().mean() + (x_2 - x4).abs().mean()
        # loss = 10 * loss_pixel + 10 * loss_source + loss_ssim
        loss_p, loss_g = criterion(Fused=f, Infra=x_1, Visible=x_2)
        loss = 5*loss_p + loss_g + 5*loss_source

        # print('total_loss:', loss.item(), 'loss_pixel:', loss_p.item(), 'loss_grad:', 5*loss_g.item(), 'loss_source:', loss_source.item())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_avg += loss.item()
    scheduler.step()
    loss_avg /= iter_num
    log_data = "[%d/%d] Average loss: %f, time cost: %.2fs, cur lr is %f." % (epoch_i, epoch, loss_avg, time() - start_time, scheduler.get_last_lr()[0])
    print(log_data)
    with open(log_path, "a") as log_file:
        log_file.write(log_data + "\n")
    if epoch_i % args.save_interval == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))
    # cur_psnr, cur_ssim = test()
    # log_data = "CS Ratio is %.2f, PSNR is %.2f, SSIM is %.4f." % (ratio, cur_psnr, cur_ssim)
    with open(log_path, "a") as log_file:
        log_file.write(log_data + "\n")