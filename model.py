import torch, types
from torch import nn
import torch.nn.functional as F
from utils import *
from backprop import RevModule, VanillaBackProp, RevBackProp
from forward import MyUNet2DConditionModel_SD_v1_5_forward, \
                    MyCrossAttnDownBlock2D_SD_v1_5_forward, \
                    MyCrossAttnUpBlock2D_SD_v1_5_forward, \
                    MyResnetBlock2D_SD_v1_5_forward, \
                    MyTransformer2DModel_SD_v1_5_forward

def PatchUpsample(x, scale):
    n,c,h,w = x.shape
    x = torch.zeros(n,c,h,scale,w,scale).to(x.device) + x.view(n,c,h,1,w,1)
    return x.view(n,c,h*scale,scale*w)
scale = 2
A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
Ap = lambda z: PatchUpsample(z, scale)


import torch

def ddnm_plus_step_from_alpha_bar(abar_t,
                                  abar_prev,
                                  sigma_y: float,
                                  eps: float = 1e-12):
    """
    输入:
      abar_t    = \bar{alpha}_t
      abar_prev = \bar{alpha}_{t-1}（t=0 时传 1.0）
      sigma_y   = 观测噪声标准差
    输出:
      lambda_t, gamma_t, sigma_t, a_t  （与输入同 device/dtype 的 Tensor）
    """
    # 选定参考 device/dtype（优先跟随 abar_t，其次 abar_prev）
    ref = abar_t if torch.is_tensor(abar_t) else abar_prev
    device = ref.device if torch.is_tensor(ref) else torch.device('cpu')
    dtype  = ref.dtype  if torch.is_tensor(ref) else torch.float32

    abar_t    = torch.as_tensor(abar_t,    dtype=dtype, device=device)
    abar_prev = torch.as_tensor(abar_prev, dtype=dtype, device=device)
    sigma_y   = torch.as_tensor(float(sigma_y), dtype=dtype, device=device)
    eps_t     = torch.as_tensor(eps, dtype=dtype, device=device)

    # α_t, β_t
    alpha_t = abar_t / torch.clamp(abar_prev, min=eps_t)
    beta_t  = 1.0 - alpha_t

    # \tilde{β}_t 与 σ_t
    tilde_beta = beta_t * (1.0 - abar_prev) / torch.clamp(1.0 - abar_t, min=eps_t)
    sigma_t = torch.sqrt(torch.clamp(tilde_beta, min=eps_t))

    # a_t
    a_t = torch.sqrt(torch.clamp(abar_prev, min=eps_t)) * beta_t / torch.clamp(1.0 - abar_t, min=eps_t)

    # λ_t（阈值比较与比值都要同 device/dtype）
    thresh = a_t * sigma_y
    one = torch.ones_like(sigma_t)  # 保证同 device/dtype
    lambda_t = torch.where(sigma_t >= thresh, one, sigma_t / torch.clamp(thresh, min=eps_t))

    # γ_t
    gamma_t = torch.clamp(sigma_t ** 2 - (a_t * lambda_t * sigma_y) ** 2, min=0.0)

    return lambda_t, gamma_t, sigma_t, a_t


class Step(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=use_amp, cache_enabled=False):
            global t
            t = self.t
            cur_alpha_bar = alpha_bar[t]
            prev_alpha_bar = alpha_bar[t-1]

            t_tensor = torch.tensor([t], device=x.device, dtype=torch.long)
            input = x
            e, m = unet(input, t_tensor)
            # update x1, x2 %%%%%%%%%%%%%%%%%%%%%%%%%%
            x = (x - (1 - cur_alpha_bar).pow(0.5) * e) / cur_alpha_bar.pow(0.5) # 1. Denoising
            x_1, x_2, x_f = x.split(1, dim=1)  # 得到 N×1×H×W 的三个张量
            lambda_t, gamma_t, sigma_t, a_t = ddnm_plus_step_from_alpha_bar(cur_alpha_bar, prev_alpha_bar, sigma_y=0.03)
            x_1 = x_1 - lambda_t * (x_1 - x1)
            x_2 = x_2 - lambda_t * (x_2 - x2)
            x_f = x_f - lambda_t * (m * (x_1 - x1) + (1-m) * (x_2 - x2) - m*x1 - (1-m)*x2 + x_f)
            x = torch.cat([x_1, x_2, x_f], dim=1)
            x = prev_alpha_bar.pow(0.5) * x + (1 - prev_alpha_bar).pow(0.5) * e # 3. DDIM Sampling
            # x = x - AT(A(x) - y) # 2. RND
            return x

class Net(nn.Module):
    def __init__(self, T, unet):
        super().__init__()
        # self.clip_model = clip_model  # ✔ 将 clip 模型作为子模块
        self.body = nn.ModuleList([Step(T-i) for i in range(T)])
        self.input_help_scale_factor = nn.Parameter(torch.tensor([1.0]))
        self.merge_scale_factor = nn.Parameter(torch.tensor([0.0]))
        self.alpha = nn.Parameter(torch.full((T,), 0.5))
        self.unet = unet
        # self.unet_add_down_rev_modules_and_injectors(T)
        # self.unet_add_up_rev_modules_and_injectors(T)
        # self.unet_remove_resnet_time_emb_proj()
        self.unet_remove_cross_attn()
        # self.unet_set_inplace_to_true()
        # self.unet_replace_forward_methods()

    def unet_add_down_rev_modules_and_injectors(self, T):
        self.unet.down_blocks[0].register_module("injectors", nn.ModuleList([Injector(320, 512) for _ in range(4)]))
        self.unet.down_blocks[1].register_module("injectors", nn.ModuleList([Injector(640, 512) for _ in range(4)]))
        for i in range(2):
            self.unet.down_blocks[i].register_module("rev_module_lists", nn.ModuleList([]))
            self.unet.down_blocks[i].register_parameter("input_help_scale_factor", nn.Parameter(torch.ones(1,)))
            self.unet.down_blocks[i].register_parameter("merge_scale_factors", nn.Parameter(torch.zeros(2,)))
            for j in range(2):
                rev_module_list = nn.ModuleList([])
                if self.unet.down_blocks[i].resnets[j].in_channels == self.unet.down_blocks[i].resnets[j].out_channels:
                    rev_module_list.append(RevModule(self.unet.down_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2*j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.down_blocks[i].injectors[2*j+1]))
                self.unet.down_blocks[i].rev_module_lists.append(rev_module_list)

    def unet_add_up_rev_modules_and_injectors(self, T):
        self.unet.up_blocks[0].register_module("injectors", nn.ModuleList([Injector(640, 512) for _ in range(6)]))
        self.unet.up_blocks[1].register_module("injectors", nn.ModuleList([Injector(320, 512) for _ in range(6)]))
        for i in range(2):
            self.unet.up_blocks[i].register_parameter("input_help_scale_factor", nn.Parameter(torch.ones(1,)))
            self.unet.up_blocks[i].register_parameter("merge_scale_factor", nn.Parameter(torch.zeros(1,)))
            rev_module_list = nn.ModuleList([])
            for j in range(3):
                if j > 0:
                    rev_module_list.append(RevModule(self.unet.up_blocks[i].resnets[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2*j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].attentions[j]))
                rev_module_list.append(RevModule(self.unet.up_blocks[i].injectors[2*j+1]))
            self.unet.up_blocks[i].register_module("rev_module_list", rev_module_list)

    def unet_replace_forward_methods(self):
        from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D
        from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D
        from diffusers.models.resnet import ResnetBlock2D
        from diffusers.models.transformers.transformer_2d import Transformer2DModel
        def replace_forward_methods(module):
            if isinstance(module, CrossAttnDownBlock2D):
                module.forward = types.MethodType(MyCrossAttnDownBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, CrossAttnUpBlock2D):
                module.forward = types.MethodType(MyCrossAttnUpBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, ResnetBlock2D):
                module.forward = types.MethodType(MyResnetBlock2D_SD_v1_5_forward, module)
            elif isinstance(module, Transformer2DModel):
                module.forward = types.MethodType(MyTransformer2DModel_SD_v1_5_forward, module)
        self.unet.apply(replace_forward_methods)
        self.unet.forward = types.MethodType(MyUNet2DConditionModel_SD_v1_5_forward, self.unet)

    def unet_remove_resnet_time_emb_proj(self):
        from diffusers.models.resnet import ResnetBlock2D
        def ResnetBlock2D_remove_time_emb_proj(module):
            if isinstance(module, ResnetBlock2D):
                module.time_emb_proj = None
        self.unet.apply(ResnetBlock2D_remove_time_emb_proj)

    def unet_remove_cross_attn(self):
        from diffusers.models.attention import BasicTransformerBlock
        def BasicTransformerBlock_remove_cross_attn(module):
            if isinstance(module, BasicTransformerBlock):
                module.attn2 = module.norm2 = None
        self.unet.apply(BasicTransformerBlock_remove_cross_attn)
    
    def unet_set_inplace_to_true(self):
        def set_inplace_to_true(module):
            if isinstance(module, nn.Dropout) or isinstance(module, nn.SiLU):
                module.inplace = True
        self.unet.apply(set_inplace_to_true)

    def forward(self, x1_, x2_, use_amp_=True):
        global unet, x1, x2, alpha_bar, use_amp
        unet, x1, x2, use_amp = self.unet, x1_, x2_, use_amp_
        alpha_bar = torch.cat([torch.ones(1, device=x1.device), self.alpha.cumprod(dim=0)])
        f_T =  (x1 + x2) / 2.0
        x = alpha_bar[-1].pow(0.5) * torch.cat([x1, x2, f_T], dim=1)
        # print('shiyu*****************', x.shape)
        for step in self.body:  # 顺序执行
            x = step(x)  # 每层都保持 4 通道
        # print('shiyu66666666666666666', x.shape)
        return x