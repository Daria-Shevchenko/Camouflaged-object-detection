import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils_mffn.builder import MODELS
from utils_mffn.ops import cus_sample
import tensorly as tl


from utils_mffn.feat_vis import save_feat_grid
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from transformers import Sam3Processor, Sam3Model

import random


# ================================================================================ 
# VERSION 12 = SAM + rbg spectrum
# ================================================================================



# # ver12 ------------------------
# def build_color_tensor(x: torch.Tensor, mode: str = "rgb_opponent"):
#     assert x.ndim == 4 and x.shape[1] == 3, f"Expected [B,3,H,W], got {x.shape}"

#     r = x[:, 0:1]
#     g = x[:, 1:2]
#     b = x[:, 2:3]

#     if mode == "rgb_opponent":
#         rg = r - g
#         gb = g - b
#         rb = r - b
#         return torch.cat([r, g, b, rg, gb, rb], dim=1)
#     elif mode == "rgb":
#         return torch.cat([r, g, b], dim=1)
#     elif mode == "rgb_gray":
#         gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#         return torch.cat([r, g, b, gray], dim=1)
#     else:
#         raise ValueError(f"Unknown mode: {mode}")


# class ColorPyramidEncoder(nn.Module):
#     def __init__(self, in_ch=6, out_c=64):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#         )

#         self.l1 = nn.Sequential(
#             nn.Conv2d(32, out_c, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )
#         self.l2 = nn.Sequential(
#             nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )
#         self.l3 = nn.Sequential(
#             nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )
#         self.l4 = nn.Sequential(
#             nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )
#         self.l5 = nn.Sequential(
#             nn.Conv2d(out_c, out_c, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         x = self.stem(x)
#         c1 = self.l1(x)
#         c2 = self.l2(c1)
#         c3 = self.l3(c2)
#         c4 = self.l4(c3)
#         c5 = self.l5(c4)
#         return c5, c4, c3, c2, c1


# class FeatureFuse(nn.Module):
#     def __init__(self, in_c=64):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(0.1))
#         self.fuse = nn.Sequential(
#             nn.Conv2d(in_c * 2, in_c, 3, padding=1, bias=False),
#             nn.BatchNorm2d(in_c),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, sam_feat, color_feat):
#         if color_feat.shape[-2:] != sam_feat.shape[-2:]:
#             color_feat = F.interpolate(
#                 color_feat,
#                 size=sam_feat.shape[-2:],
#                 mode="bilinear",
#                 align_corners=False,
#             )
#         return self.fuse(torch.cat([sam_feat, self.alpha * color_feat], dim=1))
# # ver12 ------------------------


# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)


# # ver 12===============
#         self.use_color_branch = True
#         self.color_mode = "rgb_opponent"

#         color_in_ch = 6
#         self.color_encoder = ColorPyramidEncoder(in_ch=color_in_ch, out_c=64)

#         self.color_fuse_c5 = FeatureFuse(64)
#         self.color_fuse_c4 = FeatureFuse(64)
#         self.color_fuse_c3 = FeatureFuse(64)
#         self.color_fuse_c2 = FeatureFuse(64)
#         self.color_fuse_c1 = FeatureFuse(64)
# # ver 12===============


#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)



# # ver12 ====================================
#     def enrich_with_color(self, x, sam_feats):
#         if not self.use_color_branch:
#             return sam_feats

#         color_x = build_color_tensor(x, mode=self.color_mode)
#         color_feats = self.color_encoder(color_x)

#         s5, s4, s3, s2, s1 = sam_feats
#         c5, c4, c3, c2, c1 = color_feats

#         s5 = self.color_fuse_c5(s5, c5)
#         s4 = self.color_fuse_c4(s4, c4)
#         s3 = self.color_fuse_c3(s3, c3)
#         s2 = self.color_fuse_c2(s2, c2)
#         s1 = self.color_fuse_c1(s1, c1)

#         return (s5, s4, s3, s2, s1)
# # ver12 ====================================



# # in order to speed up the code VER 1 (without speed up)

#     # def encoder_translayer(self, x):
#     #     feats = self.shared_encoder(x)
#     #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#     #     trans_feats = self.translayer(en_feats)
#     #     return trans_feats


#     # def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#     #     c1_trans_feats = self.encoder_translayer(c1_scale)
#     #     o_trans_feats = self.encoder_translayer(o_scale)
#     #     c2_trans_feats = self.encoder_translayer(c2_scale)
#     #     a1_trans_feats = self.encoder_translayer(a1_scale)
#     #     a2_trans_feats = self.encoder_translayer(a2_scale)
#     #     feats = []
#     #     for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
#     #         CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
#     #         feats.append(CAMV_outs)

#     #     x = self.d5(feats[0])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d4(x + feats[1])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d3(x + feats[2])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d2(x + feats[3])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d1(x + feats[4])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     logits = self.out_layer_01(self.out_layer_00(x))
#     #     return dict(seg=logits)


# # in order to speed up the code VER 2 (with speed up)

#     # def encoder_translayer_5(self, c1, o, c2, a1, a2):
#     #     H, W = o.shape[-2], o.shape[-1]

#     #     def resize_like(x):
#     #         if x.shape[-2:] == (H, W):
#     #             return x
#     #         return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#     #     c1 = resize_like(c1)
#     #     c2 = resize_like(c2)
#     #     a1 = resize_like(a1)
#     #     a2 = resize_like(a2)

#     #     x = torch.cat([c1, o, c2, a1, a2], dim=0)
#     #     feats = self.shared_encoder(x)
#     #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#     #     c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#     #     if not hasattr(self, "_printed_shapes"):
#     #         self._printed_shapes = True
#     #         print("trans shapes:",
#     #             [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#     #     def split5(t): return t.chunk(5, dim=0)
#     #     c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#     #     c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#     #     c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#     #     c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#     #     c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#     #     return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#     #         (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#     #         (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#     #         (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#     #         (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     # ver12 ============================================================
#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         def split5(t):
#             return t.chunk(5, dim=0)

#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         feats_c1 = (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1)
#         feats_o  = (c5_o,  c4_o,  c3_o,  c2_o,  c1_o)
#         feats_c2 = (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2)
#         feats_a1 = (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1)
#         feats_a2 = (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)

#         feats_c1 = self.enrich_with_color(c1, feats_c1)
#         feats_o  = self.enrich_with_color(o,  feats_o)
#         feats_c2 = self.enrich_with_color(c2, feats_c2)
#         feats_a1 = self.enrich_with_color(a1, feats_a1)
#         feats_a2 = self.enrich_with_color(a2, feats_a2)

#         return feats_c1, feats_o, feats_c2, feats_a1, feats_a2
#     # ver12 ============================================================

#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         # target = "COD10K-NonCAM-3-Flying-1515.png"
#         target = "COD10K-CAM-1-Aquatic-4-Crocodile-110.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups




#  ============= ============= ============= ============= ============= ============= ============= =============
# # VERSION 11 - SAM + rotate (all images)
#  ============= ============= ============= ============= ============= ============= ============= =============



# import torchvision.transforms.functional as TF
# from torchvision.transforms import InterpolationMode


# # def rotate_tensor_batch(x, angle):
# #     rotated = []
# #     for i in range(x.shape[0]):
# #         rotated.append(
# #             TF.rotate(
# #                 x[i],
# #                 angle=angle,
# #                 interpolation=InterpolationMode.BILINEAR,
# #                 expand=False,
# #                 fill=0,
# #             )
# #         )
# #     return torch.stack(rotated, dim=0)



# def rot90_batch(x, k=1):
#     return torch.rot90(x, k=k, dims=(-2, -1))


# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)


# # in order to speed up the code VER 1 (without speed up)

#     # def encoder_translayer(self, x):
#     #     feats = self.shared_encoder(x)
#     #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#     #     trans_feats = self.translayer(en_feats)
#     #     return trans_feats


#     # def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#     #     c1_trans_feats = self.encoder_translayer(c1_scale)
#     #     o_trans_feats = self.encoder_translayer(o_scale)
#     #     c2_trans_feats = self.encoder_translayer(c2_scale)
#     #     a1_trans_feats = self.encoder_translayer(a1_scale)
#     #     a2_trans_feats = self.encoder_translayer(a2_scale)
#     #     feats = []
#     #     for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
#     #         CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
#     #         feats.append(CAMV_outs)

#     #     x = self.d5(feats[0])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d4(x + feats[1])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d3(x + feats[2])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d2(x + feats[3])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d1(x + feats[4])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     logits = self.out_layer_01(self.out_layer_00(x))
#     #     return dict(seg=logits)


# # in order to speed up the code VER 2 (with speed up)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     # def train_forward(self, data, **kwargs):
#     #     assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#     #     output = self.body(
#     #         c1_scale=data["image_c1"],
#     #         o_scale=data["image_o"],
#     #         c2_scale=data["image_c2"],
#     #         a1_scale=data["image_a1"],
#     #         a2_scale=data["image_a2"],
#     #     )
#     #     loss, loss_str = self.cal_loss(
#     #         all_preds=output,
#     #         gts=data["mask"],
#     #         iter_percentage=kwargs["curr"]["iter_percentage"],
#     #     )
#     #     return dict(sal=output["seg"].sigmoid()), loss, loss_str


# # ver 10 ================================
#     # def train_forward(self, data, **kwargs):
#     #     assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)

#     #     data["image_o"] = rotate_tensor_batch(data["image_o"], angle=90)

#     #     output = self.body(
#     #         c1_scale=data["image_c1"],
#     #         o_scale=data["image_o"],
#     #         c2_scale=data["image_c2"],
#     #         a1_scale=data["image_a1"],
#     #         a2_scale=data["image_a2"],
#     #     )
#     #     loss, loss_str = self.cal_loss(
#     #         all_preds=output,
#     #         gts=data["mask"],
#     #         iter_percentage=kwargs["curr"]["iter_percentage"],
#     #     )
#     #     return dict(sal=output["seg"].sigmoid()), loss, loss_str
# # ver 10 ================================  


# # ver 11 ================================
#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)

#         k = random.choice([0, 1, 2, 3])

#         data["image_c1"] = rot90_batch(data["image_c1"], k)
#         data["image_o"]  = rot90_batch(data["image_o"], k)
#         data["image_c2"] = rot90_batch(data["image_c2"], k)
#         data["image_a1"] = rot90_batch(data["image_a1"], k)
#         data["image_a2"] = rot90_batch(data["image_a2"], k)
#         data["mask"]     = rot90_batch(data["mask"], k)

#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str
# # ver 11 ================================  


#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         # target = "COD10K-NonCAM-3-Flying-1515.png"
#         target = "COD10K-CAM-1-Aquatic-4-Crocodile-110.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups





#  ============= ============= ============= ============= ============= ============= ============= =============
# # VERSION 10 - SAM + rotate (image_o)
#  ============= ============= ============= ============= ============= ============= ============= =============


# import torchvision.transforms.functional as TF
# from torchvision.transforms import InterpolationMode


# def rotate_tensor_batch(x, angle):
#     rotated = []
#     for i in range(x.shape[0]):
#         rotated.append(
#             TF.rotate(
#                 x[i],
#                 angle=angle,
#                 interpolation=InterpolationMode.BILINEAR,
#                 expand=False,
#                 fill=0,
#             )
#         )
#     return torch.stack(rotated, dim=0)





# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)


# # in order to speed up the code VER 1 (without speed up)

#     # def encoder_translayer(self, x):
#     #     feats = self.shared_encoder(x)
#     #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#     #     trans_feats = self.translayer(en_feats)
#     #     return trans_feats


#     # def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#     #     c1_trans_feats = self.encoder_translayer(c1_scale)
#     #     o_trans_feats = self.encoder_translayer(o_scale)
#     #     c2_trans_feats = self.encoder_translayer(c2_scale)
#     #     a1_trans_feats = self.encoder_translayer(a1_scale)
#     #     a2_trans_feats = self.encoder_translayer(a2_scale)
#     #     feats = []
#     #     for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
#     #         CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
#     #         feats.append(CAMV_outs)

#     #     x = self.d5(feats[0])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d4(x + feats[1])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d3(x + feats[2])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d2(x + feats[3])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d1(x + feats[4])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     logits = self.out_layer_01(self.out_layer_00(x))
#     #     return dict(seg=logits)


# # in order to speed up the code VER 2 (with speed up)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     # def train_forward(self, data, **kwargs):
#     #     assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#     #     output = self.body(
#     #         c1_scale=data["image_c1"],
#     #         o_scale=data["image_o"],
#     #         c2_scale=data["image_c2"],
#     #         a1_scale=data["image_a1"],
#     #         a2_scale=data["image_a2"],
#     #     )
#     #     loss, loss_str = self.cal_loss(
#     #         all_preds=output,
#     #         gts=data["mask"],
#     #         iter_percentage=kwargs["curr"]["iter_percentage"],
#     #     )
#     #     return dict(sal=output["seg"].sigmoid()), loss, loss_str


# # ver 10 ================================
#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)

#         data["image_o"] = rotate_tensor_batch(data["image_o"], angle=90)

#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str
# # ver 10 ================================  

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         # target = "COD10K-NonCAM-3-Flying-1515.png"
#         target = "COD10K-CAM-1-Aquatic-4-Crocodile-110.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups



#  ============= ============= ============= ============= ============= ============= ============= =============
# # VERSION 9 - SAM + high frequency FFT (version 5; paper -> Frequency-Spatial Entanglement Learning  
# + zoom in та zoom out =============
#  ============= ============= ============= ============= ============= ============= ============= =============


# # ver9 ==========================================
# def center_crop_tensor(x, target_h, target_w):
#     _, _, h, w = x.shape
#     start_h = (h - target_h) // 2
#     start_w = (w - target_w) // 2
#     return x[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

# def center_pad_tensor(x, target_h, target_w, pad_value=0.0):
#     _, _, h, w = x.shape
#     pad_h = target_h - h
#     pad_w = target_w - w
#     pad_top = pad_h // 2
#     pad_bottom = pad_h - pad_top
#     pad_left = pad_w // 2
#     pad_right = pad_w - pad_left
#     return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=pad_value)

# def zoom_in_tensor(x, scale=1.5, mode="bilinear"):
#     b, c, h, w = x.shape
#     new_h, new_w = int(h * scale), int(w * scale)
#     x_up = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=False)
#     x_crop = center_crop_tensor(x_up, h, w)
#     return x_crop

# def zoom_out_tensor(x, scale=0.5, mode="bilinear", pad_value=0.0):
#     b, c, h, w = x.shape
#     new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
#     x_down = F.interpolate(x, size=(new_h, new_w), mode=mode, align_corners=False)
#     x_pad = center_pad_tensor(x_down, h, w, pad_value=pad_value)
#     return x_pad
# # ver9 ==========================================

# # version 8 (fft enhance module) ======================================================================
# class FFTEnhance(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.weight = nn.Sequential(
#             nn.Conv2d(channels, channels // 8, 1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 8, channels, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x_fft = torch.fft.fft2(x.float())
#         weight = self.weight(x_fft.real)
#         x_fft = x_fft * weight
#         x_out = torch.fft.ifft2(x_fft)

#         return torch.abs(x_out)


# # version 8 (fft enhance module) ======================================================================


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_pil_image

# from transformers import Sam3Processor, Sam3Model



# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         # version 8 ============================================================
#         self.fft_enhance = FFTEnhance(out_dim)
#         # version 8 ============================================================
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         # version 8 ============================================================
#         # return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
#         out = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
#         out_f = self.fft_enhance(out)
#         return out + out_f
#         # version 8 ============================================================

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#           # version 8 ============================================================
#         self.fft_enhance = FFTEnhance(in_dim)
#           # version 8 ============================================================






#         self.trans2 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )

#         self.transz1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transz2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )

#         self.coe_c_z1 = nn.Parameter(torch.Tensor(1,64))
#         self.coe_h_z1 = nn.Parameter(torch.Tensor(mm_size,mm_size))
#         self.coe_w_z1 = nn.Parameter(torch.Tensor(mm_size,mm_size))

#         self.coe_c_mz = nn.Parameter(torch.Tensor(1,64))
#         self.coe_h_mz = nn.Parameter(torch.Tensor(mm_size,mm_size))
#         self.coe_w_mz = nn.Parameter(torch.Tensor(mm_size,mm_size))

#         self.coe_c_z2 = nn.Parameter(torch.Tensor(1,64))
#         self.coe_h_z2 = nn.Parameter(torch.Tensor(mm_size,mm_size))
#         self.coe_w_z2 = nn.Parameter(torch.Tensor(mm_size,mm_size))

#         for p in [self.coe_c_z1, self.coe_h_z1, self.coe_w_z1,
#                 self.coe_c_mz, self.coe_h_mz, self.coe_w_mz,
#                 self.coe_c_z2, self.coe_h_z2, self.coe_w_z2]:
#             p.data.uniform_(-0.5, 0.5)






#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         # self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
#         self.fuse = nn.Sequential(
#             ConvBNReLU(192, 128, 1),
#             ConvBNReLU(128, 64, 3,1,1),
#             ConvBNReLU(64, 64, 3,1,1)
#         )
   
   
   
#     def forward(self, c1, o, c2, a1, a2, zin, zout, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))


#         zin = self.transz1(zin)
#         zout = self.transz2(zout)
#         attn2 = self.trans2(torch.cat([zin, m, zout], dim=1))

#         attn_z1 = tl.tenalg.mode_dot(attn2, self.coe_c_z1, mode=1)
#         attn_z1 = tl.tenalg.mode_dot(attn_z1, self.coe_h_z1, mode=2)
#         attn_z1 = tl.tenalg.mode_dot(attn_z1, self.coe_w_z1, mode=3)
#         attn_z1 = torch.softmax(attn_z1, dim=1)

#         attn_mz = tl.tenalg.mode_dot(attn2, self.coe_c_mz, mode=1)
#         attn_mz = tl.tenalg.mode_dot(attn_mz, self.coe_h_mz, mode=2)
#         attn_mz = tl.tenalg.mode_dot(attn_mz, self.coe_w_mz, mode=3)
#         attn_mz = torch.softmax(attn_mz, dim=1)

#         attn_z2 = tl.tenalg.mode_dot(attn2, self.coe_c_z2, mode=1)
#         attn_z2 = tl.tenalg.mode_dot(attn_z2, self.coe_h_z2, mode=2)
#         attn_z2 = tl.tenalg.mode_dot(attn_z2, self.coe_w_z2, mode=3)
#         attn_z2 = torch.softmax(attn_z2, dim=1)

#         zmz = attn_z1 * zin + attn_mz * m + attn_z2 * zout
#         zmz = zmz.mul(self.channel_attn(zmz))
#         zmz = zmz.mul(self.spatial_attn(zmz))


#         # version 8 ============================================================
#         # lms = self.fuse(torch.cat([ama,cmc],dim=1))

#         # lms = self.fuse(torch.cat([ama, cmc], dim=1))
#         lms = self.fuse(torch.cat([ama, cmc, zmz], dim=1))
        
#         lms_f = self.fft_enhance(lms)
#         lms = lms + lms_f
#         # version 8 ============================================================

#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def encoder_translayer_7(self, c1, o, c2, a1, a2, zin, zout):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)
#         zin = resize_like(zin)
#         zout = resize_like(zout)

#         x = torch.cat([c1, o, c2, a1, a2, zin, zout], dim=0)

#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         def split7(t):
#             return t.chunk(7, dim=0)

#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2, c5_zin, c5_zout = split7(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2, c4_zin, c4_zout = split7(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2, c3_zin, c3_zout = split7(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2, c2_zin, c2_zout = split7(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2, c1_zin, c1_zout = split7(c1f)

#         return (
#             (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1),
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2),
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1),
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2),
#             (c5_zin, c4_zin, c3_zin, c2_zin, c1_zin),
#             (c5_zout, c4_zout, c3_zout, c2_zout, c1_zout),
#         )
        
#     # def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#     #     c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#     #         self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#     #     feats = []
#     #     for c1, o, c2, a1, a2, layer in zip(
#     #         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#     #     ):
#     #         feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#     #     x = self.d5(feats[0])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d4(x + feats[1])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d3(x + feats[2])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d2(x + feats[3])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d1(x + feats[4])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     logits = self.out_layer_01(self.out_layer_00(x))
#     #     return dict(seg=logits)

#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         z_in_scale = zoom_in_tensor(o_scale, scale=1.5)
#         z_out_scale = zoom_out_tensor(o_scale, scale=0.5, pad_value=0.0)

#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, z_in_trans_feats, z_out_trans_feats = \
#             self.encoder_translayer_7(
#                 c1_scale, o_scale, c2_scale, a1_scale, a2_scale, z_in_scale, z_out_scale
#             )

#         feats = []
#         for c1, o, c2, a1, a2, zin, zout, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats,
#             a1_trans_feats, a2_trans_feats, z_in_trans_feats, z_out_trans_feats,
#             self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2, zin=zin, zout=zout))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     # def train_forward(self, data, **kwargs):
#     #     assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#     #     output = self.body(
#     #         c1_scale=data["image_c1"],
#     #         o_scale=data["image_o"],
#     #         c2_scale=data["image_c2"],
#     #         a1_scale=data["image_a1"],
#     #         a2_scale=data["image_a2"],
#     #     )
#     #     loss, loss_str = self.cal_loss(
#     #         all_preds=output,
#     #         gts=data["mask"],
#     #         iter_percentage=kwargs["curr"]["iter_percentage"],
#     #     )
#     #     return dict(sal=output["seg"].sigmoid()), loss, loss_str
    
#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         # target = "COD10K-NonCAM-3-Flying-1515.png"
#         target = "COD10K-CAM-1-Aquatic-4-Crocodile-110.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups




#  ============= ============= ============= ============= ============= ============= ============= =============
# # VERSION 8 - SAM + high frequency FFT (version 4; paper -> Frequency-Spatial Entanglement Learning =============
#  ============= ============= ============= ============= ============= ============= ============= =============


# # version 8 (fft enhance module) ======================================================================
# class FFTEnhance(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.weight = nn.Sequential(
#             nn.Conv2d(channels, channels // 8, 1),
#             nn.ReLU(),
#             nn.Conv2d(channels // 8, channels, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x_fft = torch.fft.fft2(x.float())
#         weight = self.weight(x_fft.real)
#         x_fft = x_fft * weight
#         x_out = torch.fft.ifft2(x_fft)

#         return torch.abs(x_out)


# # version 8 (fft enhance module) ======================================================================


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_pil_image

# from transformers import Sam3Processor, Sam3Model



# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         # version 8 ============================================================
#         self.fft_enhance = FFTEnhance(out_dim)
#         # version 8 ============================================================
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         # version 8 ============================================================
#         # return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
#         out = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
#         out_f = self.fft_enhance(out)
#         return out + out_f
#         # version 8 ============================================================

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#           # version 8 ============================================================
#         self.fft_enhance = FFTEnhance(in_dim)
#           # version 8 ============================================================
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         # version 8 ============================================================
#         # lms = self.fuse(torch.cat([ama,cmc],dim=1))

#         lms = self.fuse(torch.cat([ama, cmc], dim=1))
#         lms_f = self.fft_enhance(lms)
#         lms = lms + lms_f
#         # version 8 ============================================================

#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         # target = "COD10K-NonCAM-3-Flying-1515.png"
#         target = "COD10K-CAM-1-Aquatic-4-Crocodile-110.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups





#  ============= ============= ============= ============= ============= ============= ============= =============
# # VERSION 7 - SAM + high frequency FFT (version 3; Loss(Encoder(FFT(Image)); FFT(Encoder(Image)))) =============
#  ============= ============= ============= ============= ============= ============= ============= =============


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_pil_image

# from transformers import Sam3Processor, Sam3Model



# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()



# # adding version 7 (utils functions) =========================================================================

# def _minmax_norm_per_sample(x, eps=1e-6):
#     b = x.shape[0]
#     x_flat = x.view(b, -1)
#     x_min = x_flat.min(dim=1)[0].view(b, 1, 1, 1)
#     x_max = x_flat.max(dim=1)[0].view(b, 1, 1, 1)
#     return (x - x_min) / (x_max - x_min + eps)


# def image_to_fft_rgb(x, eps=1e-6):
#     with torch.amp.autocast(device_type="cuda", enabled=False):
#         x = x.float()

#         gray = x.mean(dim=1, keepdim=True)
#         fft = torch.fft.fft2(gray, dim=(-2, -1))
#         fft = torch.fft.fftshift(fft, dim=(-2, -1))

#         mag = torch.log1p(torch.abs(fft) + eps)
#         mag = _minmax_norm_per_sample(mag, eps=eps)
#         mag_rgb = mag.repeat(1, 3, 1, 1)

#     return mag_rgb


# def feature_fft_mag(x, eps=1e-6):
#     with torch.amp.autocast(device_type="cuda", enabled=False):
#         x = x.float()
#         fft = torch.fft.fft2(x, dim=(-2, -1))
#         fft = torch.fft.fftshift(fft, dim=(-2, -1))
#         mag = torch.log1p(torch.abs(fft) + eps)
#     return mag    

# # adding version 7 (utils functions) =========================================================================

# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_TOKEN")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)
#         self.use_fft_encoder_loss = True
#         self.fft_encoder_loss_weight = 0.01
#         self.fft_encoder_loss_on = ("image_o",)
#         self.fft_encoder_scales = (0, 1, 2, 3, 4)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def encoder_translayer_single(self, x):
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)
#         return (c5, c4, c3, c2, c1)

#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)

#     def cal_encoder_fft_consistency_loss(self, img):
#         fft_img = image_to_fft_rgb(img)

#         orig_feats = self.encoder_translayer_single(img)
#         fft_feats = self.encoder_translayer_single(fft_img)

#         losses = []
#         for idx in self.fft_encoder_scales:
#             orig_f = orig_feats[idx]
#             fft_f = fft_feats[idx]

#             orig_spec = feature_fft_mag(orig_f)
#             fft_spec = feature_fft_mag(fft_f)

#             losses.append(F.l1_loss(fft_spec, orig_spec, reduction="mean"))

#         if len(losses) == 0:
#             return img.new_tensor(0.0)

#         return sum(losses) / len(losses)


#     # def train_forward(self, data, **kwargs):
#     #     assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#     #     output = self.body(
#     #         c1_scale=data["image_c1"],
#     #         o_scale=data["image_o"],
#     #         c2_scale=data["image_c2"],
#     #         a1_scale=data["image_a1"],
#     #         a2_scale=data["image_a2"],
#     #     )
#     #     loss, loss_str = self.cal_loss(
#     #         all_preds=output,
#     #         gts=data["mask"],
#     #         iter_percentage=kwargs["curr"]["iter_percentage"],
#     #     )
#     #     return dict(sal=output["seg"].sigmoid()), loss, loss_str


# # adding version 7 (replaced train_forward) =========================================================================

#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)

#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#             image_c1=data["image_c1"],
#             image_o=data["image_o"],
#             image_c2=data["image_c2"],
#             image_a1=data["image_a1"],
#             image_a2=data["image_a2"],
#         )

#         return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         target = "COD10K-NonCAM-3-Flying-1515.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]
# # adding version 7 (replaced train_forward) =========================================================================


#     # def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#     #     ual_coef = get_coef(iter_percentage, method)
#     #     losses = []
#     #     loss_str = []
#     #     # for main
#     #     for name, preds in all_preds.items():
#     #         resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#     #         sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#     #         losses.append(sod_loss)
#     #         loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#     #         ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#     #         ual_loss *= ual_coef
#     #         losses.append(ual_loss)
#     #         loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#     #     return sum(losses), " ".join(loss_str)


# # adding version 7 (replaced cal_loss) =========================================================================
#     def cal_loss(
#         self,
#         all_preds: dict,
#         gts: torch.Tensor,
#         method="cos",
#         iter_percentage: float = 0,
#         image_c1=None,
#         image_o=None,
#         image_c2=None,
#         image_a1=None,
#         image_a2=None,
#     ):
#         ual_coef = get_coef(iter_percentage, method)

#         losses = []
#         loss_str = []

#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:]).float().clamp(0, 1)

#             sod_loss = F.binary_cross_entropy_with_logits(
#                 input=preds,
#                 target=resized_gts,
#                 reduction="mean"
#             )
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")

#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts) * ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")

#         if self.use_fft_encoder_loss:
#             fft_losses = []

#             if "image_o" in self.fft_encoder_loss_on and image_o is not None:
#                 fft_losses.append(self.cal_encoder_fft_consistency_loss(image_o))

#             if "image_c1" in self.fft_encoder_loss_on and image_c1 is not None:
#                 fft_losses.append(self.cal_encoder_fft_consistency_loss(image_c1))

#             if "image_c2" in self.fft_encoder_loss_on and image_c2 is not None:
#                 fft_losses.append(self.cal_encoder_fft_consistency_loss(image_c2))

#             if "image_a1" in self.fft_encoder_loss_on and image_a1 is not None:
#                 fft_losses.append(self.cal_encoder_fft_consistency_loss(image_a1))

#             if "image_a2" in self.fft_encoder_loss_on and image_a2 is not None:
#                 fft_losses.append(self.cal_encoder_fft_consistency_loss(image_a2))

#             if len(fft_losses) > 0:
#                 fft_encoder_loss = sum(fft_losses) / len(fft_losses)
#                 fft_encoder_loss = fft_encoder_loss * self.fft_encoder_loss_weight
#                 losses.append(fft_encoder_loss)
#                 loss_str.append(
#                     f"ENC_FFT_{self.fft_encoder_loss_weight:.5f}: {fft_encoder_loss.item():.5f}"
#                 )

#         return sum(losses), " ".join(loss_str)

# # adding version 7 (replaced cal_loss) =========================================================================

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups




#  ============= ============= ============= ============= ============= ============= ============= =============
# # VERSION 6 - SAM + high frequency FFT (version 2; adding loss) =============
#  ============= ============= ============= ============= ============= ============= ============= =============



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_pil_image

# from transformers import Sam3Processor, Sam3Model



# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()




# #  addding version 6=======
# def cal_freq_loss(seg_logits, seg_gts, eps=1e-6):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)

#     pred = seg_logits.sigmoid().float()
#     target = seg_gts.float()

#     pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
#     target_fft = torch.fft.fft2(target, dim=(-2, -1))

#     pred_mag = torch.log1p(torch.abs(pred_fft) + eps)
#     target_mag = torch.log1p(torch.abs(target_fft) + eps)

#     return F.l1_loss(pred_mag, target_mag, reduction="mean")
# #  addding version 6=======





# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)


# # in order to speed up the code VER 1 (without speed up)

#     # def encoder_translayer(self, x):
#     #     feats = self.shared_encoder(x)
#     #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#     #     trans_feats = self.translayer(en_feats)
#     #     return trans_feats


#     # def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#     #     c1_trans_feats = self.encoder_translayer(c1_scale)
#     #     o_trans_feats = self.encoder_translayer(o_scale)
#     #     c2_trans_feats = self.encoder_translayer(c2_scale)
#     #     a1_trans_feats = self.encoder_translayer(a1_scale)
#     #     a2_trans_feats = self.encoder_translayer(a2_scale)
#     #     feats = []
#     #     for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
#     #         CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
#     #         feats.append(CAMV_outs)

#     #     x = self.d5(feats[0])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d4(x + feats[1])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d3(x + feats[2])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d2(x + feats[3])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d1(x + feats[4])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     logits = self.out_layer_01(self.out_layer_00(x))
#     #     return dict(seg=logits)


# # in order to speed up the code VER 2 (with speed up)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         target = "COD10K-NonCAM-3-Flying-1515.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]




#     # def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#     #     ual_coef = get_coef(iter_percentage, method)
#     #     losses = []
#     #     loss_str = []
#     #     # for main
#     #     for name, preds in all_preds.items():
#     #         resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#     #         sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#     #         losses.append(sod_loss)
#     #         loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#     #         ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#     #         ual_loss *= ual_coef
#     #         losses.append(ual_loss)
#     #         loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#     #     return sum(losses), " ".join(loss_str)


#     # adding ver 6 ===========
#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)

#         freq_coef = 0.02

#         losses = []
#         loss_str = []

#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])

#             sod_loss = F.binary_cross_entropy_with_logits(
#                 input=preds,
#                 target=resized_gts,
#                 reduction="mean"
#             )
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")

#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss = ual_loss * ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")

#             freq_loss = cal_freq_loss(seg_logits=preds, seg_gts=resized_gts)
#             freq_loss = freq_loss * freq_coef
#             losses.append(freq_loss)
#             loss_str.append(f"{name}_FREQ_{freq_coef:.5f}: {freq_loss.item():.5f}")

#         return sum(losses), " ".join(loss_str)
#     # adding ver 6 ===========





#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups







#  ============= ============= ============= ============= ============= ============= ============= =============
# # VERSION 5 - SAM + high frequency FFT (version 1; show not that good results in comparison with just sam) =============
#  ============= ============= ============= ============= ============= ============= ============= =============

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_pil_image

# from transformers import Sam3Processor, Sam3Model



# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.freq_loss_weight = 0.02
#         self.freq_loss_levels = [0, 1, 2]
#         self.freq_loss_type = "l1"

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)

#     def make_fft_image(self, x: torch.Tensor, eps: float = 1e-6):
#         with torch.amp.autocast("cuda", enabled=False):
#             x32 = x.float()
#             x_gray = x32.mean(dim=1, keepdim=True)
#             x_fft = torch.fft.fft2(x_gray, norm="ortho")
#             x_mag = torch.log1p(torch.abs(x_fft))
#             x_mag = torch.fft.fftshift(x_mag, dim=(-2, -1))

#             B = x_mag.shape[0]
#             x_flat = x_mag.view(B, -1)
#             x_min = x_flat.min(dim=1)[0].view(B, 1, 1, 1)
#             x_max = x_flat.max(dim=1)[0].view(B, 1, 1, 1)
#             x_mag = (x_mag - x_min) / (x_max - x_min + eps)

#             x_mag = x_mag.repeat(1, 3, 1, 1)

#         return x_mag.to(x.dtype)

#     def encode_single(self, x: torch.Tensor):
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)
#         return [c5, c4, c3, c2, c1]

#     def fft_feature_map(self, feat: torch.Tensor):
#         with torch.amp.autocast("cuda", enabled=False):
#             feat32 = feat.float()
#             feat_fft = torch.fft.fft2(feat32, norm="ortho")
#             feat_mag = torch.log1p(torch.abs(feat_fft))
#         return feat_mag.to(feat.dtype)

#     def calc_frequency_consistency_loss(self, x: torch.Tensor):
#         spatial_feats = self.encode_single(x)
#         x_fft_img = self.make_fft_image(x)
#         freq_feats = self.encode_single(x_fft_img)

#         losses = []
#         for i in self.freq_loss_levels:
#             f_spatial = spatial_feats[i]
#             f_freq_in = freq_feats[i]

#             f_spatial_fft = self.fft_feature_map(f_spatial)

#             if self.freq_loss_type == "l1":
#                 l = F.l1_loss(f_freq_in, f_spatial_fft)
#             elif self.freq_loss_type == "mse":
#                 l = F.mse_loss(f_freq_in, f_spatial_fft)
#             else:
#                 raise ValueError(f"Unknown freq_loss_type: {self.freq_loss_type}")

#             losses.append(l)

#         return sum(losses) / len(losses)

# # in order to speed up the code VER 1 (without speed up)

#     # def encoder_translayer(self, x):
#     #     feats = self.shared_encoder(x)
#     #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#     #     trans_feats = self.translayer(en_feats)
#     #     return trans_feats


#     # def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#     #     c1_trans_feats = self.encoder_translayer(c1_scale)
#     #     o_trans_feats = self.encoder_translayer(o_scale)
#     #     c2_trans_feats = self.encoder_translayer(c2_scale)
#     #     a1_trans_feats = self.encoder_translayer(a1_scale)
#     #     a2_trans_feats = self.encoder_translayer(a2_scale)
#     #     feats = []
#     #     for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
#     #         CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
#     #         feats.append(CAMV_outs)

#     #     x = self.d5(feats[0])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d4(x + feats[1])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d3(x + feats[2])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d2(x + feats[3])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d1(x + feats[4])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     logits = self.out_layer_01(self.out_layer_00(x))
#     #     return dict(seg=logits)


# # in order to speed up the code VER 2 (with speed up)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     # def train_forward(self, data, **kwargs):
#     #     assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#     #     output = self.body(
#     #         c1_scale=data["image_c1"],
#     #         o_scale=data["image_o"],
#     #         c2_scale=data["image_c2"],
#     #         a1_scale=data["image_a1"],
#     #         a2_scale=data["image_a2"],
#     #     )
#     #     loss, loss_str = self.cal_loss(
#     #         all_preds=output,
#     #         gts=data["mask"],
#     #         iter_percentage=kwargs["curr"]["iter_percentage"],
#     #     )
#     #     return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)

#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         seg_loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )

#         freq_consistency_loss = self.calc_frequency_consistency_loss(data["image_o"])
#         total_loss = seg_loss + self.freq_loss_weight * freq_consistency_loss

#         loss_str = (
#             loss_str
#             + f" FREQ_CONS({self.freq_loss_weight:.3f}): "
#             + f"{(self.freq_loss_weight * freq_consistency_loss).item():.5f}"
#         )

#         return dict(sal=output["seg"].sigmoid()), total_loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         target = "COD10K-NonCAM-3-Flying-1515.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups





#  ============= ============= ============= ============= ============= ============= ============= =============
# VERSION 4 - SAM + high frequency SOBEL
#  ============= ============= ============= ============= ============= ============= ============= =============


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_pil_image

# from transformers import Sam3Processor, Sam3Model


# class SobelHF(nn.Module):
#     def __init__(self):
#         super().__init__()
#         kx = torch.tensor([[-1, 0, 1],
#                            [-2, 0, 2],
#                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         ky = torch.tensor([[-1, -2, -1],
#                            [ 0,  0,  0],
#                            [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
#         self.register_buffer("kx", kx)
#         self.register_buffer("ky", ky)

#     def forward(self, x):
#         if x.shape[1] == 3:
#             gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
#         else:
#             gray = x

#         gx = F.conv2d(gray, self.kx, padding=1)
#         gy = F.conv2d(gray, self.ky, padding=1)
#         mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
#         return mag




# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()

#         self.hf_proj = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         self.hf_gamma = nn.Parameter(torch.tensor(0.0))

#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, hf=None, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)

#         m = self.conv_m(o)
#         if hf is not None:
#             m = m + self.hf_gamma * self.hf_proj(hf)

#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms


# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         # -------------------------------------- adding sobel hf
#         self.hf_extractor = SobelHF()
#         self.hf_to_64 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         # ---------------------------------------------

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)


# # in order to speed up the code VER 1 (without speed up)

#     # def encoder_translayer(self, x):
#     #     feats = self.shared_encoder(x)
#     #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#     #     trans_feats = self.translayer(en_feats)
#     #     return trans_feats


#     # def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#     #     c1_trans_feats = self.encoder_translayer(c1_scale)
#     #     o_trans_feats = self.encoder_translayer(o_scale)
#     #     c2_trans_feats = self.encoder_translayer(c2_scale)
#     #     a1_trans_feats = self.encoder_translayer(a1_scale)
#     #     a2_trans_feats = self.encoder_translayer(a2_scale)
#     #     feats = []
#     #     for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
#     #         CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
#     #         feats.append(CAMV_outs)

#     #     x = self.d5(feats[0])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d4(x + feats[1])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d3(x + feats[2])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d2(x + feats[3])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d1(x + feats[4])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     logits = self.out_layer_01(self.out_layer_00(x))
#     #     return dict(seg=logits)


# # in order to speed up the code VER 2 (with speed up)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         # adding hf
#         hf = self.hf_extractor(o_scale)
#         hf = self.hf_to_64(hf)


#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []

#         # replaced by hf
#         # for c1, o, c2, a1, a2, layer in zip(
#         #     c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         # ):
#         #     feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))
#         # replaced by hf

#         # adding hf
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             hf_s = F.interpolate(hf, size=o.shape[-2:], mode="bilinear", align_corners=False)
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2, hf=hf_s))

#          # adding hf

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         target = "COD10K-NonCAM-3-Flying-1515.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups






#  ============= ============= ============= ============= ============= ============= ============= =============
# VERSION 3 - SAM
#  ============= ============= ============= ============= ============= ============= ============= =============

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.transforms.functional import to_pil_image

# from transformers import Sam3Processor, Sam3Model



# #  ==================sam3===================================

# class SAM3HFFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam3", device=None, input_is_0_1=True, fpn_level_order="auto"):
#         super().__init__()
#         self.model = Sam3Model.from_pretrained(pretrained_name)
#         self.processor = Sam3Processor.from_pretrained(pretrained_name)

#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.fpn_level_order = fpn_level_order

#         self.feat_dim = None

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         self.model.to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device
#             self.model.to(self._device)

#         x_cpu = x.detach().to("cpu")
#         pil_list = []
#         for i in range(x_cpu.shape[0]):
#             img = x_cpu[i]
#             if self.input_is_0_1:
#                 img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
#             else:
#                 img = img.clamp(0, 255).to(torch.uint8)
#             pil_list.append(to_pil_image(img))

#         inputs = self.processor(images=pil_list, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].to(self._device)


#         if hasattr(self.model, "vision_encoder"):
#             vision_encoder = self.model.vision_encoder
#         elif hasattr(self.model, "perception_encoder"):
#             vision_encoder = self.model.perception_encoder
#         elif hasattr(self.model, "model") and hasattr(self.model.model, "vision_encoder"):
#             vision_encoder = self.model.model.vision_encoder
#         else:
#             raise RuntimeError("Cannot find SAM3 vision encoder inside Sam3Model. Inspect model attributes.")

#         vision_out = vision_encoder(pixel_values=pixel_values)

#         if not hasattr(vision_out, "fpn_hidden_states") or vision_out.fpn_hidden_states is None:
#             raise RuntimeError("vision encoder output has no fpn_hidden_states. Check your Transformers SAM3 version.")

#         fpn = vision_out.fpn_hidden_states

#         fpn_list = list(fpn)

#         if self.fpn_level_order == "auto":
#             fpn_list = sorted(fpn_list, key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
#         elif self.fpn_level_order == "high_to_low":
#             pass
#         elif self.fpn_level_order == "low_to_high":
#             fpn_list = list(reversed(fpn_list))
#         else:
#             raise ValueError("fpn_level_order must be one of: auto, high_to_low, low_to_high")

#         if len(fpn_list) >= 5:
#             c1, c2, c3, c4, c5 = fpn_list[:5]
#         else:
#             while len(fpn_list) < 5:
#                 last = fpn_list[-1]
#                 down = F.avg_pool2d(last, kernel_size=2, stride=2)
#                 fpn_list.append(down)
#             c1, c2, c3, c4, c5 = fpn_list[:5]

#         if self.feat_dim is None:
#             self.feat_dim = int(c3.shape[1])

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


#         #  ==================sam3===================================

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, last_module=ASPP):
#         super().__init__()
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             last_module(in_dim=2048, out_dim=out_c),
#         )
#         self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
   
   
   
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms






# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# import torch
# from torchvision.models.feature_extraction import create_feature_extractor
# from transformers import SamModel, SamProcessor


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import SamModel, SamProcessor

# class SAMFeatureExtractor(nn.Module):
#     def __init__(self, pretrained_name="facebook/sam-vit-huge", device=None, input_is_0_1=True):
#         super().__init__()
#         self.model = SamModel.from_pretrained(pretrained_name)
#         self.proc = SamProcessor.from_pretrained(pretrained_name)
#         self.input_is_0_1 = input_is_0_1
#         self._device = device

#         self.feat_dim = int(self.model.config.vision_config.output_channels)

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)
#         if args and hasattr(args[0], "type"):
#             self._device = args[0]
#         elif "device" in kwargs:
#             self._device = kwargs["device"]
#         return self

#     @torch.no_grad()
#     def forward(self, x: torch.Tensor):
#         if self._device is None:
#             self._device = x.device

#         inputs = self.proc(
#             images=x,
#             return_tensors="pt",
#             do_rescale=not self.input_is_0_1,
#         )
#         pixel_values = inputs["pixel_values"].to(self._device)

#         img_emb = self.model.get_image_embeddings(pixel_values=pixel_values)

#         c3 = img_emb
#         c2 = F.interpolate(img_emb, scale_factor=2.0, mode="bilinear", align_corners=False)
#         c1 = F.interpolate(img_emb, scale_factor=4.0, mode="bilinear", align_corners=False)
#         c4 = F.avg_pool2d(img_emb, kernel_size=2, stride=2)
#         c5 = F.avg_pool2d(img_emb, kernel_size=4, stride=4)

#         return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}

# class TransLayerSAM(nn.Module):
#     def __init__(self, out_c, last_module=ASPP, in_dim=256):
#         super().__init__()
#         self.c5_down = nn.Sequential(last_module(in_dim=in_dim, out_dim=out_c))
#         self.c4_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(in_dim, out_c, 3, 1, 1))

#     def forward(self, xs):
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1


# from huggingface_hub import login

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)

#         # self.shared_encoder = SAMFeatureExtractor(
#         #     pretrained_name="facebook/sam-vit-huge",
#         #     input_is_0_1=True,
#         # )
#         # sam_feat_dim = self.shared_encoder.feat_dim
#         # self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         login("hf_token")

#         self.shared_encoder = SAM3HFFeatureExtractor(
#             pretrained_name="facebook/sam3",
#             input_is_0_1=True,
#         )

#         sam_feat_dim = 256 
#         self.translayer = TransLayerSAM(out_c=64, in_dim=sam_feat_dim)

#         dim = [64, 64, 64, 64, 64]
#         # size = [16, 32, 64, 128, 256]
#         size = [18, 36, 72, 144, 288]

#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)


# # in order to speed up the code VER 1 (without speed up)

#     # def encoder_translayer(self, x):
#     #     feats = self.shared_encoder(x)
#     #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#     #     trans_feats = self.translayer(en_feats)
#     #     return trans_feats


#     # def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#     #     c1_trans_feats = self.encoder_translayer(c1_scale)
#     #     o_trans_feats = self.encoder_translayer(o_scale)
#     #     c2_trans_feats = self.encoder_translayer(c2_scale)
#     #     a1_trans_feats = self.encoder_translayer(a1_scale)
#     #     a2_trans_feats = self.encoder_translayer(a2_scale)
#     #     feats = []
#     #     for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
#     #         CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
#     #         feats.append(CAMV_outs)

#     #     x = self.d5(feats[0])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d4(x + feats[1])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d3(x + feats[2])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d2(x + feats[3])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     x = self.d1(x + feats[4])
#     #     x = cus_sample(x, mode="scale", factors=2)
#     #     logits = self.out_layer_01(self.out_layer_00(x))
#     #     return dict(seg=logits)


# # in order to speed up the code VER 2 (with speed up)

#     def encoder_translayer_5(self, c1, o, c2, a1, a2):
#         H, W = o.shape[-2], o.shape[-1]

#         def resize_like(x):
#             if x.shape[-2:] == (H, W):
#                 return x
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

#         c1 = resize_like(c1)
#         c2 = resize_like(c2)
#         a1 = resize_like(a1)
#         a2 = resize_like(a2)

#         x = torch.cat([c1, o, c2, a1, a2], dim=0)
#         feats = self.shared_encoder(x)
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2f, c1f = self.translayer(en_feats)

#         if not hasattr(self, "_printed_shapes"):
#             self._printed_shapes = True
#             print("trans shapes:",
#                 [t.shape[-2:] for t in [c5, c4, c3, c2f, c1f]])

#         def split5(t): return t.chunk(5, dim=0)
#         c5_c1, c5_o, c5_c2, c5_a1, c5_a2 = split5(c5)
#         c4_c1, c4_o, c4_c2, c4_a1, c4_a2 = split5(c4)
#         c3_c1, c3_o, c3_c2, c3_a1, c3_a2 = split5(c3)
#         c2_c1, c2_o, c2_c2, c2_a1, c2_a2 = split5(c2f)
#         c1_c1, c1_o, c1_c2, c1_a1, c1_a2 = split5(c1f)

#         return (c5_c1, c4_c1, c3_c1, c2_c1, c1_c1), \
#             (c5_o,  c4_o,  c3_o,  c2_o,  c1_o),  \
#             (c5_c2, c4_c2, c3_c2, c2_c2, c1_c2), \
#             (c5_a1, c4_a1, c3_a1, c2_a1, c1_a1), \
#             (c5_a2, c4_a2, c3_a2, c2_a2, c1_a2)


#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats = \
#             self.encoder_translayer_5(c1_scale, o_scale, c2_scale, a1_scale, a2_scale)

#         feats = []
#         for c1, o, c2, a1, a2, layer in zip(
#             c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers
#         ):
#             feats.append(layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2))

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)


#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )

#         # target = "COD10K-NonCAM-3-Flying-1515.png"
#         target = "COD10K-CAM-1-Aquatic-4-Crocodile-110.png"
#         img_names = data["img_name"]

#         # print(img_names)

#         if isinstance(img_names, (list, tuple)):
#             hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
#             if not hits:
#                 return output["seg"]
#             idx = hits[0]
#         else:
#             if os.path.basename(img_names) != target:
#                 return output["seg"]
#             idx = 0

#         save_path = f"work_dirs/vis/mffn_feats_{target}.png"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)

#         feats = self.shared_encoder(data["image_o"])
#         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
#         c5, c4, c3, c2, c1 = self.translayer(en_feats)

#         save_feat_grid(
#             save_path=save_path,
#             input_img_chw=data["image_o"][idx],
#             feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
#             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
#         )

#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     def get_grouped_params(self):
#         param_groups = {}
#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder.layer"):
#                 param_groups.setdefault("pretrained", []).append(param)
#             elif name.startswith("shared_encoder."):
#                 param_groups.setdefault("fixed", []).append(param)
#             else:
#                 param_groups.setdefault("retrained", []).append(param)
#         return param_groups



#  ============= ============= ============= ============= ============= ============= ============= =============
# VERSION 2 - DINO
#  ============= ============= ============= ============= ============= ============= ============= =============


tl.set_backend('pytorch')

###############  Multi-scale features Process Module  ##################

class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

class TransLayer(nn.Module):
    def __init__(self, out_c, last_module=ASPP):
        super().__init__()
        self.c5_down = nn.Sequential(
            # ConvBNReLU(2048, 256, 3, 1, 1),
            last_module(in_dim=2048, out_dim=out_c),
        )
        self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c5, c4, c3, c2, c1
    
###############  Cross-View Attention Module  ##################

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CAMV(nn.Module):
    def __init__(self, in_dim, mm_size):
        super().__init__()
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
            ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.trans1 = nn.Sequential(
            ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
            ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
       
        self.transa1 = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.transa2 = nn.Sequential(
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
        )
        self.mm_size = mm_size
        self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_c1.data.uniform_(-0.5,0.5)
        self.coe_h_c1.data.uniform_(-0.5,0.5)
        self.coe_w_c1.data.uniform_(-0.5,0.5)
        
        self.coe_c_md.data.uniform_(-0.5,0.5)
        self.coe_h_md.data.uniform_(-0.5,0.5)
        self.coe_w_md.data.uniform_(-0.5,0.5)
        
        self.coe_c_c2.data.uniform_(-0.5,0.5)
        self.coe_h_c2.data.uniform_(-0.5,0.5)
        self.coe_w_c2.data.uniform_(-0.5,0.5)
        
        self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
        self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
        self.coe_c_a1.data.uniform_(-0.5,0.5)
        self.coe_h_a1.data.uniform_(-0.5,0.5)
        self.coe_w_a1.data.uniform_(-0.5,0.5)
        
        self.coe_c_ma.data.uniform_(-0.5,0.5)
        self.coe_h_ma.data.uniform_(-0.5,0.5)
        self.coe_w_ma.data.uniform_(-0.5,0.5)
        
        self.coe_c_a2.data.uniform_(-0.5,0.5)
        self.coe_h_a2.data.uniform_(-0.5,0.5)
        self.coe_w_a2.data.uniform_(-0.5,0.5)
        self.channel_attn = ChannelAttention(64)
        self.spatial_attn = SpatialAttention()
        self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
    def forward(self, c1, o, c2, a1, a2, return_feats=False):
        tgt_size = o.shape[2:]
        c1 = self.conv_l_pre_down(c1)
        c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
        c1 = self.conv_l_post_down(c1)
        m = self.conv_m(o)
        c2 = self.conv_s_pre_up(c2)
        c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
        c2 = self.conv_s_post_up(c2)
        attn = self.trans(torch.cat([c1, m, c2], dim=1))
        attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
        attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
        attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
        attn_c1 = torch.softmax(attn_c1, dim=1)
        
        attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
        attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
        attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
        attn_md = torch.softmax(attn_md, dim=1)
        
        attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
        attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
        attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
        attn_c2 = torch.softmax(attn_c2, dim=1)
        
        cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

        a1 = self.transa1(a1)
        a2 = self.transa2(a2)
        attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
        attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
        attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
        attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
        attn_a1 = torch.softmax(attn_a1, dim=1)
        
        attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
        attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
        attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
        attn_ma = torch.softmax(attn_ma, dim=1)
        
        attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
        attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
        attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
        attn_a2 = torch.softmax(attn_a2, dim=1)
        
        ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
        ama = ama.mul(self.channel_attn(ama))
        ama = ama.mul(self.spatial_attn(ama))
        lms = self.fuse(torch.cat([ama,cmc],dim=1))
        return lms
class Progressive_Iteration(nn.Module):
    def __init__(self, input_channels):
        super(Progressive_Iteration, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)
        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)
        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)
        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
        return ce

class CFU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups
        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)
        self.fp = Progressive_Iteration(192)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
        outs = []
        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(2, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(2, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(1, dim=1))
        out = torch.cat([o[0] for o in outs], dim=1)
        out = self.fp(out)
        out = self.fuse(out)
        return self.final_relu(out + x)

def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


import torch
from torchvision.models.feature_extraction import create_feature_extractor


@MODELS.register()
class MFFN(BasicModelClass):
    def __init__(self):
        super().__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)


        # dino_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')

        # return_nodes = {
        #     "relu": "c1",
        #     "layer1": "c2",
        #     "layer2": "c3",
        #     "layer3": "c4",
        #     "layer4": "c5",
        # }
        # self.shared_encoder = create_feature_extractor(dino_backbone, return_nodes=return_nodes)

        self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
        dim = [64, 64, 64, 64, 64]
        size = [12, 24, 48, 96, 192]
        self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
        self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
        self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(32, 1, 1)

    def encoder_translayer(self, x):
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)
        return trans_feats

    # def encoder_translayer(self, x):
    #     feats = self.shared_encoder(x)
    #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
    #     trans_feats = self.translayer(en_feats)
    #     return trans_feats


    def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
        c1_trans_feats = self.encoder_translayer(c1_scale)
        o_trans_feats = self.encoder_translayer(o_scale)
        c2_trans_feats = self.encoder_translayer(c2_scale)
        a1_trans_feats = self.encoder_translayer(a1_scale)
        a2_trans_feats = self.encoder_translayer(a2_scale)
        feats = []
        for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
            CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
            feats.append(CAMV_outs)

        x = self.d5(feats[0])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + feats[1])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + feats[2])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + feats[3])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + feats[4])
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return dict(seg=logits)
    def train_forward(self, data, **kwargs):
        assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
        output = self.body(
            c1_scale=data["image_c1"],
            o_scale=data["image_o"],
            c2_scale=data["image_c2"],
            a1_scale=data["image_a1"],
            a2_scale=data["image_a2"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    # def test_forward(self, data, **kwargs):
    #     output = self.body(
    #         c1_scale=data["image_c1"],
    #         o_scale=data["image_o"],
    #         c2_scale=data["image_c2"],
    #         a1_scale=data["image_a1"],
    #         a2_scale=data["image_a2"],
    #     )
    #     return output["seg"]


#  resnet
    def test_forward(self, data, **kwargs):
        output = self.body(
            c1_scale=data["image_c1"],
            o_scale=data["image_o"],
            c2_scale=data["image_c2"],
            a1_scale=data["image_a1"],
            a2_scale=data["image_a2"],
        )

        save_path = "work_dirs/vis/mffn_feats.png"



        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # en_feats = self.shared_encoder(data["image_o"])
            # trans_feats = self.translayer(en_feats)
            # c5, c4, c3, c2, c1 = trans_feats

            # save_feat_grid(
            #     save_path=save_path,
            #     input_img_chw=data["image_o"][0],
            #     feat_list=[c3[0], c1[0]],
            #     titles=["Trans c3", "Trans c1"],
            # )

            en_feats = self.shared_encoder(data["image_o"])
            trans_feats = self.translayer(en_feats)


            c5, c4, c3, c2, c1 = trans_feats


            feat_list = [c5[0], c4[0], c3[0], c2[0], c1[0]]
            titles = ["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"]

            save_feat_grid(
                save_path=save_path,
                input_img_chw=data["image_o"][0],
                feat_list=feat_list,
                titles=titles,
            )

        return output["seg"]


#  dino-resnet
    # def test_forward(self, data, **kwargs):
    #     output = self.body(
    #         c1_scale=data["image_c1"],
    #         o_scale=data["image_o"],
    #         c2_scale=data["image_c2"],
    #         a1_scale=data["image_a1"],
    #         a2_scale=data["image_a2"],
    #     )

    #     save_path = "work_dirs/vis/mffn_feats.png"

    #     if not os.path.exists(save_path):
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #         feats = self.shared_encoder(data["image_o"])
    #         en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
    #         c5, c4, c3, c2, c1 = self.translayer(en_feats)

    #         save_feat_grid(
    #             save_path=save_path,
    #             input_img_chw=data["image_o"][1],
    #             feat_list=[c5[0], c4[0], c3[0], c2[0], c1[0]],
    #             titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
    #         )

    #     return output["seg"]

#  dino-resnet changed! 
    # def test_forward(self, data, **kwargs):
    #     output = self.body(
    #         c1_scale=data["image_c1"],
    #         o_scale=data["image_o"],
    #         c2_scale=data["image_c2"],
    #         a1_scale=data["image_a1"],
    #         a2_scale=data["image_a2"],
    #     )

    #     target = "COD10K-NonCAM-3-Flying-1515.png"
    #     img_names = data["img_name"]

    #     print(img_names)

    #     if isinstance(img_names, (list, tuple)):
    #         hits = [i for i, n in enumerate(img_names) if os.path.basename(n) == target]
    #         if not hits:
    #             return output["seg"]
    #         idx = hits[0]
    #     else:
    #         if os.path.basename(img_names) != target:
    #             return output["seg"]
    #         idx = 0

    #     save_path = f"work_dirs/vis/mffn_feats_{target}.png"
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #     feats = self.shared_encoder(data["image_o"])
    #     en_feats = [feats["c1"], feats["c2"], feats["c3"], feats["c4"], feats["c5"]]
    #     c5, c4, c3, c2, c1 = self.translayer(en_feats)

    #     save_feat_grid(
    #         save_path=save_path,
    #         input_img_chw=data["image_o"][idx],
    #         feat_list=[c5[idx], c4[idx], c3[idx], c2[idx], c1[idx]],
    #         titles=["Trans c5", "Trans c4", "Trans c3", "Trans c2", "Trans c1"],
    #     )

    #     return output["seg"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        ual_coef = get_coef(iter_percentage, method)
        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
            sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
            ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            ual_loss *= ual_coef
            losses.append(ual_loss)
            loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("shared_encoder.layer"):
                param_groups.setdefault("pretrained", []).append(param)
            elif name.startswith("shared_encoder."):
                param_groups.setdefault("fixed", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        return param_groups

















#  ============= ============= ============= ============= ============= ============= ============= =============

# VERSION 1 - DINO
# MY VERSION WITH DINO MODEL
#  ============= ============= ============= ============= ============= ============= ============= =============

# import numpy as np
# import timm
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.utils.checkpoint import checkpoint

# from methods.module.base_model import BasicModelClass
# from methods.module.conv_block import ConvBNReLU
# from utils.builder import MODELS
# from utils.ops import cus_sample
# import tensorly as tl

# tl.set_backend('pytorch')

# ###############  Multi-scale features Process Module  ##################

# class ASPP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ASPP, self).__init__()
#         self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
#         self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
#         self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
#         self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
#         self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(x)
#         conv3 = self.conv3(x)
#         conv4 = self.conv4(x)
#         conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
#         return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

# class TransLayer(nn.Module):
#     def __init__(self, out_c, in_dims, last_module=ASPP):
#         super().__init__()

#         c1_dim, c2_dim, c3_dim, c4_dim, c5_dim = in_dims
        
#         self.c5_down = nn.Sequential(
#             # ConvBNReLU(2048, 256, 3, 1, 1),
#             # last_module(in_dim=2048, out_dim=out_c),
#             last_module(in_dim=c5_dim, out_dim=out_c),
#         )
#         # self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
#         # self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
#         # self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
#         # self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

#         self.c4_down = nn.Sequential(ConvBNReLU(c4_dim, out_c, 3, 1, 1))
#         self.c3_down = nn.Sequential(ConvBNReLU(c3_dim, out_c, 3, 1, 1))
#         self.c2_down = nn.Sequential(ConvBNReLU(c2_dim, out_c, 3, 1, 1))
#         self.c1_down = nn.Sequential(ConvBNReLU(c1_dim, out_c, 3, 1, 1))


#     def forward(self, xs):
#         assert isinstance(xs, (tuple, list))
#         assert len(xs) == 5
#         c1, c2, c3, c4, c5 = xs
#         c5 = self.c5_down(c5)
#         c4 = self.c4_down(c4)
#         c3 = self.c3_down(c3)
#         c2 = self.c2_down(c2)
#         c1 = self.c1_down(c1)
#         return c5, c4, c3, c2, c1
    
# ###############  Cross-View Attention Module  ##################

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# class CAMV(nn.Module):
#     def __init__(self, in_dim, mm_size):
#         super().__init__()
#         self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
#         self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_m = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
#         self.trans = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.trans1 = nn.Sequential(
#             ConvBNReLU(3 * in_dim, 3 * in_dim, 1),
#             ConvBNReLU(3 * in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
       
#         self.transa1 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.transa2 = nn.Sequential(
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#             ConvBNReLU(in_dim, in_dim, 3, 1, 1),
#         )
#         self.mm_size = mm_size
#         self.coe_c_c1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_md = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_md = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_c2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_c1.data.uniform_(-0.5,0.5)
#         self.coe_h_c1.data.uniform_(-0.5,0.5)
#         self.coe_w_c1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_md.data.uniform_(-0.5,0.5)
#         self.coe_h_md.data.uniform_(-0.5,0.5)
#         self.coe_w_md.data.uniform_(-0.5,0.5)
        
#         self.coe_c_c2.data.uniform_(-0.5,0.5)
#         self.coe_h_c2.data.uniform_(-0.5,0.5)
#         self.coe_w_c2.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a1 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a1 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_ma = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_ma = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a2 = nn.Parameter(data=torch.Tensor(1,64), requires_grad=True)
#         self.coe_h_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
#         self.coe_w_a2 = nn.Parameter(data=torch.Tensor(mm_size,mm_size), requires_grad=True)
        
#         self.coe_c_a1.data.uniform_(-0.5,0.5)
#         self.coe_h_a1.data.uniform_(-0.5,0.5)
#         self.coe_w_a1.data.uniform_(-0.5,0.5)
        
#         self.coe_c_ma.data.uniform_(-0.5,0.5)
#         self.coe_h_ma.data.uniform_(-0.5,0.5)
#         self.coe_w_ma.data.uniform_(-0.5,0.5)
        
#         self.coe_c_a2.data.uniform_(-0.5,0.5)
#         self.coe_h_a2.data.uniform_(-0.5,0.5)
#         self.coe_w_a2.data.uniform_(-0.5,0.5)
#         self.channel_attn = ChannelAttention(64)
#         self.spatial_attn = SpatialAttention()
#         self.fuse = nn.Sequential(ConvBNReLU(128, 128, 1),ConvBNReLU(128, 64, 3,1,1),ConvBNReLU(64, 64, 3,1,1))
#     def forward(self, c1, o, c2, a1, a2, return_feats=False):
#         tgt_size = o.shape[2:]
#         c1 = self.conv_l_pre_down(c1)
#         c1 = F.adaptive_max_pool2d(c1, tgt_size) + F.adaptive_avg_pool2d(c1, tgt_size)
#         c1 = self.conv_l_post_down(c1)
#         m = self.conv_m(o)
#         c2 = self.conv_s_pre_up(c2)
#         c2 = F.adaptive_max_pool2d(c2, tgt_size) + F.adaptive_avg_pool2d(c2, tgt_size)
#         c2 = self.conv_s_post_up(c2)
#         attn = self.trans(torch.cat([c1, m, c2], dim=1))
#         attn_c1 = tl.tenalg.mode_dot(attn,self.coe_c_c1,mode=1)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_h_c1,mode=2)
#         attn_c1 = tl.tenalg.mode_dot(attn_c1,self.coe_w_c1,mode=3)
#         attn_c1 = torch.softmax(attn_c1, dim=1)
        
#         attn_md = tl.tenalg.mode_dot(attn,self.coe_c_md,mode=1)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_h_md,mode=2)
#         attn_md = tl.tenalg.mode_dot(attn_md,self.coe_w_md,mode=3)
#         attn_md = torch.softmax(attn_md, dim=1)
        
#         attn_c2 = tl.tenalg.mode_dot(attn,self.coe_c_c2,mode=1)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_h_c2,mode=2)
#         attn_c2 = tl.tenalg.mode_dot(attn_c2,self.coe_w_c2,mode=3)
#         attn_c2 = torch.softmax(attn_c2, dim=1)
        
#         cmc = attn_c1 * c1 + attn_md * m + attn_c2 * c2

#         a1 = self.transa1(a1)
#         a2 = self.transa2(a2)
#         attn1 = self.trans1(torch.cat([a1, m, a2], dim=1))
        
#         attn_a1 = tl.tenalg.mode_dot(attn1,self.coe_c_a1,mode=1)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_h_a1,mode=2)
#         attn_a1 = tl.tenalg.mode_dot(attn_a1,self.coe_w_a1,mode=3)
#         attn_a1 = torch.softmax(attn_a1, dim=1)
        
#         attn_ma = tl.tenalg.mode_dot(attn1,self.coe_c_ma,mode=1)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_h_ma,mode=2)
#         attn_ma = tl.tenalg.mode_dot(attn_ma,self.coe_w_ma,mode=3)
#         attn_ma = torch.softmax(attn_ma, dim=1)
        
#         attn_a2 = tl.tenalg.mode_dot(attn1,self.coe_c_a2,mode=1)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_h_a2,mode=2)
#         attn_a2 = tl.tenalg.mode_dot(attn_a2,self.coe_w_a2,mode=3)
#         attn_a2 = torch.softmax(attn_a2, dim=1)
        
#         ama = attn_a1 * a1 + attn_ma * m + attn_a2 * a2
#         ama = ama.mul(self.channel_attn(ama))
#         ama = ama.mul(self.spatial_attn(ama))
#         lms = self.fuse(torch.cat([ama,cmc],dim=1))
#         return lms
# class Progressive_Iteration(nn.Module):
#     def __init__(self, input_channels):
#         super(Progressive_Iteration, self).__init__()
#         self.input_channels = input_channels
#         self.channels_single = int(input_channels / 4)
#         self.p1_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_channel_reduction = nn.Sequential(
#             nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p1 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p1_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p2 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p2_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p3 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p3_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.p4 = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())
#         self.p4_dc = nn.Sequential(
#             nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
#             nn.BatchNorm2d(self.channels_single), nn.ReLU())

#         self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
#                                     nn.BatchNorm2d(self.input_channels), nn.ReLU())

#     def forward(self, x):
#         p1_input = self.p1_channel_reduction(x)
#         p1 = self.p1(p1_input)
#         p1_dc = self.p1_dc(p1)
#         p2_input = self.p2_channel_reduction(x) + p1_dc
#         p2 = self.p2(p2_input)
#         p2_dc = self.p2_dc(p2)
#         p3_input = self.p3_channel_reduction(x) + p2_dc
#         p3 = self.p3(p3_input)
#         p3_dc = self.p3_dc(p3)

#         p4_input = self.p4_channel_reduction(x) + p3_dc
#         p4 = self.p4(p4_input)
#         p4_dc = self.p4_dc(p4)

#         ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))
#         return ce

# class CFU(nn.Module):
#     def __init__(self, in_c, num_groups=4, hidden_dim=None):
#         super().__init__()
#         self.num_groups = num_groups
#         hidden_dim = hidden_dim or in_c // 2
#         expand_dim = hidden_dim * num_groups
#         self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
#         self.interact = nn.ModuleDict()
#         self.interact["0"] = ConvBNReLU(hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         for group_id in range(1, num_groups - 1):
#             self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)
#         self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 1 * hidden_dim, 3, 1, 1)
#         self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
#         self.final_relu = nn.ReLU(True)
#         self.fp = Progressive_Iteration(192)

#     def forward(self, x):
#         xs = self.expand_conv(x).chunk(self.num_groups, dim=1)
#         outs = []
#         branch_out = self.interact["0"](xs[0])
#         outs.append(branch_out.chunk(2, dim=1))

#         for group_id in range(1, self.num_groups - 1):
#             branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#             outs.append(branch_out.chunk(2, dim=1))

#         group_id = self.num_groups - 1
#         branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
#         outs.append(branch_out.chunk(1, dim=1))
#         out = torch.cat([o[0] for o in outs], dim=1)
#         out = self.fp(out)
#         out = self.fuse(out)
#         return self.final_relu(out + x)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()


# from transformers import AutoModel, AutoImageProcessor

# @MODELS.register()
# class MFFN(BasicModelClass):
#     def __init__(self):
#         super().__init__()
#         # self.shared_encoder = timm.create_model(
#         #     model_name="resnet50", 
#         #     pretrained=True, 
#         #     in_chans=3, 
#         #     features_only=True
#         # )

#         # for p in self.shared_encoder.parameters():
#         #     p.requires_grad = False

#         self.shared_encoder = DinoV2Backbone()

#         self.fpn = SimpleFPN(in_dim=64, out_dim=64)
#         self.translayer = TransLayer(out_c=64, in_dims=[64, 64, 64, 64, 64])

#         # self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
#         dim = [64, 64, 64, 64, 64]
#         # size = [12, 24, 48, 96, 192]
#         size = [16, 32, 64, 128, 256]
#         self.CAMV_layers = nn.ModuleList([CAMV(in_dim=in_c, mm_size=mm_s) for in_c, mm_s in zip(dim, size)])
        
#         self.d5 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d4 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d3 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d2 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.d1 = nn.Sequential(CFU(64, num_groups=6, hidden_dim=32))
#         self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
#         self.out_layer_01 = nn.Conv2d(32, 1, 1)

#     def encoder_translayer(self, x):
#         en_feats = self.shared_encoder(x)
#         en_feats = self.fpn(en_feats)
#         trans_feats = self.translayer(en_feats)
#         return trans_feats

#     def body(self, c1_scale, o_scale, c2_scale, a1_scale, a2_scale):
#         c1_trans_feats = self.encoder_translayer(c1_scale)
#         o_trans_feats = self.encoder_translayer(o_scale)
#         c2_trans_feats = self.encoder_translayer(c2_scale)
#         a1_trans_feats = self.encoder_translayer(a1_scale)
#         a2_trans_feats = self.encoder_translayer(a2_scale)
#         feats = []
#         for c1, o,c2,a1,a2, layer in zip(c1_trans_feats, o_trans_feats, c2_trans_feats, a1_trans_feats, a2_trans_feats, self.CAMV_layers):
#             CAMV_outs = layer(c1=c1, o=o, c2=c2, a1=a1, a2=a2)
#             feats.append(CAMV_outs)

#         x = self.d5(feats[0])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d4(x + feats[1])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d3(x + feats[2])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d2(x + feats[3])
#         x = cus_sample(x, mode="scale", factors=2)
#         x = self.d1(x + feats[4])
#         x = cus_sample(x, mode="scale", factors=2)
#         logits = self.out_layer_01(self.out_layer_00(x))
#         return dict(seg=logits)

#     def train_forward(self, data, **kwargs):
#         assert not {"image_c1", "image_o", "image_c2", "image_a1", "image_a2", "mask"}.difference(set(data)), set(data)
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         loss, loss_str = self.cal_loss(
#             all_preds=output,
#             gts=data["mask"],
#             iter_percentage=kwargs["curr"]["iter_percentage"],
#         )
#         return dict(sal=output["seg"].sigmoid()), loss, loss_str

#     def test_forward(self, data, **kwargs):
#         output = self.body(
#             c1_scale=data["image_c1"],
#             o_scale=data["image_o"],
#             c2_scale=data["image_c2"],
#             a1_scale=data["image_a1"],
#             a2_scale=data["image_a2"],
#         )
#         return output["seg"]

#     def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
#         ual_coef = get_coef(iter_percentage, method)
#         losses = []
#         loss_str = []
#         # for main
#         for name, preds in all_preds.items():
#             resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
#             sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
#             losses.append(sod_loss)
#             loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")
#             ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
#             ual_loss *= ual_coef
#             losses.append(ual_loss)
#             loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
#         return sum(losses), " ".join(loss_str)

#     # def get_grouped_params(self):
#     #     param_groups = {}
#     #     for name, param in self.named_parameters():
#     #         if name.startswith("shared_encoder.layer"):
#     #             param_groups.setdefault("pretrained", []).append(param)
#     #         elif name.startswith("shared_encoder."):
#     #             param_groups.setdefault("fixed", []).append(param)
#     #         else:
#     #             param_groups.setdefault("retrained", []).append(param)
#     #     return param_groups

#     def get_grouped_params(self):
#         params_groups = {
#             "pretrained": [],
#             "retrained": []
#         }

#         for name, param in self.named_parameters():
#             if name.startswith("shared_encoder"):
#                 params_groups["pretrained"].append(param)
#             else:
#                 params_groups["retrained"].append(param)

#         return params_groups


# class DinoV2Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = AutoModel.from_pretrained("facebook/dinov2-small")
#         self.dim = 384 

#         self.reductions = nn.ModuleList([
#             nn.Conv2d(self.dim, 64, 1),
#             nn.Conv2d(self.dim, 64, 1),
#             nn.Conv2d(self.dim, 64, 1),
#             nn.Conv2d(self.dim, 64, 1),
#             nn.Conv2d(self.dim, 64, 1),
#         ])

#     def forward(self, x):
#         B = x.size(0)

#         outputs = []
#         hidden_states = []

#         x = self.model.embeddings(x)
#         hidden_states.append(x)

#         for block in self.model.encoder.layer:
#             x = block(x)[0]
#             hidden_states.append(x)

#         picks = [
#             hidden_states[1],
#             hidden_states[3],
#             hidden_states[6],
#             hidden_states[9],
#             hidden_states[12],
#         ]

#         for i, h in enumerate(picks):
#             h = h[:, 1:, :] 
#             H = W = int((h.size(1)) ** 0.5)
#             h = h.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)

#             h = self.reductions[i](h)

#             outputs.append(h)

#         return outputs


# class SimpleFPN(nn.Module):
#     def __init__(self, in_dim=64, out_dim=64):
#         super().__init__()
#         self.lateral = nn.Conv2d(in_dim, out_dim, 1)

#     def forward(self, xs):
#         x5 = self.lateral(xs[4])
#         x4 = F.interpolate(self.lateral(xs[3]), scale_factor=2)
#         x3 = F.interpolate(self.lateral(xs[2]), scale_factor=4)
#         x2 = F.interpolate(self.lateral(xs[1]), scale_factor=8)
#         x1 = F.interpolate(self.lateral(xs[0]), scale_factor=16)
#         return [x1, x2, x3, x4, x5]