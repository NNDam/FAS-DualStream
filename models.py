import torch
import copy
import utils
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# class CrossAttensionFusion1D(torch.nn.Module):
#     def __init__(self, embed_size = 384, hidden_size = 512):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.embed_size = embed_size
#         self.q = torch.nn.Linear(embed_size, embed_size)
#         self.k = torch.nn.Linear(embed_size, embed_size)
#         self.v = torch.nn.Linear(embed_size, embed_size)
#         self.q_bpf = torch.nn.Linear(embed_size, embed_size)
#         self.k_bpf = torch.nn.Linear(embed_size, embed_size)
#         self.v_bpf = torch.nn.Linear(embed_size, embed_size)
#         self.proj_out = torch.nn.Linear(2*embed_size, hidden_size)
#         self.norm     = Normalize(embed_size)
#         self.norm_bpf = Normalize(embed_size)

#     def forward(self, x, x_bpf):
#         B, _ = x.shape
#         scale = int(self.embed_size)**(-0.5)

#         # Normal branch
#         h = x
#         h = self.norm(h)
#         q = self.q(h)
#         k = self.k(h)
#         v = self.v(h)

#         # BPF branch
#         h_bpf = x_bpf
#         h_bpf = self.norm_bpf(h_bpf)
#         q_bpf = self.q_bpf(h_bpf)
#         k_bpf = self.k_bpf(h_bpf)
#         v_bpf = self.v_bpf(h_bpf)

#         # Compute attention for normal branch
#         q_bpf.mul_(scale)
#         q_bpf = q_bpf.reshape((B, self.embed_size, 1))
#         k     = k.reshape((B, 1, self.embed_size))
#         v     = v.reshape((B, self.embed_size, 1))
#         w = torch.matmul(q_bpf, k) # B x embed_size x embed_size
#         w = F.softmax(w, dim=2)    # B x embed_size x embed_size
#         f = w.matmul(v)            # B x embed_size x 1
#         f = f.view(B, self.embed_size) + x

#         # Compute attention for bpf branch
#         q.mul_(scale)
#         q      = q.reshape((B, self.embed_size, 1))
#         k_bpf  = k_bpf.reshape((B, 1, self.embed_size))
#         v_bpf  = v_bpf.reshape((B, self.embed_size, 1))
#         w_bpf = torch.matmul(q, k_bpf) # B x embed_size x embed_size
#         w_bpf = F.softmax(w_bpf, dim=2)    # B x embed_size x embed_size
#         f_bpf = w_bpf.matmul(v_bpf)            # B x embed_size x 1
#         f_bpf = f_bpf.view(B, self.embed_size) + x_bpf

#         # Concat
#         fused = torch.cat([f, f_bpf], 1)
#         out = self.proj_out(fused)
#         return out

class CrossAttensionFusion2D(torch.nn.Module):
    def __init__(self, embed_size = 384, hidden_size = 512, n_head = 32):
        """
            embed_size: channel of previous layer
            hidden_size: channel of output feature map
            n_head: number of attention head
        """
        assert(embed_size % n_head == 0), 'The size of head should be divided by the number of channels.'
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_head = 32
        self.attn_size = self.embed_size // self.n_head
        self.q = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.q_bpf = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.k_bpf = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.v_bpf = torch.nn.Conv2d(embed_size, embed_size, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(2*embed_size, hidden_size, kernel_size=1, stride=1, padding=0)


    def forward(self, x, x_bpf):
        B, C, H, W = x.shape
        scale = int(self.attn_size)**(-0.5)

        # Normal branch
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # BPF branch
        q_bpf = self.q_bpf(x_bpf)
        k_bpf = self.k_bpf(x_bpf)
        v_bpf = self.v_bpf(x_bpf)

        # Compute attention for normal branch
        q_bpf.mul_(scale)
        q_bpf = q_bpf.reshape((B, self.n_head, self.attn_size, H*W))
        q_bpf = q_bpf.permute(0, 3, 1, 2) # b, hw, head, att
        k     = k.reshape((B, self.n_head, self.attn_size, H*W))
        k     = k.permute(0, 3, 1, 2) # b, hw, head, att
        v     = v.reshape((B, self.n_head, self.attn_size, H*W))
        v     = v.permute(0, 3, 1, 2) # b, hw, head, att

        q_bpf = q_bpf.transpose(1, 2) # b, head, hw, att
        v     = v.transpose(1, 2) # b, head, hw, att
        k     = k.transpose(1, 2).transpose(2,3) # b, head, att, hw

        w = torch.matmul(q_bpf, k) # b, head, hw, hw
        w = F.softmax(w, dim=3)    # b, head, hw, hw
        f = w.matmul(v)            # b, head, hw, att
        f = f.transpose(1, 2).contiguous() # b, hw, head, att
        f = f.view(B, H, W, -1) # b, h, w, head*att
        f = f.permute(0, 3, 1, 2) # b, head*att, h, w
        f = f + x

        # Compute attention for bpf branch
        q.mul_(scale)
        q     = q.reshape((B, self.n_head, self.attn_size, H*W))
        q     = q.permute(0, 3, 1, 2) # b, hw, head, att
        k_bpf = k_bpf.reshape((B, self.n_head, self.attn_size, H*W))
        k_bpf = k_bpf.permute(0, 3, 1, 2) # b, hw, head, att
        v_bpf = v_bpf.reshape((B, self.n_head, self.attn_size, H*W))
        v_bpf = v_bpf.permute(0, 3, 1, 2) # b, hw, head, att

        q     = q.transpose(1, 2) # b, head, hw, att
        v_bpf = v_bpf.transpose(1, 2) # b, head, hw, att
        k_bpf = k_bpf.transpose(1, 2).transpose(2,3) # b, head, attn, hw

        w_bpf = torch.matmul(q, k_bpf) 
        w_bpf = F.softmax(w_bpf, dim=3)  
        f_bpf = w_bpf.matmul(v_bpf)          
        f_bpf = f_bpf.transpose(1, 2).contiguous() 
        f_bpf = f_bpf.view(B, H, W, -1) 
        f_bpf = f_bpf.permute(0, 3, 1, 2)
        f_bpf = f_bpf + x_bpf

        # Concat
        fused = torch.cat([f, f_bpf], 1)
        out = self.proj_out(fused)
        return out


class FaceEncoder(torch.nn.Module):
    def __init__(self, arch, pretrained_path, image_size = 112, feature_dim = 384):
        super().__init__()

        # self.model, self.train_preprocess, self.val_preprocess = clip.load(
        #     args.model, args.device, jit=False)
        print('  - Create face encoder model from pretrained {}: {}'.format(arch, pretrained_path))
        self.arch = arch
        self.image_size = image_size

        if self.arch == 'dinov2_vits14':
            self.backbone     = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.backbone_bpf = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if self.arch == 'r18_imagenet':
            from torchvision import datasets, transforms, models
            self.backbone     = models.resnet18(pretrained=True)
            self.backbone_bpf = models.resnet18(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-2]))
            self.backbone_bpf = torch.nn.Sequential(*(list(self.backbone_bpf.children())[:-2]))
        else:
            raise NotImplementedError("Architecture not support: {}".format(arch))
        
        self.head = CrossAttensionFusion2D(embed_size = feature_dim, hidden_size = feature_dim)
        self.fc_fused   = torch.nn.Linear(feature_dim, 1)
        self.fc_org     = torch.nn.Linear(feature_dim, 1)
        self.fc_bpf     = torch.nn.Linear(feature_dim, 1)

    def forward(self, x, x_bpf):
        x       = self.backbone(x)
        x_bpf   = self.backbone_bpf(x_bpf)
        x_fused = self.head(x, x_bpf)
        x       = torch.mean(x, [2, 3])
        x_bpf   = torch.mean(x_bpf, [2, 3])
        x_fused = torch.mean(x_fused, [2, 3])
        x       = self.fc_org(x)
        x_bpf   = self.fc_bpf(x_bpf)
        x_fused = self.fc_fused(x_fused)
        return x_fused, x, x_bpf

    def save(self, filename):
        # print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)