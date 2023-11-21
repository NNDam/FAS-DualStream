import torch
import copy
import utils
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class CrossAttensionFusion(torch.nn.Module):
    def __init__(self, embed_size = 384, hidden_size = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.q = torch.nn.Linear(embed_size, embed_size)
        self.k = torch.nn.Linear(embed_size, embed_size)
        self.v = torch.nn.Linear(embed_size, embed_size)
        self.q_bpf = torch.nn.Linear(embed_size, embed_size)
        self.k_bpf = torch.nn.Linear(embed_size, embed_size)
        self.v_bpf = torch.nn.Linear(embed_size, embed_size)
        self.proj_out = torch.nn.Linear(2*embed_size, hidden_size)
        self.norm     = Normalize(embed_size)
        self.norm_bpf = Normalize(embed_size)

    def forward(self, x, x_bpf):
        B, _ = x.shape
        scale = int(self.embed_size)**(-0.5)

        # Normal branch
        h = x
        h = self.norm(h)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # BPF branch
        h_bpf = x_bpf
        h_bpf = self.norm_bpf(h_bpf)
        q_bpf = self.q_bpf(h_bpf)
        k_bpf = self.k_bpf(h_bpf)
        v_bpf = self.v_bpf(h_bpf)

        # Compute attention for normal branch
        q_bpf.mul_(scale)
        q_bpf = q_bpf.reshape((B, self.embed_size, 1))
        k     = k.reshape((B, 1, self.embed_size))
        v     = v.reshape((B, self.embed_size, 1))
        w = torch.matmul(q_bpf, k) # B x embed_size x embed_size
        w = F.softmax(w, dim=2)    # B x embed_size x embed_size
        f = w.matmul(v)            # B x embed_size x 1
        f = f.view(B, self.embed_size) + x

        # Compute attention for bpf branch
        q.mul_(scale)
        q      = q.reshape((B, self.embed_size, 1))
        k_bpf  = k_bpf.reshape((B, 1, self.embed_size))
        v_bpf  = v_bpf.reshape((B, self.embed_size, 1))
        w_bpf = torch.matmul(q, k_bpf) # B x embed_size x embed_size
        w_bpf = F.softmax(w_bpf, dim=2)    # B x embed_size x embed_size
        f_bpf = w_bpf.matmul(v_bpf)            # B x embed_size x 1
        f_bpf = f_bpf.view(B, self.embed_size) + x_bpf

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
        else:
            raise NotImplementedError("Architecture not support: {}".format(arch))
        
        self.head = CrossAttensionFusion(embed_size = feature_dim, hidden_size = feature_dim)
        self.fc_fused   = torch.nn.Linear(feature_dim, 1)
        self.fc_org     = torch.nn.Linear(feature_dim, 1)
        self.fc_bpf     = torch.nn.Linear(feature_dim, 1)

    def forward(self, x, x_bpf):
        x       = self.backbone(x)
        x_bpf   = self.backbone_bpf(x_bpf)
        x_fused = self.head(x, x_bpf)
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