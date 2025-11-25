import torch
from timm.models.vision_transformer import trunc_normal_
from timm.layers import SqueezeExcite
import torch.nn as nn
from timm import models
from timm.layers import SqueezeExcite
import torch
import torch.nn.functional as F
register_model = models.register_model


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))  
        self.bias = nn.Parameter(torch.zeros(num_features))   
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)  
    


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        self.add_module('act', nn.GELU())  
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()       
        w = bn.weight / (bn.running_var + bn.eps)**0.5    # scale = gamma / sqrt(var + eps)
        w = c.weight * w[:, None, None, None]        # W' = W * scale
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5              # b' = beta - (gamma * mu) / sqrt(var + eps)
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)     
        m.bias.data.copy_(b)        
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()           
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])   
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self
    
class DilatedConvDW(nn.Module):
    def __init__(self, ed) :
        super().__init__()
        # 3x3 DW卷积，不同空洞率
        self.conv_dil1 = Conv2d_BN(ed, ed, 3, 1, pad=1, dilation=1, groups=ed)  # Changed padding to pad
        self.conv_dil2 = Conv2d_BN(ed, ed, 3, 1, pad=2, dilation=2, groups=ed)  # Changed padding to pad
        self.conv_dil3 = Conv2d_BN(ed, ed, 3, 1, pad=3, dilation=3, groups=ed)  # Changed padding to pad
        
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        out1 = self.conv_dil1(x)
        out2 = self.conv_dil2(x)
        out3 = self.conv_dil3(x)
        
        
        out = out1 +  out2 + out3 
        
        
        return self.bn(out + x)


class DWBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(DWBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                DilatedConvDW(inp),  # 替换为新的空洞卷积模块
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

class FFN(nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, ed*2)
        self.act = nn.ReLU()
        self.pw2 = Conv2d_BN(ed*2, ed)
        
    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))

class SHDCBlock(nn.Module):
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.pdim = pdim
        self.dim=dim
        
        self.pre_norm = DyT(pdim)
        self.qkv = Conv2d_BN(pdim, qk_dim*2 + pdim, 1)
        self.proj = nn.Sequential(
            nn.ReLU(),
            Conv2d_BN(dim, dim)
        )
        self.SHDCBlock=DWBlock(inp=self.dim-self.pdim,hidden_dim=(self.dim-self.pdim)*2,oup=self.dim-self.pdim,kernel_size=3,stride=1,use_se=True,use_hs=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x1, x2 = x.split([self.pdim, C-self.pdim], dim=1)
        x2=self.SHDCBlock(x2)
        x1 = self.pre_norm(x1)
        q, k, v = self.qkv(x1).split([self.qk_dim, self.qk_dim, self.pdim], 1)
        # [B, qk_dim, H*W] @ [B, H*W, qk_dim] -> [B, qk_dim, qk_dim]
        attn = (q.flatten(2).transpose(1,2) @ k.flatten(2)) * self.scale
        attn = attn.softmax(dim=-1)
        x1 = (v.flatten(2) @ attn.transpose(1,2)).view(B, self.pdim, H, W)

        return self.proj(torch.cat([x1, x2], dim=1))

class BasicBlock(nn.Module):
    def __init__(self, dim, qk_dim, pdim, block_type):
        super().__init__()
        self.conv = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim))
        self.mixer = Residual(SHDCBlock(dim, qk_dim, pdim))if block_type == "s" else nn.Identity()
        self.ffn = Residual(FFN(dim))
        
    def forward(self, x):
        return self.ffn(self.mixer(self.conv(x)))

class PatchMerging(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        hid_dim = in_dim * 4
        self.conv = nn.Sequential(
            Conv2d_BN(in_dim, hid_dim, 1),
            nn.ReLU(),
            Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim),
            SqueezeExcite(hid_dim, 0.25),
            Conv2d_BN(hid_dim, out_dim, 1)
        )
    
    def forward(self, x):
        return self.conv(x)

class DySample(nn.Module):
    def __init__(self, in_channels,way, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.way=way
        if self.way==1:
            self.index=1
        else:
            self.index=in_channels//2
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())
        self.cnn = nn.Conv2d(in_channels, self.index, kernel_size=1, stride=1, padding=0)
      

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h],indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h],indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.cnn(self.forward_lp(x))
    
class DyFusionUp(nn.Module):

    """
    x1为编码器的残差连接部分  :[B,C,H,W]
    x2为上采样的部分: [B,C,H,W]-->[B,C/2,H*2,W*2]
    """
    def __init__(self, in_channels_x1, in_channels_x2,out_channel):
        super(DyFusionUp, self).__init__()
        #assert in_channels_x1*2==in_channels_x2 ,"n_channels_x1*2不等于in_channels_x2"
        self.out_channel=out_channel
        self.dy_sample = DySample(in_channels_x2,way=0)
        self.adjust_channel=nn.Conv2d(in_channels_x2//2,self.out_channel,kernel_size=1,stride=1,padding=0)
        self.cnn_block = DWBlock(inp=in_channels_x1 *2,hidden_dim=in_channels_x1 *2*2,oup=in_channels_x1 *2,kernel_size=3,stride=1,use_se=True,use_hs=False)  
        self.conv3x3 = nn.Conv2d(in_channels_x1 *2, in_channels_x1, kernel_size=3, stride=1, padding=1)
       
    
    def forward(self, x1, x2):
        x2_upsampled = self.dy_sample(x2)
        x2_upsampled=self.adjust_channel(x2_upsampled)

        x_concat = torch.cat((x1, x2_upsampled), dim=1)
        x_concat=self.cnn_block(x_concat)
        out = self.conv3x3(x_concat)
        
        return out
    


class DyGLNet(nn.Module):
    def __init__(self, 
                 embed_dim=[128, 224, 320],
                 partial_dim=[32, 48, 68],
                 qk_dim=[16, 16, 16],
                 depth=[2, 4, 5],
                 types=["i", "s", "s"]):
        super().__init__()
        
        self.patch_embed = nn.Sequential(
            Conv2d_BN(3, 16, 3, 1, 1), nn.ReLU(),
            Conv2d_BN(16, 32, 3, 1, 1), nn.ReLU(),
            Conv2d_BN(32, 64, 3, 2, 1), nn.ReLU(),  
            Conv2d_BN(64, 128, 3, 2, 1)             
        )
        
        self.blocks1 = nn.Sequential(*[
            BasicBlock(embed_dim[0], qk_dim[0], partial_dim[0], types[0])
            for _ in range(depth[0])])
        
        self.blocks2 = nn.Sequential(
            PatchMerging(embed_dim[0], embed_dim[1]),
            *[BasicBlock(embed_dim[1], qk_dim[1], partial_dim[1], types[1])
              for _ in range(depth[1])]
        )
        
        self.blocks3 = nn.Sequential(
            PatchMerging(embed_dim[1], embed_dim[2]),
            *[BasicBlock(embed_dim[2], qk_dim[2], partial_dim[2], types[2])
              for _ in range(depth[2])]
        )

        self.upsample1 = DyFusionUp(embed_dim[1], embed_dim[2], embed_dim[1])
        self.upsample2 = DyFusionUp(embed_dim[0], embed_dim[1], embed_dim[0])
        
        self.upsample3 = DyFusionUp(in_channels_x1=64,  
                                 in_channels_x2=128, 
                                 out_channel=64)
        self.upsample4 = DyFusionUp(in_channels_x1=32,  
                                 in_channels_x2=64,
                                 out_channel=32)
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed[0](x)  
        x = self.patch_embed[1](x)  
        x = self.patch_embed[2](x)  
        x = self.patch_embed[3](x)  
        skip0 = x
        x = self.patch_embed[4](x)  
        x = self.patch_embed[5](x)  
        skip1 = x                  
        
        x = self.patch_embed[6](x)  
        
        x = self.blocks1(x)       
        skip2 = x                  
        x = self.blocks2(x)         
        skip3 = x                  
        x = self.blocks3(x)        
        
        x = self.upsample1(skip3, x)  
        x = self.upsample2(skip2, x)  
        
        x = self.upsample3(skip1, x) 
        x = self.upsample4(skip0, x) 
  
        x = self.final_conv(x)       
        
        return x

    

SHViT_s1_cfg = {
    'embed_dim': [128, 224, 320],
    'partial_dim': [32, 48, 68],
    'qk_dim': [16, 16, 16],
    'depth': [2, 4, 5],
    'types': ["i", "s", "s"]
}

@register_model
def shvit_s1_modified(pretrained=False, **kwargs):
    model = DyGLNet(**SHViT_s1_cfg)
    if pretrained:
        checkpoint = torch.load('Medical_Image_Segmentation/pretrained/shvit_s1.pth', map_location='cuda',weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        
       
        model.load_state_dict(state_dict, strict=False)
    return model

if __name__ == "__main__":
    model = DyGLNet(**SHViT_s1_cfg)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Final output shape: {output.shape}")  
