import sys
from xml.etree.ElementTree import QName
sys.path.append("..")
import torch
import torch.nn as nn
from einops import rearrange
from model.graph_frames import Graph
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    

class linear_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop=0.1):
        super(linear_block,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.GELU(),
            nn.Dropout(drop)
        )
    def forward(self,x):
        x = self.linear(x)
        return x

class encoder(nn.Module): # 2,256,512
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        
        dim_0 = 2
        dim_2 = 64
        dim_3 = 128
        dim_4 = 256
        dim_5 = 512
        
        self.fc1 = nn.Linear(dim_0, dim_2)   
        self.fc3 = nn.Linear(dim_2, dim_3)
        self.fc4 = nn.Linear(dim_3, dim_4)
        self.fc5 = nn.Linear(dim_4, dim_5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
class uMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.linear512_256 = linear_block(512,256,drop)
        self.linear256_256 = linear_block(256,256,drop) 
        self.linear256_512 = linear_block(256,512,drop)

    def forward(self, x):
        # down          
        x = self.linear512_256(x)
        res_256 = x 
        # mid
        x = self.linear256_256(x)
        x = x + res_256
        # up
        x = self.linear256_512(x) 
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 
        self.attn_drop = nn.Dropout(attn_drop) # p=0
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)  # 0

    def forward(self, x, f):
        B, N, C = x.shape # b,j,c
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale # b,heads,17,4 @ b,heads,4,17 = b,heads,17,17
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        f = f.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() # b,j,h,c -> b,h,j,c
        x = (attn @ v)
        attn2gcn = x.clone().permute(0, 2, 1, 3).contiguous().reshape(B, N, C).contiguous()
        x = x + f
        x = x.transpose(1, 2).reshape(B, N, C).contiguous() 
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn2gcn
    
    
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()

        self.adj = adj # 4,17,17
        self.kernel_size = adj.size(0)
        #
        self.conv1d = nn.Conv1d(in_channels, out_channels * self.kernel_size, kernel_size=1)
    def forward(self, x): # b,j,c
        # conv1d
        x = rearrange(x,"b j c -> b c j") 
        x = self.conv1d(x)   # b,c*kernel_size,j = b,c*4,j
        x = rearrange(x,"b ck j -> b ck 1 j")
        b, kc, t, v = x.size()
        x = x.view(b, self.kernel_size, kc//self.kernel_size, t, v) # b,k, kc/k, 1, j 
        x = torch.einsum('bkctv, kvw->bctw', (x, self.adj))   # bctw   b,c,1,j 
        x = x.contiguous()
        x = rearrange(x, 'b c 1 j -> b j c') 
        return x.contiguous()


class Block(nn.Module): # drop=0.1
    def __init__(self, length, dim, adj, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        # length =17, dim = args.channel = 512, tokens_dim = args.token_dim=256, channels_dim = args.d_hid = 1024
        super().__init__()
        
        # GCN
        self.norm1 = norm_layer(length)
        self.GCN_Block1 = GCN(dim, dim, adj)
        self.GCN_Block2 = GCN(dim, dim, adj)

        self.adj = adj
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # attention
        self.norm_att1=norm_layer(dim)
        self.num_heads = 8
        qkv_bias =  True
        qk_scale = None
        attn_drop = 0.2
        proj_drop = 0.25
        self.attn = Attention(
            dim, num_heads=self.num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop) 

        # 512,1024
        self.norm2 = norm_layer(dim)
        self.uMLP = uMLP(in_features=dim, hidden_features=256, act_layer=act_layer, drop=0.20)
        
        gcn2attn_p = 0.15
        Attn2gcn_p = 0.15
        self.gcn2Attn_drop = nn.Dropout(p = gcn2attn_p)
        self.Attn2gcn_drop = nn.Dropout(p = Attn2gcn_p)
        self.s_gcn2attn = nn.Parameter(torch.tensor([(0.5)], dtype=torch.float32), requires_grad=False) 
        self.s_attn2gcn = nn.Parameter(torch.tensor([(0.8)], dtype=torch.float32), requires_grad=False) 

    def forward(self, x):
        # B,J,dim 
        res1 = x # b,j,c
        x_atten = x.clone()
        x_gcn_1 = x.clone()
        # GCN
        x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous() 
        x_gcn_1 = self.norm1(x_gcn_1) # b,c,j
        x_gcn_1 = rearrange(x_gcn_1,"b j c -> b c j").contiguous()
        x_gcn_1 = self.GCN_Block1(x_gcn_1)  # b,j,c

        # Atten
        x_atten = self.norm_att1(x_atten)
        x_atten, attn2gcn = self.attn(x_atten, f= self.gcn2Attn_drop(x_gcn_1*self.s_gcn2attn))
        
        x_gcn_2 = self.GCN_Block2(x_gcn_1 + self.Attn2gcn_drop(attn2gcn*self.s_attn2gcn))  # b, j, c

        x = res1 + self.drop_path(x_gcn_2 + x_atten)

        # uMLP
        res2 = x  # b,j,c
        x = self.norm2(x)
        x =  res2 + self.drop_path(self.uMLP(x))  
        return x
        
class IGANet(nn.Module):
    def __init__(self, depth, embed_dim, adj, drop_rate=0.10, length=27):
        super().__init__()
        
        drop_path_rate = 0.2
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(
                length, embed_dim, adj, 
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.graph = Graph('hm36_gt', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False).cuda(0)

        self.encoder = encoder(2,args.channel//2,args.channel)
        #  
        self.IGANet = IGANet(args.layers, args.channel, self.A, length=args.n_joints) # 256

        self.fcn = nn.Linear(args.channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous() # B 17 2
        
        # encoder
        x = self.encoder(x)     # B 17 512

        x = self.IGANet(x)    # B 17 512
        
        # regression
        x = self.fcn(x)         # B 17 3

        x = rearrange(x, 'b j c -> b 1 j c').contiguous() # B, 1, 17, 3
        return x
