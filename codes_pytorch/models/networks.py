import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import numpy as np

class Operator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super(Operator,self).__init__()
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.encoder = nn.Conv2d(in_nc, base_nf, 1, 1) 
        self.mid_conv = nn.Conv2d(base_nf, base_nf, 1, 1) 
        self.decoder = nn.Conv2d(base_nf, out_nc, 1, 1)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self,x,val):
        x_code = self.encoder(x)
        y_code = x_code + val
        y_code = self.act(self.mid_conv(y_code))
        y = self.decoder(y_code)
        return y


class RendererSingle(nn.Module):
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super().__init__()
        self.renderer = Operator(in_nc,out_nc,base_nf)

    def forward(self, img, value):
        unary = self.renderer(img, 0.0)
        pair = self.renderer(img, value)
        return unary, pair


class Renderer(nn.Module):
    def __init__(self, in_nc=3, out_nc=3,base_nf=64):
        super(Renderer,self).__init__()
        self.in_nc = in_nc
        self.base_nf = base_nf
        self.out_nc = out_nc
        self.ex_block = Operator(in_nc,out_nc,base_nf)
        self.bc_block = Operator(in_nc,out_nc,base_nf)
        self.vb_block = Operator(in_nc,out_nc,base_nf)
    def forward(self,x_ex,x_bc,x_vb,v_ex,v_bc,v_vb):
        
        rec_ex = self.ex_block(x_ex,0)
        rec_bc = self.bc_block(x_bc,0)
        rec_vb = self.vb_block(x_vb,0)

        map_ex = self.ex_block(x_ex,v_ex)
        map_bc = self.bc_block(x_bc,v_bc)
        map_vb = self.vb_block(x_vb,v_vb)

        return rec_ex, rec_bc, rec_vb, map_ex, map_bc, map_vb



class Encoder(nn.Module):
    def __init__(self, in_nc=3, encode_nf=32):
        super(Encoder, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, encode_nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(encode_nf, encode_nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.max = nn.AdaptiveMaxPool2d((1,1))

    def forward(self, x):
        b, _,_,_ = x.size()
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        std, mean = torch.std_mean(conv2_out, dim=[2, 3], keepdim=False)
        maxs = self.max(conv2_out).squeeze(2).squeeze(2)
        out = torch.cat([std, mean, maxs], dim=1)
        return out


class Predictor(nn.Module):
    def __init__(self,fea_dim):
        super(Predictor,self).__init__()
        self.fc3 = nn.Linear(fea_dim,1)
        self.tanh = nn.Tanh()
    def forward(self,img_fea):
        val = self.tanh(self.fc3(img_fea))
        return val    

    
class NeurOP(nn.Module):
    def __init__(self, in_nc=3, out_nc = 3, base_nf = 64, encode_nf =32 , load_path = None):
        super(NeurOP,self).__init__()
        self.fea_dim = encode_nf * 3
        self.image_encoder = Encoder(in_nc,encode_nf)
        renderer = Renderer(in_nc,out_nc,base_nf)
        if load_path is not None: 
            renderer.load_state_dict(torch.load(load_path))
            
        self.bc_renderer = renderer.bc_block
        self.bc_predictor =  Predictor(self.fea_dim)
        
        self.ex_renderer = renderer.ex_block
        self.ex_predictor =  Predictor(self.fea_dim)
        
        self.vb_renderer = renderer.vb_block
        self.vb_predictor =  Predictor(self.fea_dim)

        self.renderers = [self.bc_renderer,self.ex_renderer,self.vb_renderer]
        self.predict_heads = [self.bc_predictor,self.ex_predictor,self.vb_predictor]
            
    def render(self,x,vals):
        b,_,h,w = img.shape
        imgs = []
        for nop, scalar in zip(self.renderers,vals):
            img = nop(img,scalar)
            output_img = torch.clamp(img, 0, 1.0)
            imgs.append(output_img)
        return imgs
    
    def forward(self,img, return_vals = True):
        b,_,h,w = img.shape
        vals = []
        for nop, predict_head in zip(self.renderers,self.predict_heads):
            img_resized = F.interpolate(input=img, size=(256, int(256*w/h)), mode='bilinear', align_corners=False)
            feat = self.image_encoder(img_resized)
            scalar = predict_head(feat)
            vals.append(scalar)
            img = nop(img,scalar)
        img = torch.clamp(img, 0, 1.0)
        if return_vals:
            return img,vals
        else:
            return img