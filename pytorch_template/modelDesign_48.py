# =======================================================================================================================
# =======================================================================================================================
"""在modelDesign_*.py加载自定义的其他模型时，请在modelDesign_*.py中使用如下代码获取模型路径："""
import os
import sys

# 获取当前文件所在的路径部分
CUR_DIRNAME = os.path.dirname(__file__)
# 你自定义的模型名称
YOUR_MODEL_NAME = 'your_model.pth.tar'  # Pytorch模型
# YOUR_MODEL_NAME = 'your_model.h5'   # TensorFlow模型
# 拼接为你自定义模型的完整路径
YOUR_MODEL_PATH = os.path.join(CUR_DIRNAME, YOUR_MODEL_NAME)

# =======================================================================================================================
# =======================================================================================================================
import numpy as np
import torch.nn as nn
import torch
# import torch.nn.functional as F
from torch.utils.data import Dataset


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pres, labels):
        pres = pres.view(-1, 12, 32, 2)
        labels = labels.view(-1, 12, 32, 2)
        pa = pres[:, :, :, 0]
        pb = pres[:, :, :, 1]
        a = labels[:, :, :, 0]
        b = labels[:, :, :, 1]
        loss = (a * pa + b * pb).sum(2).pow(2) + (b * pa - a * pb).sum(2).pow(2)
        # loss_s = (a.pow(2) + b.pow(2)).sum(2) * (pa.pow(2) + pb.pow(2)).sum(2) + self.eps
        loss_s = (a.pow(2) + b.pow(2)).sum(2) + self.eps
        loss = loss / loss_s
        loss = 1 - loss.mean(1).mean(0)
        return loss


# =======================================================================================================================
# =======================================================================================================================
# Number to Bit Defining Function Defining
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


# =======================================================================================================================
# =======================================================================================================================
# Quantization and Dequantization Layers Defining
class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):
    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):
    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out


# =======================================================================================================================
# =======================================================================================================================
# Encoder and Decoder Class Defining
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class Encoder(nn.Module):
    num_quan_bits = 2

    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, int(feedback_bits / self.num_quan_bits))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.num_quan_bits)

    def forward(self, x,quant=True):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.sig(out)
        if quant:
            out = self.quantize(out)
        return out


class Decoder(nn.Module):
    num_quan_bits = 2

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.num_quan_bits)
        self.fc = nn.Linear(int(feedback_bits / self.num_quan_bits), 768)
        self.out_cov = conv3x3(2, 32)
        self.out_cov2 = conv3x3(32, 2)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, quant=True):
        if quant:
            out = self.dequantize(x)
        else:
            out = x
        out = out.view(-1, int(self.feedback_bits / self.num_quan_bits))
        out = self.sig(self.fc(out))
        out = out.view(-1, 2, 12, 32)
        out = self.out_cov(out)
        out = self.relu(out)
        out = self.out_cov2(out)
        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(-1, 768)
        return out


class AutoEncoder(nn.Module):
    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x, quant=True):
        feature = self.encoder(x, quant)
        out = self.decoder(feature, quant)
        return out


# =======================================================================================================================
# =======================================================================================================================
# Testing Function Defining
def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1 * num2))
    return cos


def cal_score(w_true, w_pre, NUM_SAMPLES, NUM_SUBBAND):
    img_total = 64
    num_sample_subband = NUM_SAMPLES * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_subband, img_total])
    W_pre = np.reshape(w_pre, [num_sample_subband, img_total])
    W_true2 = W_true[0:num_sample_subband, 0:int(img_total):2] + 1j * W_true[0:num_sample_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_subband, 0:int(img_total):2] + 1j * W_pre[0:num_sample_subband, 1:int(img_total):2]
    score_cos = 0
    for i in range(num_sample_subband):
        W_true2_sample = W_true2[i:i + 1, ]
        W_pre2_sample = W_pre2[i:i + 1, ]
        score_tmp = cos_sim(W_true2_sample, W_pre2_sample)
        score_cos = score_cos + abs(score_tmp) * abs(score_tmp)
    score_cos = score_cos / num_sample_subband
    return score_cos


# =======================================================================================================================
# =======================================================================================================================
# Data Loader Class Defining
class DatasetFolder(Dataset):
    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]
