import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ERGAS
def ERGAS(true, pred, scale=4):
    diff = true - pred
    rmse = torch.sqrt(torch.mean(diff ** 2, dim=(1, 2, 3)))
    mean_true = torch.mean(true, dim=(1, 2, 3))
    ergas = 100 * scale * torch.sqrt(torch.mean((rmse / mean_true) ** 2))
    return ergas.item()

# SSIM
def SSIM(true, pred):
    true_np = true.permute(0, 2, 3, 1).cpu().numpy()
    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
    ssim_value = ssim(true_np, pred_np, multichannel=True, data_range=true_np.max() - true_np.min())
    return ssim_value

# PSNR
def PSNR(true, pred):
    true_np = true.permute(0, 2, 3, 1).cpu().numpy()
    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
    psnr_value = psnr(true_np, pred_np, data_range=true_np.max() - true_np.min())
    return psnr_value

# Q4
def Q4(true, pred):
    true_np = true.permute(0, 2, 3, 1).cpu().numpy()
    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
    u_true = np.mean(true_np, axis=(0, 1))
    u_pred = np.mean(pred_np, axis=(0, 1))
    cov_true = np.cov(true_np.reshape(-1, true_np.shape[-1]).T)
    cov_pred = np.cov(pred_np.reshape(-1, pred_np.shape[-1]).T)
    cov_comb = np.cov(np.concatenate([true_np, pred_np], axis=0).reshape(-1, true_np.shape[-1]).T)
    q4_value = 4 * np.linalg.det(cov_comb) / (np.linalg.det(cov_true) + np.linalg.det(cov_pred)) ** 2
    return q4_value

# SAM
def SAM(true, pred):
    true_flat = true.view(true.size(0), -1)
    pred_flat = pred.view(pred.size(0), -1)
    dot_product = torch.sum(true_flat * pred_flat, dim=1)
    norm_true = torch.norm(true_flat, dim=1)
    norm_pred = torch.norm(pred_flat, dim=1)
    sam_value = torch.acos(dot_product / (norm_true * norm_pred)).mean().item()
    return sam_value

# D_lambda
def D_lambda(true, pred):
    true_np = true.permute(0, 2, 3, 1).cpu().numpy()
    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()

    mean_true = np.mean(true_np, axis=(1, 2), keepdims=True)
    mean_pred = np.mean(pred_np, axis=(1, 2), keepdims=True)

    var_true = np.var(true_np, axis=(1, 2), keepdims=True)
    var_pred = np.var(pred_np, axis=(1, 2), keepdims=True)

    cov_true = np.mean((true_np - mean_true) * (true_np - mean_true), axis=(1, 2))
    cov_pred = np.mean((pred_np - mean_pred) * (pred_np - mean_pred), axis=(1, 2))

    d_lambda = np.mean(np.abs(cov_true - cov_pred) / (cov_true + cov_pred))
    return d_lambda

# D_s
def D_s(true, pred):
    true_np = true.permute(0, 2, 3, 1).cpu().numpy()
    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()

    ssim_values = [ssim(true_np[i], pred_np[i], multichannel=True) for i in range(true_np.shape[0])]
    mean_ssim = np.mean(ssim_values)
    d_s = 1 - mean_ssim
    return d_s

# QNR
def QNR(true, pred, alpha=1, beta=1):
    d_lambda_value = D_lambda(true, pred)
    d_s_value = D_s(true, pred)
    qnr_value = (1 - d_lambda_value) ** alpha * (1 - d_s_value) ** beta
    return qnr_value, d_lambda_value, d_s_value

