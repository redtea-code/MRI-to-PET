import math
import numpy as np

import cv2

from scipy import ndimage


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim_3d(img1, img2, data_range=255):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 生成一维高斯核
    kernel_1d = cv2.getGaussianKernel(11, 1.5).ravel()

    # 可分离的三维高斯卷积
    def convolve_gaussian_3d(image):
        # 沿着三个轴依次进行一维卷积
        temp = ndimage.convolve1d(image, kernel_1d, axis=0, mode='reflect')
        temp = ndimage.convolve1d(temp, kernel_1d, axis=1, mode='reflect')
        temp = ndimage.convolve1d(temp, kernel_1d, axis=2, mode='reflect')
        return temp

    # 计算均值mu
    mu1 = convolve_gaussian_3d(img1)
    mu2 = convolve_gaussian_3d(img2)

    # 裁剪有效区域
    mu1 = mu1[...,5:-5, 5:-5, 5:-5]
    mu2 = mu2[...,5:-5, 5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = convolve_gaussian_3d(img1 ** 2)[...,5:-5, 5:-5, 5:-5] - mu1_sq
    sigma2_sq = convolve_gaussian_3d(img2 ** 2)[...,5:-5, 5:-5, 5:-5] - mu2_sq
    sigma12 = convolve_gaussian_3d(img1 * img2)[...,5:-5, 5:-5, 5:-5] - mu1_mu2

    # 计算SSIM映射
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    return ssim_map.mean()


def calculate_psnr_3d(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # 创建适用于任意维度的切片对象
    slices = tuple(slice(border, dim - border) for dim in img1.shape)
    img1 = img1[slices]
    img2 = img2[slices]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


if __name__ == '__main__':
    import torch

    x1 = torch.rand(8, 1, 64, 64, 16).numpy()
    x2 = torch.rand(8, 1, 64, 64, 16).numpy()
    ssim = ssim_3d(x1, x2,1)
    psnr = calculate_psnr_3d(x1, x2)
    print(ssim, psnr)
