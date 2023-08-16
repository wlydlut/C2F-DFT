import os
from metrics import calculate_psnr, calculate_ssim
import torch
import cv2
import lpips
import numpy as np
device = torch.device('cuda')
gt_path = './Datasets/test/GoPro/target/'
results_path = './results_fine/GoPro/'

lpips_fn = lpips.LPIPS(net='alex').to(device)   ###########LPIPS

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))

assert len(imgsName) == len(gtsName)

cumulative_psnr, cumulative_ssim, cumulative_lpips = 0, 0, 0

for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)
    cur_psnr = calculate_psnr(res, gt, test_y_channel=False)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=False)

    gt_tensor = torch.tensor(np.array(gt)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    res_tensor = torch.tensor(np.array(res)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    lpips = lpips_fn(gt_tensor.to(device) * 2 - 1, res_tensor.to(device) * 2 - 1).squeeze().item()

    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    print('LPIPS is %.4f ' % (lpips))

    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
    cumulative_lpips += lpips

print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / len(imgsName), cumulative_ssim / len(imgsName)))
print('Testing set, LPIPS is %.4f' % (cumulative_lpips / len(imgsName)))
print(results_path)
