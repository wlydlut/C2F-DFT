import sys
sys.path.append("/data/C2F-DFT/")

import numpy as np
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import utils
import h5py
from basicsr.models.archs.DFT_arch import DFT
from skimage import img_as_ubyte
from basicsr.utils import set_random_seed
import scipy.io as sio
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'


def overlapping_grid_indices(x_cond):
    _, c, h, w = x_cond.shape

    h_list = [h]

    w_list = [w]

    return h_list, w_list

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps_overlapping(input_, model_restoration, device, betas, seq, seq_next, eta):

    x = torch.randn(input_.size(), device=device)
    n = input_.size(0)
    x0_preds = []
    xs = [x]

    for k, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * k).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())
        xt = xs[-1].to('cuda')

        et = model_restoration(torch.cat([data_transform(input_), xt], dim=1), t)


        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.to('cpu'))

        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        xs.append(xt_next.to('cpu'))

    return xs, x0_preds


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Real Image Denoising using C2F-DFT')

    parser.add_argument('--input_dir', default='./Datasets/test/DND/', type=str, help='Directory of validation images')

    parser.add_argument('--result_dir', default='./results_fine/Real_Denoising/DND/', type=str,
                        help='Directory for results')
    parser.add_argument('--weights', default='./pretrained_models/net_g_denoised_fine.pth', type=str, help='Path to weights')

    parser.add_argument('--save_images', default=True, action='store_true',
                        help='Save denoised images in result directory')

    args = parser.parse_args()

    ####### Load yaml #######
    yaml_file = 'Options/RealDenoising_C2F-DFT_Fine.yml'
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    opt = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
    # random seed
    seed = opt.get('manual_seed')
    set_random_seed(seed)

    s = opt['network_g'].pop('type')
    ##########################

    result_dir_mat = os.path.join(args.result_dir, 'mat')
    os.makedirs(result_dir_mat, exist_ok=True)

    if args.save_images:
        result_dir_png = os.path.join(args.result_dir, 'png')
        os.makedirs(result_dir_png, exist_ok=True)

    model_restoration = DFT(**opt['network_g'])
    checkpoint = torch.load(args.weights)
    model_restoration.load_state_dict(checkpoint['params'])
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    betas = get_beta_schedule(
        beta_schedule=opt['diffusion'].get('beta_schedule'),
        beta_start=opt['diffusion'].get('beta_start'),
        beta_end=opt['diffusion'].get('beta_end'),
        num_diffusion_timesteps=opt['diffusion'].get('num_diffusion_timesteps')
    )
    betas = torch.from_numpy(betas).float().to(device)

    skip = opt['diffusion'].get('num_diffusion_timesteps') // opt['diffusion'].get('sampling_timesteps')
    seq = range(0, opt['diffusion'].get('num_diffusion_timesteps'), skip)
    eta = 0.
    seq_next = [-1] + list(seq[:-1])

    israw = False
    eval_version = "1.0"

    # Load info
    infos = h5py.File(os.path.join(args.input_dir, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']

    # Process data
    with torch.no_grad():
        for i in tqdm(range(50)):
            Idenoised = np.zeros((20,), dtype=np.object)
            filename = '%04d.mat' % (i + 1)
            filepath = os.path.join(args.input_dir, 'images_srgb', filename)
            img = h5py.File(filepath, 'r')
            Inoisy = np.float32(np.array(img['InoisySRGB']).T)

            # bounding box
            ref = bb[0][i]
            boxes = np.array(info[ref]).T

            for k in range(20):
                idx = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
                noisy_patch = torch.from_numpy(Inoisy[idx[0]:idx[1], idx[2]:idx[3], :]).unsqueeze(0).permute(0, 3, 1,
                                                                                                             2).cuda()

                xs, _ = generalized_steps_overlapping(noisy_patch, model_restoration, device, betas, seq, seq_next, eta)

                restored_patch = inverse_data_transform(xs[-1])

                restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                Idenoised[k] = restored_patch

                if args.save_images:
                    save_file = os.path.join(result_dir_png, '%04d_%02d.png' % (i + 1, k + 1))
                    denoised_img = img_as_ubyte(restored_patch)
                    utils.save_img(save_file, denoised_img)

            # save denoised data
            sio.savemat(os.path.join(result_dir_mat, filename),
                        {"Idenoised": Idenoised,
                         "israw": israw,
                         "eval_version": eval_version},
                        )
