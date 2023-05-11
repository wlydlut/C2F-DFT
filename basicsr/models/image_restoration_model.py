import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


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


class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        betas = get_beta_schedule(
            beta_schedule=self.opt['diffusion'].get('beta_schedule'),
            beta_start=self.opt['diffusion'].get('beta_start'),
            beta_end=self.opt['diffusion'].get('beta_end'),
            num_diffusion_timesteps=self.opt['diffusion'].get('num_diffusion_timesteps')
        )

        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        ###################Coarse training Pipeline
        # if train_opt.get('pixel_opt'):
        #     pixel_type = train_opt['pixel_opt'].pop('type')
        #     cri_pix_cls = getattr(loss_module, pixel_type)
        #     self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
        #         self.device)

        ###################Fine training Pipeline
        if train_opt.get('pixel_opt1'):
            pixel_type1 = train_opt['pixel_opt1'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type1)
            self.cri_pix1 = cri_pix_cls(**train_opt['pixel_opt1']).to(
                self.device)
            pixel_type2 = train_opt['pixel_opt2'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type2)
            self.cri_pix2 = cri_pix_cls(**train_opt['pixel_opt2']).to(
                self.device)
        ######################################
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params, **train_opt['optim_g'])


        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def data_transform(self, X):
        return 2 * X - 1.0

    ###################Coarse training Pipeline
    # def optimize_parameters(self, current_iter):
    #     self.optimizer_g.zero_grad()
    #
    #     self.lq = self.data_transform(self.lq)
    #     self.gt = self.data_transform(self.gt)
    #
    #     n = self.lq.size(0)
    #     e = torch.randn_like(self.lq)
    #     b = self.betas
    #
    #     # antithetic sampling
    #     t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
    #     t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
    #
    #
    #     a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    #     x = self.gt * a.sqrt() + e * (1.0 - a).sqrt()
    #
    #     preds = self.net_g(torch.cat([self.lq, x], dim=1), t.float())
    #
    #     if not isinstance(preds, list):
    #         preds = [preds]
    #
    #     self.output = preds[-1]
    #
    #     loss_dict = OrderedDict()
    #     # pixel loss
    #     l_pix = 0.
    #     for pred in preds:
    #         l_pix += self.cri_pix(pred, e)
    #
    #     loss_dict['l_pix'] = l_pix
    #
    #     l_pix.backward()
    #     if self.opt['train']['use_grad_clip']:
    #         torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
    #     self.optimizer_g.step()
    #
    #     self.log_dict = self.reduce_loss_dict(loss_dict)
    #
    #     if self.ema_decay > 0:
    #         self.model_ema(decay=self.ema_decay)

    ###################Fine training Pipeline
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.lq = self.data_transform(self.lq)
        self.gt = self.data_transform(self.gt)
        n = self.lq.size(0)
        x = torch.randn_like(self.lq)
        eta = 0.0
        # antithetic sampling
        skip = self.opt['diffusion'].get('num_diffusion_timesteps') // self.opt['diffusion'].get('sampling_timesteps')
        seq = range(0, self.opt['diffusion'].get('num_diffusion_timesteps'), skip)
        seq_next = [-1] + list(seq[:-1])
        preds = []
        xs = [x]

        l_pix = 0.

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(self.lq.device)

            next_t = (torch.ones(n) * j).to(self.lq.device)
            at = self.compute_alpha(self.betas, t.long())
            at_next = self.compute_alpha(self.betas, next_t.long())
            xt = xs[-1]
            et = self.net_g(torch.cat([self.lq, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            preds.append(x0_t)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

        loss_dict = OrderedDict()

        # pixel loss
        l_pix = 0.84 * (1 - self.cri_pix1(xt_next, self.gt)) + self.cri_pix2(xt_next, self.gt)  # ssim+l1

        loss_dict['l_pix'] = l_pix
        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def inverse_data_transform(self, X):
        return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def generalized_steps(self, img, model, eta=0.):
        n = img.size(0)
        x_cond = self.data_transform(img)
        x = torch.randn_like(x_cond)

        skip = self.opt['diffusion'].get('num_diffusion_timesteps') // self.opt['diffusion'].get('sampling_timesteps')
        seq = range(0, self.opt['diffusion'].get('num_diffusion_timesteps'), skip)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(self.betas, t.long())
            at_next = self.compute_alpha(self.betas, next_t.long())
            xt = xs[-1]
            et = model(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t)
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)
        return xs, x0_preds

    def test_val(self, img=None):
        if img is None:
            img = self.lq
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                xs, _ = self.generalized_steps(img, self.net_g_ema, eta=0.)
            self.output = self.inverse_data_transform(xs[-1])
        else:
            self.net_g.eval()
            with torch.no_grad():
                xs, _ = self.generalized_steps(img, self.net_g, eta=0.)
            self.output = self.inverse_data_transform(xs[-1])
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            self.lq = self.lq.flatten(start_dim=0, end_dim=1) if self.lq.ndim == 5 else self.lq
            self.gt = self.gt.flatten(start_dim=0, end_dim=1) if self.gt.ndim == 5 else self.gt

            self.test_val()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')

                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')

                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
