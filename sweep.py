from argparse import Namespace
from fit_single_img_flame import fit
from Flame.utils.config import cfg
from core.options import ImageFittingOptions

import torch
import cv2
import numpy as np
import os
import glob
from kornia.color import rgb_to_lab
import PIL
from torch.functional import F

import wandb


    


def bench(config):
    # args = Namespace(cache_folder="fitting_cache", 
    #              device="cuda:0",
    #              exp_reg_w=0.001, 
    #              first_nrf_iters=1000, 
    #              first_rf_iters=1000,
    #              gpu=0,
    #              id_reg_w=0.05, 
    #              img_path="benchmarks/FAIR_benchmark/validation_set/", 
    #              lm_loss_w=5000.0, 
    #              nframes_shape=16, 
    #              nrf_lr=0.01, 
    #              padding_ratio=0.3, 
    #              perceptual_w=1.0, 
    #              recon_model="bfm09", 
    #              reni_reg_w=0.03, 
    #              res_folder="results", 
    #              rest_nrf_iters=30, 
    #              rest_rf_iters=50, 
    #              rf_lr=0.01, 
    #              rgb_loss_w=0.001, 
    #              rot_reg_w=1, 
    #              tar_size=224, 
    #              tex_reg_w=0.00017, 
    #              tex_w=100.0, 
    #              trans_reg_w=1.0)
    args = Namespace(cache_folder="fitting_cache", 
                 device="cuda:0",
                 exp_reg_w=config.exp_reg_w, 
                 first_nrf_iters=1000, 
                 first_rf_iters=1000,
                 gpu=0,
                 id_reg_w=config.id_reg_w, 
                 img_path="benchmarks/FAIR_benchmark/validation_set/", 
                 lm_loss_w=config.lm_loss_w, 
                 nframes_shape=16, 
                 nrf_lr=0.01, 
                 padding_ratio=0.3, 
                 perceptual_w=1.0, 
                 recon_model="bfm09", 
                 reni_reg_w=config.reni_reg_w, 
                 res_folder="results", 
                 rest_nrf_iters=30, 
                 rest_rf_iters=50, 
                 rf_lr=0.01, 
                 rgb_loss_w=config.rgb_loss_w, 
                 rot_reg_w=1, 
                 tar_size=224, 
                 tex_reg_w=config.tex_reg_w, 
                 tex_w=config.tex_w, 
                 trans_reg_w=1.0)

    #mask for ITA calculation on uv maps
    fair_mask = cv2.resize(cv2.imread(cfg.mask_path_fair), (cfg.uv_size, cfg.uv_size)).astype(np.float32) / 255.
    fair_mask = torch.from_numpy(fair_mask[None, :, :, :]).permute(0,3,1,2).to(args.device)
    fair_mask = fair_mask[:,0,:,:] == 1.0

    #load benchmark data
    crop_paths = glob.glob(args.img_path + 'crops/*/*.png')
    gt_albedo_paths = glob.glob(args.img_path + 'crop-albedos/*/*.png')
    gt_lms_paths = glob.glob(args.img_path + 'crop-lmks/*/*.npy')

    skin_I = []
    skin_II = []
    skin_III = []
    skin_IV = []
    skin_V = []
    skin_VI = []

    ITA_errors = torch.empty((6, len(crop_paths)), dtype=torch.float32, device = args.device)
    thing = torch.empty((1, 3), dtype=torch.float32, device = args.device)


    #for i in range(len(crop_paths)):
    for i in range(3):
        #prepare gt_albedo, calculate ITA and skin type category
        # gt_albedo = cv2.resize(cv2.imread(gt_albedo_paths[i]), (cfg.uv_size, cfg.uv_size)).astype(np.float32) / 255.
        # gt_albedo = torch.from_numpy(gt_albedo[None, :, :, :]).permute(0,3,1,2).to(args.device)
        gt_albedo = PIL.Image.open(gt_albedo_paths[i]).convert('RGB')
        gt_albedo = np.asarray(gt_albedo) / 255.
        gt_albedo = torch.from_numpy(gt_albedo[None, :, :, :]).permute(0,3,1,2).to(args.device)
        gt_albedo = F.interpolate(gt_albedo, (cfg.uv_size, cfg.uv_size))

        gt_ITA = ITA_calc(gt_albedo, fair_mask)
        ITA_category = ITA_classify(gt_ITA)
        
        gt_lms = np.load(gt_lms_paths[i]).astype(np.float32)
        gt_lms = torch.from_numpy(gt_lms[None, ...]).float().to(args.device)

        args.img_path = crop_paths[i]
        pred_albedo, photo_loss, lm_loss = fit(args, benchmark=True, lms=gt_lms)

        pred_ITA = ITA_calc(pred_albedo, fair_mask)
        #error = ITA_error(pred_ITA, gt_ITA)
        error = ITA_error(pred_ITA, gt_ITA)
        #ITA_errors[ITA_category, i] = error
        thing[0,i] = error + photo_loss*100 + lm_loss * 100000

    return torch.mean(thing)



def ITA_calc(img, mask):
    img = rgb_to_lab(img)

    ITA = (img[:,0,:,:] - 50) / (img[:,2,:,:] + 1e-8)
    ITA = torch.atan(ITA) * 180 / torch.pi
    ITA = ITA[mask]
    return torch.mean(ITA)

def ITA_error(pred_ITA, gt_ITA):
    return abs(pred_ITA - gt_ITA)

def ITA_classify(gt_ITA):
    if gt_ITA < -30:
        return 5
    elif gt_ITA > -30 and gt_ITA < 10:
        return 4
    elif gt_ITA > 10 and gt_ITA < 28:
        return 3
    elif gt_ITA > 28 and gt_ITA < 41:
        return 2
    elif gt_ITA > 41 and gt_ITA < 55:
        return 1
    elif gt_ITA > 55:
        return 0
    else:
        return -1


#if __name__ == '__main__':
def main():
    wandb.login()
    wandb.init(project='3DMM')
    score = bench(wandb.config)
    wandb.log({'score': score})

    
    
    # args = ["--img_path=data/0.png", "--res_folder=results", 
    #                 "--tar_size=224", "--first_nrf_iters=1000", "--id_reg_w=0.05",
    #                 "--tex_reg_w=1.7e-04", "--rgb_loss_w=0.001", "--lm_loss_w=5000.0",
    #                 "--exp_reg_w=0.001", "--tex_w=100.0", "--trans_reg_w=1", "--reni_reg_w=0.03",
    #                 "--padding_ratio=0.3", "--perceptual_w=1.0"]
    
    # args = ImageFittingOptions()
    # args = args.parse()
    #args.img_path = 'TODO' #maybe make img_path the folder, then in the eval loop can specify which image / loop through folder
    #bench(args)

sweep_configuration = {
    'method': 'bayes',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'score'
        },
    'parameters': 
    {
        'exp_reg_w': {'max': 0.5, 'min': 0.001},
        'id_reg_w': {'max': 0.1, 'min': 0.0001},
        'tex_reg_w': {'max': 1.0, 'min': 0.2},
        'reni_reg_w': {'max': 0.4, 'min': 0.0001},
        'lm_loss_w': {'max': 200000.0, 'min': 15000.0},
        'rgb_loss_w': {'max': 250.0, 'min': 100.0},
        'tex_w': {'max': 300.0, 'min': 100.0},
        'perceptual_w': {'max': 100.0, 'min': 0.1}
     }
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='my-first-sweep'
    )

wandb.agent(sweep_id, function=main, count=20)