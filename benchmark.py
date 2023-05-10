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


    


def bench():
    # args = Namespace(cache_folder="fitting_cache", 
    #              device="cuda:0",
    #              exp_reg_w=0.4534, 
    #              first_nrf_iters=1000, 
    #              first_rf_iters=1000,
    #              gpu=0,
    #              id_reg_w=0.005506, 
    #              img_path="benchmarks/FAIR_benchmark/validation_set/", 
    #              lm_loss_w=195464.299, 
    #              nframes_shape=16, 
    #              nrf_lr=0.01, 
    #              padding_ratio=0.3, 
    #              perceptual_w=89.341, 
    #              recon_model="bfm09", 
    #              reni_reg_w=0.01982, 
    #              res_folder="results", 
    #              rest_nrf_iters=30, 
    #              rest_rf_iters=50, 
    #              rf_lr=0.01, 
    #              rgb_loss_w=200.555, 
    #              rot_reg_w=1, 
    #              tar_size=224, 
    #              tex_reg_w=0.8126, 
    #              tex_w=155.912, 
    #              trans_reg_w=1.0)
    args = Namespace(cache_folder="fitting_cache", 
                 device="cuda:0",
                 exp_reg_w=0.1435, 
                 first_nrf_iters=1000, 
                 first_rf_iters=1000,
                 gpu=0,
                 id_reg_w=0.0403, 
                 img_path="benchmarks/FAIR_benchmark/validation_set/", 
                 lm_loss_w=126454.27, 
                 nframes_shape=16, 
                 nrf_lr=0.01, 
                 padding_ratio=0.3, 
                 perceptual_w=39.341, 
                 recon_model="bfm09", 
                 reni_reg_w=0.001, 
                 res_folder="results", 
                 rest_nrf_iters=30, 
                 rest_rf_iters=50, 
                 rf_lr=0.01, 
                 rgb_loss_w=450.095, 
                 rot_reg_w=1, 
                 tar_size=224, 
                 tex_reg_w=0.0742, 
                 tex_w=62.828, 
                 trans_reg_w=1.0)
    # args = Namespace(cache_folder="fitting_cache", 
    #              device="cuda:0",
    #              exp_reg_w=config.exp_reg_w, 
    #              first_nrf_iters=1000, 
    #              first_rf_iters=1000,
    #              gpu=0,
    #              id_reg_w=config.id_reg_w, 
    #              img_path="benchmarks/FAIR_benchmark/validation_set/", 
    #              lm_loss_w=config.lm_loss_w, 
    #              nframes_shape=16, 
    #              nrf_lr=0.01, 
    #              padding_ratio=0.3, 
    #              perceptual_w=1.0, 
    #              recon_model="bfm09", 
    #              reni_reg_w=config.reni_reg_w, 
    #              res_folder="results", 
    #              rest_nrf_iters=30, 
    #              rest_rf_iters=50, 
    #              rf_lr=0.01, 
    #              rgb_loss_w=config.rgb_loss_w, 
    #              rot_reg_w=1, 
    #              tar_size=224, 
    #              tex_reg_w=config.tex_reg_w, 
    #              tex_w=config.tex_w, 
    #              trans_reg_w=1.0)

    #mask for ITA calculation on uv maps
    fair_mask = cv2.resize(cv2.imread(cfg.mask_path_fair), (cfg.uv_size, cfg.uv_size)).astype(np.float32) / 255.
    fair_mask = torch.from_numpy(fair_mask[None, :, :, :]).permute(0,3,1,2).to(args.device)
    fair_mask = fair_mask[:,0,:,:] == 1.0

    #load benchmark data
    crop_paths = glob.glob(args.img_path + 'crops/*/*.png')
    gt_albedo_paths = glob.glob(args.img_path + 'crop-albedos/*/*.png')
    gt_lms_paths = glob.glob(args.img_path + 'crop-lmks/*/*.npy')

    

    ITA_errors = torch.empty((6, len(crop_paths)), dtype=torch.float32, device = args.device)
    mae = torch.empty((1, len(crop_paths)), dtype=torch.float32, device = args.device)


    #for i in range(len(crop_paths)):
    for i in range(1):
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
        pred_albedo, photo_loss, lm_loss = fit(args, benchmark=True, lms=gt_lms, outpath=crop_paths[i])

        pred_ITA = ITA_calc(pred_albedo, fair_mask)
        error = ITA_error(pred_ITA, gt_ITA)
        ITA_errors[ITA_category, i] = error
        mae[i] = MAE(pred_albedo, gt_albedo)

    mae = torch.mean(mae).numpy()
    skin_I = torch.mean(ITA_errors[0,:]).numpy()
    skin_II = torch.mean(ITA_errors[1,:]).numpy()
    skin_III = torch.mean(ITA_errors[2,:]).numpy()
    skin_IV = torch.mean(ITA_errors[3,:]).numpy()
    skin_V = torch.mean(ITA_errors[4,:]).numpy()
    skin_VI = torch.mean(ITA_errors[5,:]).numpy()
    avg = torch.mean(ITA_errors).numpy()
    bias = torch.std(skin_I, skin_II, skin_III, skin_IV, skin_V, skin_VI).numpy()

    outres = np.array([avg, bias, avg+bias, mae, skin_I, skin_II, skin_III, skin_IV, skin_V, skin_VI])

    np.savetxt('tmp/res.txt', outres, delimiter=',')




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
    
def MAE(pred_albedo, gt_albedo):
    return torch.nn.L1Loss(pred_albedo, gt_albedo)


if __name__ == '__main__':


    
    
    # args = ["--img_path=data/0.png", "--res_folder=results", 
    #                 "--tar_size=224", "--first_nrf_iters=1000", "--id_reg_w=0.05",
    #                 "--tex_reg_w=1.7e-04", "--rgb_loss_w=0.001", "--lm_loss_w=5000.0",
    #                 "--exp_reg_w=0.001", "--tex_w=100.0", "--trans_reg_w=1", "--reni_reg_w=0.03",
    #                 "--padding_ratio=0.3", "--perceptual_w=1.0"]
    
    # args = ImageFittingOptions()
    # args = args.parse()
    # args.img_path = "benchmarks/FAIR_benchmark/validation_set/" 
    bench()

