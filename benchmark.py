from argparse import Namespace
from fit_single_img_flame import fit
from Flame.utils.config import cfg
from core.options import ImageFittingOptions
from core import utils

import torch
import cv2
import numpy as np
import os
import glob
from kornia.color import rgb_to_lab
import PIL
from torch.functional import F
import random

import wandb


    


def bench():
    

    args = Namespace(cache_folder="fitting_cache", 
                 device="cuda:0",
                 exp_reg_w=0.1435, 
                 first_nrf_iters=1000, 
                 first_rf_iters=1000,
                 gpu=0,
                 id_reg_w=0.0403, 
                 img_path="benchmarks/FAIR_benchmark/validation_set/", 
                 lm_loss_w=136454.27, 
                 nframes_shape=16, 
                 nrf_lr=0.01, 
                 padding_ratio=0.3, 
                 perceptual_w=39.341, 
                 recon_model="bfm09", 
                 reni_reg_w=0.001, 
                 res_folder="results", 
                 rest_nrf_iters=30, 
                 rest_rf_iters=50, 
                 rf_lr=0.1, 
                 rgb_loss_w=650.095, 
                 rot_reg_w=0, 
                 tar_size=224, 
                 tex_reg_w=0.242, 
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

    prune_inds = prune_dataset(crop_paths, gt_albedo_paths, fair_mask, args.device)
    random.shuffle(prune_inds)
    

    ITA_errors = torch.zeros((6, len(prune_inds)), dtype=torch.float32, device = args.device)
    mae = torch.zeros((1, len(prune_inds)), dtype=torch.float32, device = args.device)

    utils.mymkdirs(args.res_folder)

    outpath = os.path.join(
            args.res_folder, 'res.txt')


    #for i in range(len(crop_paths)):
    for n, i in enumerate(prune_inds):
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
        pred_albedo, photo_loss, lm_loss = fit(args, benchmark=True, lms=gt_lms, outpath=  crop_paths[i] + str(i))
        

        pred_ITA = ITA_calc(pred_albedo, fair_mask)
        error = ITA_error(pred_ITA, gt_ITA)
        ITA_errors[ITA_category, n] = error

        mae[0, n] = MAE(pred_albedo, gt_albedo)

        del gt_albedo
        del pred_albedo
        del gt_lms
        del gt_ITA
        del photo_loss
        del lm_loss

        
        mae_out = nonzero_mean(mae).detach().cpu().numpy()
        skin_I_out = nonzero_mean(ITA_errors[0,:]).detach().cpu().numpy()
        skin_II_out = nonzero_mean(ITA_errors[1,:]).detach().cpu().numpy()
        skin_III_out = nonzero_mean(ITA_errors[2,:]).detach().cpu().numpy()
        skin_IV_out = nonzero_mean(ITA_errors[3,:]).detach().cpu().numpy()
        skin_V_out = nonzero_mean(ITA_errors[4,:]).detach().cpu().numpy()
        skin_VI_out = nonzero_mean(ITA_errors[5,:]).detach().cpu().numpy()
        bias_out = np.std([skin_I_out, skin_II_out, skin_III_out, skin_IV_out, skin_V_out, skin_VI_out])
        avg_out = nonzero_mean(ITA_errors).detach().cpu().numpy()
        
        outerror = np.array([error.cpu().detach().numpy()])
        outpath_error = os.path.join(
            args.res_folder, crop_paths[i][-13:-6] + '_' + crop_paths[i][-5:-4] + '.txt')

        np.savetxt(outpath_error, outerror, fmt='%1.2f')

        outres = np.array([n, avg_out, bias_out, avg_out+bias_out, mae_out, skin_I_out, skin_II_out, skin_III_out, skin_IV_out, skin_V_out, skin_VI_out])

        del mae_out
        del skin_I_out
        del skin_II_out
        del skin_III_out
        del skin_IV_out
        del skin_V_out
        del skin_VI_out
        del bias_out
        del avg_out
        del ITA_category
        del pred_ITA
        del error

        
        np.savetxt(outpath, outres, delimiter=',', fmt='%1.2f')
    print('stop')
    


def prune_dataset(crop_paths, gt_albedo_paths, fair_mask, device):
    skin_types = np.empty((6, len(crop_paths)), dtype=object)
    prune_inds = []

    for i in range(len(crop_paths)):
        gt_albedo = PIL.Image.open(gt_albedo_paths[i]).convert('RGB')
        gt_albedo = np.asarray(gt_albedo) / 255.
        gt_albedo = torch.from_numpy(gt_albedo[None, :, :, :]).permute(0,3,1,2).to(device)
        gt_albedo = F.interpolate(gt_albedo, (cfg.uv_size, cfg.uv_size))

        gt_ITA = ITA_calc(gt_albedo, fair_mask)
        ITA_category = ITA_classify(gt_ITA)

        skin_types[ITA_category, i] = i


    prune_inds.append(skin_types[0,:][skin_types[0,:]!= None][:7])
    prune_inds.append(skin_types[1,:][skin_types[1,:]!= None][:26])
    prune_inds.append(skin_types[2,:][skin_types[2,:]!= None][:13])
    prune_inds.append(skin_types[3,:][skin_types[3,:]!= None][:13])
    prune_inds.append(skin_types[4,:][skin_types[4,:]!= None][:12])
    prune_inds.append(skin_types[5,:][skin_types[5,:]!= None][:11])
    # prune_inds.append(skin_types[0,:][skin_types[0,:]!= None][:1])
    # prune_inds.append(skin_types[1,:][skin_types[1,:]!= None][:1])
    # prune_inds.append(skin_types[2,:][skin_types[2,:]!= None][:1])
    # prune_inds.append(skin_types[3,:][skin_types[3,:]!= None][:1])
    # prune_inds.append(skin_types[4,:][skin_types[4,:]!= None][:1])
    # prune_inds.append(skin_types[5,:][skin_types[5,:]!= None][:1])
    
    return np.concatenate(prune_inds)

def ITA_calc(img, mask):
    img = rgb_to_lab(img)

    ITA = (img[:,0,:,:] - 50) / (img[:,2,:,:] + 1e-8)
    ITA = torch.atan(ITA) * 180 / torch.pi
    ITA = ITA[mask]
    return torch.mean(ITA)

def ITA_nomask(img):
    img = rgb_to_lab(img)

    ITA = (img[:,0,:,:] - 50) / (img[:,2,:,:] + 1e-8)
    ITA = torch.atan(ITA) * 180 / torch.pi
    return torch.mean(ITA)

def nonzero_mean(X):
    mask = X != 0

    return (X*mask).sum()/mask.sum()

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
    return ITA_error(ITA_nomask(pred_albedo), ITA_nomask(gt_albedo))


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

