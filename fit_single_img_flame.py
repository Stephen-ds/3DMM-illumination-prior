from facenet_pytorch import MTCNN

from core.RENISetup import RENI
from RENI.src.utils.utils import sRGB_old
from RENI.src.utils.pytorch3d_envmap_shader import EnvironmentMap

from Flame.models.FLAME import FLAME, FLAMETex
from Flame.utils.config import cfg
from Flame.utils import util
from Flame.utils.renderer import Renderer

import torch.nn.functional as F
import torch.nn as nn
import torchvision
import datetime

from core.options import ImageFittingOptions
import cv2
import face_alignment
import numpy as np
from core import get_recon_model
import os
import torch
import core.utils as utils
from tqdm import tqdm
import core.losses as losses
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
import wandb

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"]


def fit(args):
    writer = SummaryWriter()
    #wandb.init(project="3DMM",
    #    notes="playing with cameras")

    # init face detection and lms detection models
    print('loading models')
    mtcnn = MTCNN(device=args.device, select_largest=False)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, flip_input=False)
    recon_model = get_recon_model(model=args.recon_model,
                                  device=args.device,
                                  batch_size=1,
                                  img_size=args.tar_size)
    
    flame = FLAME(cfg).to(args.device)
    flametex = FLAMETex(cfg).to(args.device)

    reni = RENI(32, 64, device=args.device)
    reni_model = reni.model

    render = Renderer(args.tar_size, obj_filename=cfg.mesh_file).to(args.device)

    print('loading images')
    img_arr = cv2.imread(args.img_path)[:, :, ::-1]
    orig_h, orig_w = img_arr.shape[:2]

    uv_skin_mask = cv2.resize(cv2.imread('Flame/data/uv_face_mask.png'), (256, 256)).astype(np.float32) / 255.
    uv_skin_mask = torch.from_numpy(uv_skin_mask[None, :, :, :]).permute(0,3,1,2).to(args.device)

    print('image is loaded. width: %d, height: %d' % (orig_w, orig_h))

    # detect the face using MTCNN
    bboxes, probs = mtcnn.detect(img_arr)

    if bboxes is None:
        print('no face detected')
    else:
        face_img, sidelen, bbox, bbox_offset  = utils.pad_img(bboxes[0], img_arr, args.padding_ratio)
        print('A face is detected. l: %d, t: %d, r: %d, b: %d'
            % (bbox[0], bbox[1], bbox[2], bbox[3]))

    resized_face_img = cv2.resize(face_img, (args.tar_size, args.tar_size))

    lms = fa.get_landmarks_from_image(resized_face_img)[0]
    lms = lms[:, :2][None, ...]
    lms = torch.tensor(lms, dtype=torch.float32, device=args.device)
    img_tensor = torch.tensor(
        resized_face_img[None, ...], dtype=torch.float32, device=args.device).permute(0,3,1,2)
    img_tensor = img_tensor / 255

    print('landmarks detected.')

    savefolder = os.path.sep.join([cfg.save_folder, os.path.basename(args.img_path)[:-4]])

    print('start rigid fitting')
    bz = img_tensor.shape[0]
    shape = nn.Parameter(torch.zeros(bz, cfg.shape_params).float().to(args.device))
    tex = nn.Parameter(torch.zeros(bz, cfg.tex_params).float().to(args.device))
    exp = nn.Parameter(torch.zeros(bz, cfg.expression_params).float().to(args.device))
    pose = nn.Parameter(torch.zeros(bz, cfg.pose_params).float().to(args.device))
    cam = torch.zeros(bz, cfg.camera_params); cam[:, 0] = 5.
    cam = nn.Parameter(cam.float().to(args.device))
    lights = nn.Parameter(torch.zeros(bz, 9, 3).float().to(args.device))
    nonrigid_optimizer = torch.optim.Adam(
        [{'params': [shape, exp, pose, cam, tex, lights]},
        {'params': reni_model.parameters(), 'lr': 1e-1}], lr=cfg.e_lr, weight_decay=cfg.e_wd)
        
    rigid_optimizer = torch.optim.Adam(
        [pose, cam],
        lr=cfg.e_lr,
        weight_decay=cfg.e_wd
    )

    # rigid fitting of pose and camera with 51 static face landmarks,
    # this is due to the non-differentiable attribute of contour landmarks trajectory
    for k in range(200):
        losses = {}
        vertices, landmarks2d, landmarks3d = flame(shape_params=shape, expression_params=exp, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        landmarks2d[..., 1:] = - landmarks2d[..., 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d, cam)
        landmarks3d[..., 1:] = - landmarks3d[..., 1:]

        landmarks2d = util.scale_landmarks(img_tensor, landmarks2d)
        landmarks3d = util.scale_landmarks(img_tensor, landmarks3d)

        losses['landmark'] = util.l2_distance(landmarks2d[:, 17:, :2], lms[:, 17:, :2]) * cfg.w_lmks

        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        rigid_optimizer.zero_grad()
        all_loss.backward()
        rigid_optimizer.step()

        loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        for key in losses.keys():
            loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))
        if k % 10 == 0:
            print(loss_info)

        if k % 10 == 0:
            grids = {}
            visind = range(bz)  # [0]
            grids['images'] = torchvision.utils.make_grid(img_tensor[visind]).detach().cpu()
            grids['lms'] = torchvision.utils.make_grid(
                util.tensor_vis_landmarks(img_tensor[visind], lms[visind], isScale=False))
            grids['landmarks2d'] = torchvision.utils.make_grid(
                util.tensor_vis_landmarks(img_tensor[visind], landmarks2d[visind], isScale=False))
            grids['landmarks3d'] = torchvision.utils.make_grid(
                util.tensor_vis_landmarks(img_tensor[visind], landmarks3d[visind], isScale=False))

            grid = torch.cat(list(grids.values()), 1)
            grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
            grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
            cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

    orig_face_log = (img_tensor*255).cpu().numpy().squeeze().astype(np.uint8)
    print('done rigid fitting. lm_loss: %f' %
          losses['landmark'].detach().cpu().numpy())
    
    layout = {'Losses':{'losses':['Multiline',['losses/lm', 'losses/id_reg', 'losses/exp_reg',
                                               'losses/tex_reg', 'losses/reni_reg',
                                               'losses/tex', 'losses/photo']]}}

    writer.add_custom_scalars(layout)
    
    # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
    for k in range(200, 1000):
        losses = {}
        vertices, landmarks2d, landmarks3d = flame(shape_params=shape, expression_params=exp, pose_params=pose)
        trans_vertices = util.batch_orth_proj(vertices, cam)
        trans_vertices[..., 1:] = - trans_vertices[..., 1:]
        landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        landmarks2d[..., 1:] = - landmarks2d[..., 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d, cam)
        landmarks3d[..., 1:] = - landmarks3d[..., 1:]

        landmarks2d = util.scale_landmarks(img_tensor, landmarks2d)
        landmarks3d = util.scale_landmarks(img_tensor, landmarks3d)

        losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], lms[:, :, :2]) * cfg.w_lmks
        losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * cfg.w_shape_reg  # *1e-4
        losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * cfg.w_expr_reg  # *1e-4
        losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * cfg.w_pose_reg

        ### RENI ###
        D = reni.directions.repeat(bz, 1, 1).type_as(img_tensor)
        S = reni.sineweight.repeat(bz, 1, 1).type_as(img_tensor)
        Z = reni_model.mu # get latent code
        reni_output = reni_model(Z, D)
        reni_output = reni.unnormalise(reni_output)

        envmap = EnvironmentMap(
            environment_map=reni_output,
            directions=D,
            sineweight=S
        )
        ######

        ## render
        albedos = flametex(tex) * uv_skin_mask
        ops = render(vertices, trans_vertices, albedos, envmap)
        predicted_images = ops['images']
        render_mask = ops['pos_mask']
        losses['photometric_texture'] = (render_mask * (ops['images'] - img_tensor).abs()).mean() * cfg.w_pho

        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        nonrigid_optimizer.zero_grad()
        all_loss.backward()
        nonrigid_optimizer.step()

        loss_info = '----iter: {}, time: {}\n'.format(k, datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        for key in losses.keys():
            loss_info = loss_info + '{}: {}, '.format(key, float(losses[key]))

        if k % 10 == 0:
            print(loss_info)

        # visualize
        if k % 10 == 0:
            grids = {}
            visind = range(bz)  # [0]
            grids['images'] = torchvision.utils.make_grid(img_tensor[visind]).detach().cpu()
            grids['lms'] = torchvision.utils.make_grid(
                util.tensor_vis_landmarks(img_tensor[visind], lms[visind], isScale=False))
            grids['landmarks2d'] = torchvision.utils.make_grid(
                util.tensor_vis_landmarks(img_tensor[visind], landmarks2d[visind], isScale=False))
            grids['landmarks3d'] = torchvision.utils.make_grid(
                util.tensor_vis_landmarks(img_tensor[visind], landmarks3d[visind], isScale=False))
            grids['albedoimage'] = torchvision.utils.make_grid(
                (ops['albedo_images'])[visind].detach().cpu())
            grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
            shape_images = render.render_shape(vertices, trans_vertices, img_tensor)
            grids['shape'] = torchvision.utils.make_grid(
                F.interpolate(shape_images[visind], [224, 224])).detach().float().cpu()


            # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
            grid = torch.cat(list(grids.values()), 1)
            grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
            grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

            cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

    single_params = {
        'shape': shape.detach().cpu().numpy(),
        'exp': exp.detach().cpu().numpy(),
        'pose': pose.detach().cpu().numpy(),
        'cam': cam.detach().cpu().numpy(),
        'verts': trans_vertices.detach().cpu().numpy(),
        'albedos':albedos.detach().cpu().numpy(),
        'tex': tex.detach().cpu().numpy(),
        'lit': lights.detach().cpu().numpy()
    }

    with torch.no_grad():
        ### Make predictions ###
        coeffs = recon_model.get_packed_tensors()
        pred_dict = recon_model(coeffs, envmap=envmap, render=True)
        rendered_img = pred_dict['rendered_img']
        mask = pred_dict['mask']
        ######

        ### Reconstruct full image with predicted face overlayed ###
        rendered_img = rendered_img.cpu().numpy().squeeze()
        out_img = (rendered_img*255).astype(np.uint8)
        out_mask = (mask > 0).cpu().numpy().squeeze().astype(np.uint8)
        resized_out_img = cv2.resize(out_img, (sidelen, sidelen))
        resized_mask = cv2.resize(
            out_mask, (sidelen, sidelen), cv2.INTER_NEAREST)[..., None]
        
        #Crop out the black fill pixels from the padding
        resized_out_img = resized_out_img[bbox_offset[1]:bbox_offset[3], bbox_offset[0]:bbox_offset[2], :]
        resized_mask = resized_mask[bbox_offset[1]:bbox_offset[3], bbox_offset[0]:bbox_offset[2], :]

        composed_img = img_arr.copy().astype(np.uint8)
        composed_face = composed_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] * \
        (1 - resized_mask) + resized_out_img * resized_mask
        composed_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = composed_face
        ######

        ### Save outputs ###
        utils.mymkdirs(args.res_folder)
        basename = os.path.basename(args.img_path)[:-4]
        # save the composed image
        out_composed_img_path = os.path.join(
            args.res_folder, basename + '_composed_img.jpg')
        cv2.imwrite(out_composed_img_path, composed_img[:, :, ::-1])
        # save the coefficients
        out_coeff_path = os.path.join(
            args.res_folder, basename + '_coeffs.npy')
        np.save(out_coeff_path,
                coeffs.detach().cpu().numpy().squeeze())

        # save the mesh into obj format
        out_obj_path = os.path.join(
            args.res_folder, basename+'_mesh.obj')
        vs = pred_dict['vs'].cpu().numpy().squeeze()
        tri = pred_dict['tri'].cpu().numpy().squeeze()
        color = pred_dict['color'].cpu().numpy().squeeze()
        utils.save_obj(out_obj_path, vs, tri+1, color)


        print('composed image is saved at %s' % args.res_folder)
        writer.close()


if __name__ == '__main__':
    args = ImageFittingOptions()
    args = args.parse()
    args.device = 'cuda:%d' % args.gpu
    #torch.autograd.set_detect_anomaly(True)
    fit(args)
