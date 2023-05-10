from facenet_pytorch import MTCNN, InceptionResnetV1

from core.RENISetup import RENI
from RENI.src.utils.utils import sRGB_old
from RENI.src.utils.pytorch3d_envmap_shader import EnvironmentMap
from mpl_toolkits.mplot3d import Axes3D

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
import core.losses as lossfuncs
import matplotlib.pyplot as plt
from PIL import Image
from core.BFM09Model import BFM09ReconModel
from core.BaseModel import BaseReconModel
from core.preprocess_img import get_skinmask
from Flame.utils.crop_lms import get_face_inds
from seg.test import evaluate #face-parsing.pytorch


from torch.utils.tensorboard import SummaryWriter
import wandb

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"]


def fit(args, benchmark=False, lms=None, outpath=None):

    writer = SummaryWriter()
    #wandb.init(project="3DMM",
    #    notes="playing with cameras")

    # init face detection and lms detection models
    print('loading models')
    mtcnn = MTCNN(device=args.device, select_largest=False)
    mtcnn_features = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=args.device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(args.device)
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

    render = Renderer(args.tar_size, cfg.mesh_file, cfg.mask_path_face, cfg.mask_path_face_eyes, cfg.mask_path_fair, 
                      args.device, uv_size=cfg.uv_size).to(args.device)

    print('loading images')
    img_arr = cv2.imread(args.img_path)[:, :, ::-1]
    orig_h, orig_w = img_arr.shape[:2]

    uv_skin_mask = cv2.resize(cv2.imread(cfg.mask_path_face), (cfg.uv_size, cfg.uv_size)).astype(np.float32) / 255.
    uv_skin_mask = torch.from_numpy(uv_skin_mask[None, :, :, :]).permute(0,3,1,2).to(args.device)
    uv_skin_mask = uv_skin_mask[:,0,:,:]

    uv_albedo_mask = cv2.resize(cv2.imread(cfg.mask_path_albedo), (cfg.uv_size, cfg.uv_size)).astype(np.float32) / 255.
    uv_albedo_mask = torch.from_numpy(uv_albedo_mask[None, :, :, :]).permute(0,3,1,2).to(args.device)
    uv_albedo_mask = uv_albedo_mask[:,0,:,:]

    uv_albedo_weight_mask = cv2.resize(cv2.imread(cfg.mask_path_albedo_weight), (cfg.uv_size, cfg.uv_size)).astype(np.float32) / 255.
    uv_albedo_weight_mask = torch.from_numpy(uv_albedo_weight_mask[None, :, :, :]).permute(0,3,1,2).to(args.device)
    uv_albedo_weight_mask = uv_albedo_weight_mask[:,0,:,:]

    print('image is loaded. width: %d, height: %d' % (orig_w, orig_h))

    ten = img_arr[None, ...].copy()
    ten = torch.from_numpy(ten).to(args.device)
    gt_embed_face,_ = mtcnn(ten, return_prob=True)
    gt_embed_face = gt_embed_face[0].unsqueeze(0).permute(0,3,1,2)
    gt_embeddings = resnet(gt_embed_face)
    gt_embeddings = gt_embeddings.clone().detach().requires_grad_(True)

    # detect the face using MTCNN -TODO don't run mtcnn twice here
    if not benchmark:
        bboxes, probs = mtcnn.detect(img_arr)
        bboxes = bboxes[0] #TODO don't just take [0], make it work for batches
    else:
        bboxes = np.array([0, 0, img_arr.shape[0], img_arr.shape[1]])

    if bboxes is None:
        print('no face detected')
    else:
        face_img, sidelen, bbox, bbox_offset  = utils.pad_img(bboxes, img_arr, args.padding_ratio)
        print('A face is detected. l: %d, t: %d, r: %d, b: %d'
            % (bbox[0], bbox[1], bbox[2], bbox[3]))
        
        resized_face_img = cv2.resize(face_img, (args.tar_size, args.tar_size))


    
    # gt_face_mask = get_skinmask(resized_face_img)
    # gt_face_mask = torch.from_numpy(gt_face_mask).unsqueeze(1).to(device=args.device)
    gt_face_mask, gt_skin_mask = evaluate(image=face_img, cp='79999_iter.pth')
    gt_face_mask = torch.from_numpy(gt_face_mask[None,None,...]).float().to(args.device)
    gt_face_mask = F.interpolate(gt_face_mask, (args.tar_size, args.tar_size))

    gt_skin_mask = torch.from_numpy(gt_skin_mask[None,None,...]).float().to(args.device)
    gt_skin_mask = F.interpolate(gt_skin_mask, (args.tar_size, args.tar_size))
    
    if not benchmark:
        lms = fa.get_landmarks_from_image(resized_face_img)[0]
        lms = lms[:, :2][None, ...]
        lms = torch.tensor(lms, dtype=torch.float32, device=args.device)
        img_tensor = torch.tensor(
            resized_face_img[None, ...], dtype=torch.float32, device=args.device).permute(0,3,1,2)
        img_tensor = img_tensor / 255
    else:
        lms[:,:,0] = (lms[:,:,0] + bbox_offset[0]) * (args.tar_size / sidelen)
        lms[:,:,1] = (lms[:,:,1] + bbox_offset[1]) * (args.tar_size / sidelen)
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
    rot_tensor = torch.zeros((bz, 3), dtype=torch.float32)
    rot_tensor = nn.Parameter(rot_tensor.to(args.device))
    trans_tensor = torch.zeros((bz, 3), dtype=torch.float32)
    trans_tensor = nn.Parameter(trans_tensor.to(args.device))
    envmap_intensity = nn.Parameter(torch.full((bz, 1, 1), 0.8).float().to(args.device))
    nonrigid_optimizer = torch.optim.Adam(
        [{'params': [shape, pose, exp, rot_tensor, trans_tensor, tex]},
        {'params': reni_model.parameters(), 'lr': 1e-1}], lr=args.nrf_lr)
        
    rigid_optimizer = torch.optim.Adam(
        [rot_tensor, trans_tensor],
        lr=args.rf_lr
    )


    # rigid fitting of pose and camera with 51 static face landmarks,
    # this is due to the non-differentiable attribute of contour landmarks trajectory
    lm_weights = utils.get_lm_weights(args.device)
    for k in range(200):
        losses = {}
        vertices, landmarks2d, landmarks3d = flame(shape_params=shape, expression_params=exp, pose_params=pose)
        #vertices = vertices[:,face_inds,:]
        vertices = vertices * 10
        landmarks2d = landmarks2d * 10

        #trans_vertices = BaseReconModel.compute_rotation_matrix(1, rot_tensor)
        rot = BaseReconModel.compute_rotation_matrix(1, rot_tensor)
        trans_vertices = BaseReconModel.rigid_transform(1, vertices, rot, trans_tensor)
        #trans_vertices = util.batch_persp_proj(trans_vertices, args.tar_size)

        #landmarks2d = util.rigid_transform(landmarks2d, rot_tensor, trans_tensor)
        #landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        landmarks2d = BaseReconModel.rigid_transform(1, landmarks2d, rot, trans_tensor)
        landmarks2d = util.batch_persp_proj(landmarks2d, args.tar_size)

        #landmarks2d = util.scale_landmarks(img_tensor, landmarks2d)

        #[:, 17:, :2]
        losses['landmark'] = lossfuncs.lm_loss(landmarks2d[:,17:,:2], lms[:, 17:, :2], lm_weights[17:],
                                     img_size=args.tar_size) * args.lm_loss_w

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

        # if k % 10 == 0:
        #     grids = {}
        #     visind = range(bz)  # [0]
        #     grids['images'] = torchvision.utils.make_grid(img_tensor[visind]).detach().cpu()
        #     grids['lms'] = torchvision.utils.make_grid(
        #         util.tensor_vis_landmarks(img_tensor[visind], lms[visind], isScale=False))
        #     grids['landmarks2d'] = torchvision.utils.make_grid(
        #         util.tensor_vis_landmarks(img_tensor[visind], landmarks2d[visind], isScale=False))

        #     grid = torch.cat(list(grids.values()), 1)
        #     #grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
        #     grid_image = grid.numpy().transpose(1, 2, 0).copy() * 255
        #     grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        #     cv2.imwrite('{}/{}.jpg'.format(savefolder, k), grid_image)

        #     lms_out = lms.cpu().detach().numpy()
        #     landmarks2d_out = landmarks2d.cpu().detach().numpy()

        #     fig, ax = plt.subplots(figsize=(4, 4))
        #     ax.axis('off')
        #     ax.imshow((img_tensor*255).permute(0,2,3,1).cpu().detach().numpy().squeeze().astype(np.uint8))
        #     ax.scatter(lms_out[:,:, 0], lms_out[:,:, 1], s=8)
        #     ax.scatter(landmarks2d_out[:,:, 0], landmarks2d_out[:,:, 1], s=8, color='r')
        #     #wandb.log({'landmarks_proj': fig})
        #     writer.add_figure('landmarks', fig, global_step=k)
        #     plt.close(fig)

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
        #vertices = vertices[:,face_inds,:]
        vertices = vertices * 10
        landmarks2d = landmarks2d * 10

        #trans_vertices = util.rigid_transform(vertices, rot_tensor, trans_tensor)
        rot = BaseReconModel.compute_rotation_matrix(1, rot_tensor)
        trans_vertices = BaseReconModel.rigid_transform(1, vertices, rot, trans_tensor)
        #trans_vertices= util.batch_persp_proj(trans_vertices, args.tar_size)

        #landmarks2d = util.rigid_transform(landmarks2d, rot_tensor, trans_tensor)
        #landmarks2d = util.batch_orth_proj(landmarks2d, cam)
        #landmarks2d = util.batch_persp_proj(landmarks2d, args.tar_size)
        landmarks2d = BaseReconModel.rigid_transform(1, landmarks2d, rot, trans_tensor)
        landmarks2d = util.batch_persp_proj(landmarks2d, args.tar_size)

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
        #albedos = flametex(tex) * uv_skin_mask
        albedos = torch.clamp(flametex(tex), 1e-6, 1)
        ops = render(vertices, trans_vertices, albedos, envmap, envmap_intensity)
        predicted_images = ops['images']
        render_mask = ops['face_mask']
        #fair_mask = ops['fair_mask']
        #weighted_photo_mask = torch.where(gt_skin_mask != 0, render_mask * 1.5, render_mask) * gt_face_mask
        #photo_mask = render_mask * gt_face_mask
        photo_mask = render_mask
        #losses['photometric_texture'] = (render_mask * gt_face_mask * (predicted_images - img_tensor).abs()).mean() * args.rgb_loss_w
        photo_loss = lossfuncs.photo_loss(predicted_images, img_tensor, photo_mask > 0) 
        losses['photometric_texture'] = photo_loss * args.rgb_loss_w

        #testing masks
        facem = predicted_images * render_mask
        faceme = predicted_images * ops['face_eyes_mask']
        facef = predicted_images * ops['fair_mask']
        refvis = albedos * uv_skin_mask

        embed_face,_ = mtcnn(predicted_images.permute(0,2,3,1)*255, return_prob=True)
        ##TODO add failsafe default if no face found so that loss will return something appropriately high
        embed_face = embed_face[0].unsqueeze(0).permute(0,3,1,2)
        if embed_face != None:
            embeddings = resnet(embed_face)
        else: 
            embeddings = -1
        ## loss
        lm_loss = lossfuncs.lm_loss(landmarks2d[:,:,:2], lms, lm_weights,
                                img_size=args.tar_size)
        losses['landmark'] = lm_loss * args.lm_loss_w
        losses['shape_reg'] = lossfuncs.get_l2(shape) * args.id_reg_w  # *1e-4
        losses['expression_reg'] = lossfuncs.get_l2(exp) * args.exp_reg_w  # *1e-4
        losses['tex_reg'] = lossfuncs.get_l2(tex) * args.tex_reg_w
        #losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * cfg.w_pose_reg
        losses['reni reg'] = lossfuncs.get_l2(Z) * args.reni_reg_w
        losses['perceptual'] = lossfuncs.perceptual_loss(embeddings, gt_embeddings) * args.perceptual_w
        #losses['intensity_norm'] = lossfuncs.intensity_norm_loss(envmap_intensity)
        #losses['ita'] = lossfuncs.ita_loss(predicted_images, img_tensor, ops['fair_mask'] * gt_face_mask)

        weighted_albedo_mask = torch.where(uv_albedo_weight_mask != 0, uv_albedo_mask * 1.5, uv_albedo_mask).flatten(1,2)
        albedos_perm = albedos.permute(0,2,3,1).flatten(1,2)
        #albedoss = ops['albedo_images']
        #maskk = ops['fair_mask']
        losses['reflectance'] = lossfuncs.reflectance_loss(
            albedos_perm, weighted_albedo_mask) * args.tex_w

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
        # if k % 10 == 0:
        #     grids = {}
        #     visind = range(bz)  # [0]
        #     grids['images'] = torchvision.utils.make_grid(img_tensor[visind]).detach().cpu()
        #     grids['lms'] = torchvision.utils.make_grid(
        #         util.tensor_vis_landmarks(img_tensor[visind], lms[visind], isScale=False))
        #     grids['landmarks2d'] = torchvision.utils.make_grid(
        #         util.tensor_vis_landmarks(img_tensor[visind], landmarks2d[visind], isScale=False))
        #     grids['albedoimage'] = torchvision.utils.make_grid(
        #         (torch.clamp(ops['albedo_images'], 0, 1))[visind].detach().cpu())
        #     grids['render'] = torchvision.utils.make_grid(predicted_images[visind].detach().float().cpu())
        #     shape_images = render.render_shape(vertices, trans_vertices, img_tensor, envmap, envmap_intensity)
        #     grids['shape'] = torchvision.utils.make_grid(
        #         F.interpolate(shape_images[visind], [args.tar_size, args.tar_size])).detach().float().cpu()
            
        #     #mpl_sphere(envmap.environment_map.clone().detach().cpu().numpy())


        #     # grids['tex'] = torchvision.utils.make_grid(F.interpolate(albedos[visind], [224, 224])).detach().cpu()
        #     grid = torch.cat(list(grids.values()), 1)
        #     grid_image = grid.numpy().transpose(1, 2, 0).copy() * 255
        #     grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)

        #     writer.add_image('lm_grid', grid_image, global_step=k, dataformats='HWC')
            

        #     # RENI environment map
        #     envmap_im = reni_output.view(-1, reni.H, reni.W, 3)
        #     envmap_im = sRGB_old(envmap_im).cpu().detach()
        #     writer.add_image('envmap', envmap_im.squeeze(), global_step=k, dataformats='HWC')


        #     # Predicted face albedo (diffuse)

        #     rendered_img_albedo = albedos.squeeze()*255
        #     render_out_albedo = rendered_img_albedo.cpu().detach().numpy().squeeze().astype(np.uint8)
        #     writer.add_image('albedo', render_out_albedo, global_step=k)

            ## Predicted specular contribution
            # rendered_img_specular = pred_dict['specular_img']
            # render_out_specular = rendered_img_specular.detach().permute(3,1,2,0)*255
            # render_out_specular = render_out_specular.cpu().numpy().squeeze().astype(np.uint8)
            # writer.add_image('specular', render_out_specular, global_step=i)

        #######


    single_params = {
        'shape': shape.detach().cpu().numpy(),
        'exp': exp.detach().cpu().numpy(),
        #'pose': pose.detach().cpu().numpy(),
        'cam': cam.detach().cpu().numpy(),
        'verts': trans_vertices.detach().cpu().numpy(),
        'albedos':albedos.detach().cpu().numpy(),
        'tex': tex.detach().cpu().numpy(),
    }

    with torch.no_grad():
        ### Make predictions ###
        #albedos = flametex(tex) * uv_skin_mask
        ops = render(vertices, trans_vertices, albedos, envmap, envmap_intensity)
        rendered_img = (ops['images']).permute(0,2,3,1)
        mask = (ops['face_eyes_mask'] * gt_face_mask).permute(0,2,3,1)
        ######

        ### Reconstruct full image with predicted face overlayed ###
        rendered_img = rendered_img.cpu().numpy().squeeze()
        out_img = (rendered_img*255).astype(np.uint8)
        out_mask = (mask > 0).cpu().numpy().squeeze().astype(np.uint8)
        composed_img = img_arr.copy().astype(np.uint8)

        resized_out_img = cv2.resize(out_img, (sidelen, sidelen))
        resized_mask = cv2.resize(
            out_mask, (sidelen, sidelen), cv2.INTER_NEAREST)[..., None]
        
        #Crop out the black fill pixels from the padding
        resized_out_img = resized_out_img[bbox_offset[1]:bbox_offset[3], bbox_offset[0]:bbox_offset[2], :]
        resized_mask = resized_mask[bbox_offset[1]:bbox_offset[3], bbox_offset[0]:bbox_offset[2], :]

        composed_face = composed_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] * \
        (1 - resized_mask) + resized_out_img * resized_mask
        composed_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = composed_face
        ######

        ### Save outputs ###
        utils.mymkdirs(args.res_folder)
        if outpath == None:
            basename = os.path.basename(args.img_path)[:-4]
        else:
            basename = os.path.basename(outpath)
        # save the composed image
        out_composed_img_path = os.path.join(
            args.res_folder, basename + '_composed_img.jpg')
        cv2.imwrite(out_composed_img_path, composed_img[:, :, ::-1])
        # save the coefficients
        # out_coeff_path = os.path.join(
        #     args.res_folder, basename + '_coeffs.npy')
        # np.save(out_coeff_path,
        #         coeffs.detach().cpu().numpy().squeeze())

        # # save the mesh into obj format
        # out_obj_path = os.path.join(
        #     args.res_folder, basename+'_mesh.obj')
        # vs = pred_dict['vs'].cpu().numpy().squeeze()
        # tri = pred_dict['tri'].cpu().numpy().squeeze()
        # color = pred_dict['color'].cpu().numpy().squeeze()
        # utils.save_obj(out_obj_path, vs, tri+1, color)


        print('composed image is saved at %s' % args.res_folder)
        writer.close()
        
        if benchmark:
            return albedos, photo_loss, lm_loss

def mpl_sphere(img):
    # define a grid matching the map size, subsample along with pixels
    theta = np.linspace(0, np.pi, img.shape[0])
    phi = np.linspace(0, 2*np.pi, img.shape[1])

    count = 180 # keep 180 points along theta and phi
    theta_inds = np.linspace(0, img.shape[0] - 1, count).round().astype(int)
    phi_inds = np.linspace(0, img.shape[1] - 1, count).round().astype(int)
    theta = theta[theta_inds]
    phi = phi[phi_inds]
    img = img[np.ix_(theta_inds, phi_inds)]

    theta,phi = np.meshgrid(theta, phi)
    R = 1

    # sphere
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)

    # create 3d Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.T, y.T, z.T, facecolors=img, cstride=1, rstride=1) # we've already pruned ourselves

    # make the plot more spherical
    ax.axis('scaled')

if __name__ == '__main__':
    args = ImageFittingOptions()
    args = args.parse()
    args.device = 'cuda:%d' % args.gpu
    #torch.autograd.set_detect_anomaly(True)
    fit(args)
