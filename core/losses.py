import numpy as np
import torch
import torch.nn.functional as F


def photo_loss_bfm(pred_img, gt_img, img_mask):
    pred_img = pred_img.float()
    diff = torch.sum(torch.square(pred_img-gt_img), 3)
    ind_mask = diff.nonzero(as_tuple=True)
    
    loss = torch.sqrt(diff[ind_mask]) * img_mask[ind_mask]
    loss = torch.sum(loss) / torch.sum(img_mask[ind_mask])
    loss = torch.mean(loss)

    return loss


def photo_loss(pred_img, gt_img, img_mask):
    diff = (pred_img - gt_img)
    loss = torch.sqrt(torch.sum(torch.square(
        diff), 1) + 1e-19) * img_mask
    loss = torch.sum(loss, dim=(2, 3)) / torch.sum(img_mask, dim=(2, 3))
    loss = torch.mean(loss)

    return loss


def lm_loss(pred_lms, gt_lms, weight, img_size):
    loss = torch.sum(torch.square(pred_lms/img_size - gt_lms /
                                  img_size), dim=2) * weight.reshape(1, -1)
    loss = torch.mean(loss.sum(1))

    return loss


# def reg_loss(id_coeff, ex_coeff, tex_coeff):

#     loss = torch.square(id_coeff).sum() + \
#         torch.square(tex_coeff).sum() * 1.7e-3 + \
#         torch.square(ex_coeff).sum(1).mean() * 0.8

#     return loss
def get_l2(tensor):
    return torch.square(tensor).sum()


def reflectance_loss(tex, skin_mask):

    skin_mask = skin_mask.unsqueeze(2)
    tex_mean = torch.sum(tex*skin_mask, 1, keepdims=True)/torch.sum(skin_mask)
    loss = torch.sum(torch.square((tex-tex_mean)*skin_mask)) / \
        (tex.shape[0]*torch.sum(skin_mask))

    return loss


def gamma_loss(gamma):

    gamma = gamma.reshape(-1, 3, 9)
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean(torch.square(gamma - gamma_mean))

    return gamma_loss

def perceptual_loss(pred_id, gt_id):
    pred_id = torch.nn.functional.normalize(pred_id, p=2, dim=1)
    gt_id = torch.nn.functional.normalize(gt_id, p=2, dim=1)

    sim = torch.sum(pred_id*gt_id, 1)

    return torch.sum(max(0.0, 1 - sim))/pred_id.shape[0]
