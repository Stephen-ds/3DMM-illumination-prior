import torch
import torch.nn as nn
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.io import IO

from core.BaseModel import BaseReconModel
from RENI.src.utils.utils import sRGB_old

#from core.RENISetup import init_envmap
from RENI.src.utils.utils import sRGB

from Flame.utils.config import cfg
from Flame.models.FLAME import FLAME, FLAMETex
from Flame.utils.renderer import Renderer

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    blending
)


class FLAMEReconModel(BaseReconModel):
    def __init__(self, model_dict, **kargs):
        super(FLAMEReconModel, self).__init__(**kargs)

        self.cfg = cfg
        self.flame = FLAME(self.cfg).to(self.device)
        self.flametex = FLAMETex(self.cfg).to(self.device)

        self.renderer = self._get_renderer()

        self.skinmask = torch.tensor(
            model_dict['skinmask'], requires_grad=False, device=self.device)

        self.kp_inds = torch.tensor(
            model_dict['keypoints']-1).squeeze().long().to(self.device)

        self.meanshape = torch.tensor(model_dict['meanshape'],
                                      dtype=torch.float32, requires_grad=False,
                                      device=self.device)

        self.idBase = torch.tensor(model_dict['idBase'],
                                   dtype=torch.float32, requires_grad=False,
                                   device=self.device)

        self.expBase = torch.tensor(model_dict['exBase'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)

        self.meantex = torch.tensor(model_dict['meantex'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)

        self.texBase = torch.tensor(model_dict['texBase'],
                                    dtype=torch.float32, requires_grad=False,
                                    device=self.device)

        self.tri = torch.tensor(model_dict['tri']-1,
                                dtype=torch.int64, requires_grad=False,
                                device=self.device)

        self.point_buf = torch.tensor(model_dict['point_buf']-1,
                                      dtype=torch.int64, requires_grad=False,
                                      device=self.device)
        
    def _get_renderer(self):
        R, T = look_at_view_transform(10, 0, 0)  # camera's position
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01,
                                        zfar=50,
                                        fov=2*np.arctan(self.img_size//2/self.focal)*180./np.pi)
        return Renderer(self.img_size, obj_filename=self.cfg.mesh_file).to(self.device)

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs):
        id_coeff = coeffs[:, :80]  # identity(shape) coeff of dim 80
        exp_coeff = coeffs[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeffs[:, 144:224]  # texture(albedo) coeff of dim 80
        # ruler angles(x,y,z) for rotation of dim 3
        angles = coeffs[:, 224:227]
        # lighting coeff for 3 channel SH function of dim 27
        #here - gamma
        gamma = coeffs[:, 227:254]
        translation = coeffs[:, 254:257]  # translation coeff of dim 3
        kd = coeffs[:,257:258] #kd coeff of dim 1
        shine = coeffs[:,258:259] #shine coeff of dim 1

        return id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, kd, shine

    #here - gamma
    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, kd, shine):
        coeffs = torch.cat([id_coeff, exp_coeff, tex_coeff,
                            angles, gamma, translation, kd, shine], dim=1)
        return coeffs

    def forward(self, coeffs, envmap=None, render=True):
        batch_num = coeffs.shape[0]
        #here - gamma
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, kd, shine = self.split_coeffs(
            coeffs)

        vs = self.get_vs(id_coeff, exp_coeff)

        rotation = self.compute_rotation_matrix(angles)

        vs_t = self.rigid_transform(
            vs, rotation, translation)

        lms_t = self.get_lms(vs_t)
        lms_proj = self.project_vs(lms_t)
        lms_proj = torch.stack(
            [lms_proj[:, :, 0], self.img_size-lms_proj[:, :, 1]], dim=2)
        if render:
            face_texture = self.get_color(tex_coeff)
            face_norm = self.compute_norm(vs, self.tri, self.point_buf)
            face_norm_r = face_norm.bmm(rotation)
            #here - could potentially remove since we add illumination
            # at render time
            #face_color = self.add_illumination(
            #    face_texture, face_norm_r, gamma)
            #here this could just be albedo since lighting comes from envmap
            #face_color_tv = TexturesVertex(face_color)
            face_color_tv = TexturesVertex(face_texture)

            mesh = Meshes(vs_t, self.tri.repeat(
                batch_num, 1, 1), face_color_tv)
            
            #IO.save_mesh(mesh, 'results/mesh.obj')
            #IO.save_mesh()
            
            #envmap = init_envmap(img_size=self.img_size)
            rendered_img = self.renderer(mesh, envmap=envmap, kd=kd, shininess=shine)
            normals = rendered_img[2]
            albedo_img = rendered_img[1]
            #specular_img = rendered_img[1]
            mask = rendered_img[3]
            rendered_img = rendered_img[0]
            #here might need to think about this clamping
            #rendered_img = torch.clamp(rendered_img[0], 0, 255)
            #here need to think about face_color - it is face_texture
            # with illumination added. See where RENI renderer adds this
            # and maybe you can return it from inside there
            return {'rendered_img': rendered_img,
                    'albedo_img': albedo_img,
                    #'specular_img': specular_img,
                    'normals': normals,
                    'mask': mask,
                    'lms_proj': lms_proj,
                    'face_texture': face_texture,
                    'vs': vs_t,
                    'tri': self.tri,
                    'color': face_texture}
        
        else:
            return {'lms_proj': lms_proj}

    def get_vs(self, id_coeff, exp_coeff):
        #vs are vertices
        n_b = id_coeff.size(0)

        face_shape = torch.einsum('ij,aj->ai', self.idBase, id_coeff) + \
            torch.einsum('ij,aj->ai', self.expBase, exp_coeff) + self.meanshape

        #3 is because it's the xyz coords for each vertex
        face_shape = face_shape.view(n_b, -1, 3)
        face_shape = face_shape - \
            self.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

        return face_shape

    def get_color(self, tex_coeff):
        n_b = tex_coeff.size(0)
        face_texture = torch.einsum(
            'ij,aj->ai', self.texBase, tex_coeff) + self.meantex

        #3 because 3 rgb channels
        face_texture = face_texture.view(n_b, -1, 3)
        face_texture = torch.clamp(face_texture, 0.0, 255.0)
        return face_texture/255

    def get_skinmask(self):
        return self.skinmask

    def init_coeff_dims(self):
        self.id_dims = cfg.shape_params
        self.tex_dims = cfg.tex_params
        self.exp_dims = cfg.expression_params
        self.pose_dims = cfg.pose_params
