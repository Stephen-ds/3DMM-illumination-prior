import torch
import torch.nn as nn
import numpy as np
import RENI.src.utils.pytorch3d_envmap_shader as rRender


from pytorch3d.structures import Meshes
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


class BaseReconModel(nn.Module):
    def __init__(self, img_size, batch_size=1, device='cuda:0'):
        super(BaseReconModel, self).__init__()

        self.batch_size = batch_size
        self.img_size = img_size
        self.focal = img_size * (1015/224)

        self.device = torch.device(device)
        self.renderer = self._get_renderer(self.device)
        self.renderer_albedo = self._get_renderer_albedo(self.device)

        self.p_mat = self._get_p_mat(device)
        self.reverse_z = self._get_reverse_z(device)
        self.camera_pos = self._get_camera_pose(device)

        self.rot_tensor = None
        self.exp_tensor = None
        self.id_tensor = None
        self.tex_tensor = None
        self.trans_tensor = None
        #here
        self.gamma_tensor = None
        self.kd_tensor = None
        self.shine_tensor = None

        self.init_coeff_dims()

        self.init_coeff_tensors()

    def _get_camera_pose(self, device):
        camera_pos = torch.tensor(
            [0.0, 0.0, 10.0], device=device).reshape(1, 1, 3)
        return camera_pos

    def _get_p_mat(self, device):
        half_image_width = self.img_size // 2
        p_matrix = np.array([self.focal, 0.0, half_image_width,
                             0.0, self.focal, half_image_width,
                             0.0, 0.0, 1.0], dtype=np.float32).reshape(1, 3, 3)
        return torch.tensor(p_matrix, device=device)

    def _get_reverse_z(self, device):
        reverse_z = np.reshape(
            np.array([1.0, 0, 0, 0, 1, 0, 0, 0, -1.0], dtype=np.float32), [1, 3, 3])
        return torch.tensor(reverse_z, device=device)

    def _get_renderer(self, device):
        return rRender.build_renderer(self.img_size, self.focal, device)
    
    def _get_renderer_albedo(self, device):
        return rRender.build_renderer_albedo(self.img_size, self.focal, 1, device)

    def compute_norm(self, vs, tri, point_buf):

        face_id = tri
        point_id = point_buf
        v1 = vs[:, face_id[:, 0], :]
        v2 = vs[:, face_id[:, 1], :]
        v3 = vs[:, face_id[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)
        empty = torch.zeros((face_norm.size(0), 1, 3),
                            dtype=face_norm.dtype, device=face_norm.device)
        face_norm = torch.cat((face_norm, empty), 1)
        v_norm = face_norm[:, point_id, :].sum(2)
        v_norm = v_norm / v_norm.norm(dim=2).unsqueeze(2)

        return v_norm

    def project_vs(self, vs):
        batchsize = vs.shape[0]

        vs = torch.matmul(vs, self.reverse_z.repeat(
            (batchsize, 1, 1))) + self.camera_pos
        aug_projection = torch.matmul(
            vs, self.p_mat.repeat((batchsize, 1, 1)).permute((0, 2, 1)))

        face_projection = aug_projection[:, :, :2] / \
            torch.reshape(aug_projection[:, :, 2], [batchsize, -1, 1])
        return face_projection

    def compute_rotation_matrix(self, angles):
        n_b = angles.size(0)
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(
            n_b * 3, 1, 1).view(3, n_b, 3, 3).to(angles.device)

        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz

        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])

        return rotation.permute(0, 2, 1)

    def add_illumination(self, face_texture, norm, gamma):
        #here big changes
        n_b, num_vertex, _ = face_texture.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8

        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        Y0 = torch.ones(n_v_full).to(gamma.device).float() * a0 * c0
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(Y0)
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

    def rigid_transform(self, vs, rot, trans):

        vs_r = torch.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)

        return vs_t

    def get_lms(self, vs):
        raise NotImplementedError()

    def forward(self, coeffs, render=True):
        raise NotImplementedError()

    def get_vs(self, id_coeff, exp_coeff):
        raise NotImplementedError()

    def get_color(self, tex_coeff):
        raise NotImplementedError()

    def get_rot_tensor(self):
        return self.rot_tensor

    def get_trans_tensor(self):
        return self.trans_tensor

    def get_exp_tensor(self):
        return self.exp_tensor

    def get_tex_tensor(self):
        return self.tex_tensor

    def get_id_tensor(self):
        return self.id_tensor

    #here
    def get_gamma_tensor(self):
        return self.gamma_tensor
    
    def get_kd_tensor(self):
        return self.kd_tensor
    
    def get_shine_tensor(self):
        return self.shine_tensor

    def init_coeff_dims(self):
        raise NotImplementedError()

    def get_packed_tensors(self):
        #here gamma
        return self.merge_coeffs(self.id_tensor.repeat(self.batch_size, 1),
                                 self.exp_tensor,
                                 self.tex_tensor.repeat(self.batch_size, 1),
                                 self.rot_tensor, self.gamma_tensor,
                                 self.trans_tensor,
                                 self.kd_tensor,
                                 self.shine_tensor)

    def init_coeff_tensors(self, id_coeff=None, tex_coeff=None):
        if id_coeff is None:
            self.id_tensor = torch.zeros(
                (1, self.id_dims), dtype=torch.float32,
                requires_grad=True, device=self.device)
        else:
            assert id_coeff.shape == (1, self.id_dims)
            self.id_tensor = torch.tensor(
                id_coeff, dtype=torch.float32,
                requires_grad=True, device=self.device)

        if tex_coeff is None:
            self.tex_tensor = torch.zeros(
                (1, self.tex_dims), dtype=torch.float32,
                requires_grad=True, device=self.device)
        else:
            assert tex_coeff.shape == (1, self.tex_dims)
            self.tex_tensor = torch.tensor(
                tex_coeff, dtype=torch.float32,
                requires_grad=True, device=self.device)

        self.exp_tensor = torch.zeros(
            (self.batch_size, self.exp_dims), dtype=torch.float32,
            requires_grad=True, device=self.device)
        #here
        self.gamma_tensor = torch.zeros(
            (self.batch_size, 27), dtype=torch.float32,
            requires_grad=False, device=self.device)
        self.trans_tensor = torch.zeros(
            (self.batch_size, 3), dtype=torch.float32,
            requires_grad=True, device=self.device)
        self.rot_tensor = torch.zeros(
            (self.batch_size, 3), dtype=torch.float32,
            requires_grad=True, device=self.device)
        self.kd_tensor = torch.full((self.batch_size, 1), 0.9, dtype=torch.float32,
            requires_grad=True, device=self.device)
        self.shine_tensor = torch.full((self.batch_size, 1), 0.2, dtype=torch.float32,
            requires_grad=True, device=self.device)
