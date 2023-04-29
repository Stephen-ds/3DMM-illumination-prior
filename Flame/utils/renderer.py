"""
Author: Yao Feng
Copyright (c) 2020, Yao Feng
All rights reserved.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer.cameras import try_get_projection_transform
from Flame.utils import util
from Flame.utils.crop_lms import get_face_inds

import cv2
import pickle

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesVertex,
    Materials,
)

from RENI.src.utils.utils import sRGB_old


class Pytorch3dRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.

    Notice:
        x,y,z are in image space
    """

    def __init__(self, image_size=224):
        """
        Args:
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        fixed_vetices = vertices.clone()
        #fixed_vetices[..., :2] = -fixed_vetices[..., :2]
        meshes_screen = Meshes(verts=fixed_vetices.float(), faces=faces.long())
        raster_settings = self.raster_settings

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1  # []
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        # import ipdb; ipdb.set_trace()
        return pixel_vals

class PerspectiveRasterizer(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.

    Notice:
        x,y,z are in image space
    """

    def __init__(self, focal, image_size=224):
        """
        Args:
            raster_settings: the parameters for rasterization. This should be a
                named tuple.
        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        raster_settings = util.dict2obj(raster_settings)
        self.raster_settings = raster_settings
        R, T = look_at_view_transform(10, 0, 0)  # camera's position
        cameras = FoVPerspectiveCameras(R=R, T=T, znear=0.01,
                                        zfar=50,
                                        fov=2*np.arctan(image_size//2/focal)*180./np.pi)
        self.cameras = cameras

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.

        Returns:
            meshes_proj: a Meshes object with the vertex positions projected
            in NDC space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        self.cameras.to(meshes_world.device)

        n_cameras = len(self.cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        verts_view = self.cameras.get_world_to_view_transform(**kwargs).transform_points(
            verts_world, eps=eps
        )
        # Call transform_points instead of explicitly composing transforms to handle
        # the case, where camera class does not have a projection matrix form.
        verts_proj = self.cameras.transform_points(verts_world, eps=eps)
        to_ndc_transform = self.cameras.get_ndc_camera_transform(**kwargs)
        projection_transform = try_get_projection_transform(self.cameras, kwargs)
        if projection_transform is not None:
            projection_transform = projection_transform.compose(to_ndc_transform)
            verts_ndc = projection_transform.transform_points(verts_view, eps=eps)
        else:
            # Call transform_points instead of explicitly composing transforms to handle
            # the case, where camera class does not have a projection matrix form.
            verts_proj = self.cameras.transform_points(verts_world, eps=eps)
            verts_ndc = to_ndc_transform.transform_points(verts_proj, eps=eps)

        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        return meshes_ndc


    def forward(self, vertices, faces, attributes=None):
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        fixed_vetices = vertices
        #fixed_vetices[..., :2] = -fixed_vetices[..., :2]
        meshes_screen = Meshes(verts=fixed_vetices.float(), faces=faces.long())
        raster_settings = self.raster_settings

        meshes_proj = self.transform(meshes_screen)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.blur_radius > 0.0

        if raster_settings.perspective_correct is not None:
            perspective_correct = raster_settings.perspective_correct
        else:
            perspective_correct = self.cameras.is_perspective()

        znear = self.cameras.get_znear()
        if isinstance(znear, torch.Tensor):
            znear = znear.min().item()
        z_clip = None if not perspective_correct or znear is None else znear / 2

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_proj,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1  # []
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        # import ipdb; ipdb.set_trace()
        return pixel_vals

class Renderer(nn.Module):
    def __init__(self, image_size, obj_filename, mask_path_face, mask_path_face_eyes, mask_path_FAIR, device, uv_size=256):
        super(Renderer, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size
        self.device = device

        #vert_inds = get_face_inds()
        verts, faces, aux = load_obj(obj_filename)
        #verts = verts[vert_inds]
        aux = aux
        uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)   
        faces = faces.verts_idx[None, ...]
        #isin_f = torch.isin(faces, vert_inds).sum(2)
        #faces = faces[isin_f == 3].view(1, -1, 3)
        #uvfaces = uvfaces[isin_f == 3].view(1, -1, 3)
        self.rasterizer = PerspectiveRasterizer(image_size * (1015/224), image_size)
        self.uv_rasterizer = PerspectiveRasterizer(uv_size * (1015/224), uv_size)

        # faces
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coordsw
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = util.face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        ## face masks
        # face
        uv_face_mask = cv2.resize(cv2.imread(mask_path_face), (uv_size, uv_size)).astype(np.float32) / 255.
        uv_face_mask = torch.from_numpy(uv_face_mask[None, :, :, :]).permute(0,3,1,2).to(self.device)
        uv_face_mask = uv_face_mask[:,0,:,:]
        self.uv_face_mask = uv_face_mask

        # face + eyes + mouth
        uv_face_eyes_mask = cv2.resize(cv2.imread(mask_path_face_eyes), (uv_size, uv_size)).astype(np.float32) / 255.
        uv_face_eyes_mask = torch.from_numpy(uv_face_eyes_mask[None, :, :, :]).permute(0,3,1,2).to(self.device)
        uv_face_eyes_mask = uv_face_eyes_mask[:,0,:,:]
        self.uv_face_eyes_mask = uv_face_eyes_mask

        # FAIR
        uv_fair_mask = cv2.resize(cv2.imread(mask_path_FAIR), (uv_size, uv_size)).astype(np.float32) / 255.
        uv_fair_mask = torch.from_numpy(uv_fair_mask[None, :, :, :]).permute(0,3,1,2).to(self.device)
        uv_fair_mask = uv_fair_mask[:,0,:,:]
        self.uv_fair_mask = uv_fair_mask


        # shape colors
        colors = torch.tensor([74, 120, 168])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = util.face_vertices(colors, faces)
        self.register_buffer('face_colors', face_colors)

        ## lighting
        # pi = np.pi
        # constant_factor = torch.tensor(
        #     [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
        #      ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
        #      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
        #      (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
        #      (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))])
        # self.register_buffer('constant_factor', constant_factor)



    def forward(self, vertices, transformed_vertices, albedos, envmap=None, envmap_intensity=None):
        '''
        lihgts:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
        vertices: [N, V, 3], vertices in work space, for calculating normals, then shading
        transformed_vertices: [N, V, 3], range(-1, 1), projected vertices, for rendering
        '''
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        #transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10

        envmap_intensity = torch.clamp(envmap_intensity, min=0)

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))

        # render
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1), transformed_face_normals,
                                face_vertices, face_normals], -1)
        # import ipdb;ipdb.set_trace()
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        alpha_images = rendering[:, -1, :, :][:, None, :, :]

        albedos = albedos
        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]

        albedo_images = F.grid_sample(albedos, grid, align_corners=False)
        albedo_images = torch.where(alpha_images > 0., albedo_images, 
                                    torch.zeros(1, dtype=torch.float32, device=rendering.device))
        
        face_mask = F.grid_sample(self.uv_face_mask.unsqueeze(1), grid, align_corners=False)
        face_mask = torch.where(alpha_images > 0., face_mask, 
                                    torch.zeros(1, dtype=torch.float32, device=rendering.device))
        face_eyes_mask = F.grid_sample(self.uv_face_eyes_mask.unsqueeze(1), grid, align_corners=False)
        face_eyes_mask = torch.where(alpha_images > 0., face_eyes_mask, 
                                    torch.zeros(1, dtype=torch.float32, device=rendering.device))
        fair_mask = F.grid_sample(self.uv_fair_mask.unsqueeze(1), grid, align_corners=False)
        fair_mask = torch.where(alpha_images > 0., fair_mask, 
                                    torch.zeros(1, dtype=torch.float32, device=rendering.device))
        

        # remove inner mouth region
        transformed_normal_map = torch.where(albedo_images.sum(1) > 0.0, rendering[:, 3:6, :, :], 
                                    -torch.ones(1, dtype=torch.float32, device=rendering.device))
        # pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()
        #transformed_normal_map = rendering[:, 3:6, :, :]
        pos_mask = (transformed_normal_map[:, 2:, :, :] > -0.05).float()
        albedo_images = albedo_images * pos_mask

        # shading
        if envmap is not None:
            normal_images = rendering[:, 9:12, :, :]
            shading_images = self.add_RENI(normal_images.permute(0, 2, 3, 1), envmap, envmap_intensity)
            # print('albedo')
            # print(torch.max(albedo_images))
            # print(torch.mean(albedo_images))
            # print(torch.min(albedo_images))
            # print('shade')
            # print(torch.max(shading_images))
            # print(torch.mean(shading_images))
            # print(torch.min(shading_images))
            images = albedo_images * shading_images * envmap_intensity.permute(0,2,1).unsqueeze(-1)
            images = sRGB_old(images, permute = False)
            shading_images = sRGB_old(shading_images, permute = False)

        else:
            images = albedo_images
            shading_images = images

        outputs = {
            'images': images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'face_mask': face_mask,
            'face_eyes_mask': face_eyes_mask,
            'fair_mask': fair_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals
        }

        return outputs
    
    def add_RENI(self, normal_images, envmap, envmap_intensity):
        light_directions = envmap.directions.to(
            device=normal_images.device
        )  # (B, J, 3) unit vector associated with the direction of each pixel in a panoramic image where J = H*W
        light_colors = envmap.environment_map.to(
            device=normal_images.device
        ) / envmap_intensity # (B, J, 3) RGB color of the environment map.

        normal_images = F.normalize(normal_images, p=2, dim=-1, eps=1e-6)
        #create copies of light directions for batch matrix multiplication
        L_batch = light_directions.unsqueeze(1).repeat(1, normal_images.shape[1], 1, 1) # (B, H, J, 3)

        diffuse = torch.einsum("bhwk,bhjk->bhwj", normal_images, L_batch)  # (B, H, W, J)
        diffuse = torch.clamp(diffuse, min=0.0, max=1.0)                   
        # scale every dot product by colour of light source, prescaled by sineweight
        diffuse = torch.einsum("bjk,bhwj->bhwk", light_colors, diffuse)  # (B, H, W, 3)

        return diffuse.permute(0,3,1,2)



    def add_SHlight(self, normal_images, sh_coeff):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1], \
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ],
            1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        # import ipdb; ipdb.set_trace()
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]; light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nlgiht, nv, 3]
        '''
        light_direction = lights[:, :, :3];
        light_intensities = lights[:, :, 3:]
        directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
        return shading

    def render_shape(self, vertices, transformed_vertices, images=None, envmap=None, envmap_intensity=None):
        batch_size = vertices.shape[0]
        # if lights is None:
        #     light_positions = torch.tensor([[-0.1, -0.1, 0.2],
        #                                     [0, 0, 1]]
        #                                    )[None, :, :].expand(batch_size, -1, -1).float()
        #     light_intensities = torch.ones_like(light_positions).float()
        #     lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        #transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] + 10
        envmap_intensity = torch.clamp(envmap_intensity, 0.05, 1.0)

        # Attributes
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = util.vertex_normals(vertices, self.faces.expand(batch_size, -1, -1));
        face_normals = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = util.vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1));
        transformed_face_normals = util.face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        # render
        attributes = torch.cat(
            [self.face_colors.expand(batch_size, -1, -1, -1), transformed_face_normals, face_vertices,
             face_normals], -1)
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        # albedo
        albedo_images = rendering[:, :3, :, :]
        # shading
        normal_images = rendering[:, 9:12, :, :]
        if envmap is not None:
            normal_images = rendering[:, 9:12, :, :]
            shading_images = self.add_RENI(normal_images.permute(0, 2, 3, 1), envmap, envmap_intensity)
            images = albedo_images * shading_images
            images = sRGB_old(images, permute = False)
        else:
            images = albedo_images

        return images

    def render_normal(self, transformed_vertices, normals):
        '''
        -- rendering normal
        '''
        batch_size = normals.shape[0]

        # Attributes
        attributes = util.face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :]
        normal_images = rendering[:, :3, :, :]
        return normal_images

    def world2uv(self, vertices):
        '''
        sample vertices from world space to uv space
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertices = util.face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]

        return uv_vertices
    
    def apply_tex(self, uvcoords_images, transformed_vertices, mask_name=None):
        uv_verts = self.world2uv(transformed_vertices)
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]

        albedo_images = F.grid_sample(albedos, grid, align_corners=False)


    def save_obj(self, filename, vertices, textures):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        util.save_obj(filename, vertices, self.faces[0], textures=textures, uvcoords=self.raw_uvcoords[0],
                      uvfaces=self.uvfaces[0])