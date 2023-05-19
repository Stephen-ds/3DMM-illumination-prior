import RENI.src.models.RENI as rModel
from RENI.src.utils.pytorch3d_envmap_shader import EnvironmentMap
from RENI.src.utils.utils import get_directions, get_sineweight
from RENI.src.utils.custom_transforms import UnMinMaxNormlise
from RENI.src.models.RENI import RENIVariationalAutoDecoder
from RENI.src.utils.utils import sRGB
from kornia.color.rgb import linear_rgb_to_rgb

from torchvision import transforms
from PIL import Image

import torch

def get_render(self, model_output, directions, sineweight, envmap):      
    render, _ = self.renderer(
        meshes_world=self.mesh, R=self.R, T=self.T, envmap=envmap
        )
    return render


class RENI():
    def __init__(self, H, W, device):
        self.device = device
        #self.img_size = img_size

        chkpt_path = 'RENI/models/latent_dim_36_net_5_256_vad_cbc_tanh_hdr/version_0/checkpoints/fit_latent_epoch=759.ckpt'
        self.chkpt = torch.load(chkpt_path, map_location=self.device)
        self.config = self.chkpt['hyper_parameters']['config']

        self.H = H
        self.W = W

        dataset_size = 1
        equivariance = self.config.RENI.EQUIVARIANCE
        latent_dim = self.config.RENI.LATENT_DIMENSION
        hidden_layers = self.config.RENI.HIDDEN_LAYERS
        hidden_features = self.config.RENI.HIDDEN_FEATURES
        out_features = self.config.RENI.OUT_FEATURES
        last_layer_linear = self.config.RENI.LAST_LAYER_LINEAR
        output_activation = self.config.RENI.OUTPUT_ACTIVATION
        first_omega_0 = self.config.RENI.FIRST_OMEGA_0
        hidden_omega_0 = self.config.RENI.HIDDEN_OMEGA_0
        

        model = RENIVariationalAutoDecoder(dataset_size,
                                   latent_dim,
                                   equivariance,
                                   hidden_features,
                                   hidden_layers,
                                   out_features,
                                   last_layer_linear,
                                   output_activation,
                                   first_omega_0,
                                   hidden_omega_0,
                                   fixed_decoder=True)

        model.load_state_dict(self.chkpt['state_dict'], device=self.device)
        model.to(self.device)

        self.model = model

        self.directions = get_directions(self.W).to(device=self.device)
        self.sineweight = get_sineweight(self.W).to(device=self.device)
        
    def unnormalise(self, img):
        transforms = self.config.DATASET[self.config.DATASET.NAME].TRANSFORMS
        minmax = transforms[0][1]
        unnormalise = UnMinMaxNormlise(minmax)

        return unnormalise(img)

    def to_sRGB(self, img, H, W):
        img = img.view(-1, H, W, 3)
        img = img.permute(0,3,1,2)
        #img = sRGB(img)
        img = linear_rgb_to_rgb(img)
        img = img.permute(0,2,3,1)

        return img

