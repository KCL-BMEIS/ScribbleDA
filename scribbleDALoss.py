import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time

from .Permutohedral_attention_module.PAM_cuda.pl import PermutohedralLattice as pl

class CRFLoss(nn.Module):
    def __init__(self,
                 alpha=5.,
                 beta=5.,
                 gamma=5.,
                 w=[1.0,0.0],
                 is_da=False,
                 use_norm=True):
        super(CRFLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._w = w
        self.pl = pl.apply
        self.is_da = is_da
        self.use_norm = use_norm

    def forward(self, I, U):
        """
        :param I: feature maps used in the dense pairwise term of CRF
        :param U: activation maps used in the unary term of CRF (before Softmax)
        :return: CRF loss
        """
        batch_size, n_feat, x_shape, y_shape, z_shape = I.size()
        nb_voxels = x_shape*y_shape*z_shape 
        spatial_rank = 3 #3D loss
        n_ch = U.size(1)
        
        spatial_x, spatial_y, spatial_z = torch.meshgrid(torch.arange(x_shape).cuda(), 
                                                         torch.arange(y_shape).cuda(), 
                                                         torch.arange(z_shape).cuda())

        

        if self._alpha>0:
            spatial_coords = torch.stack([spatial_x, spatial_y, spatial_z], 0)
            spatial_coords = spatial_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            spatial_coords = spatial_coords.type(torch.cuda.FloatTensor).detach()
            features = torch.cat([spatial_coords / self._alpha, I / self._beta], 1)

        else:
            features = I / self._beta
            
        if not self.is_da:
            if self._alpha>0:
                features = torch.reshape(features, (batch_size, n_feat + spatial_rank, -1))
            else:
                features = torch.reshape(features, (batch_size, n_feat, -1))
            ones = torch.ones(batch_size, 1, nb_voxels, device='cuda') 
            
        else:
            features = torch.reshape(features, (1, n_feat, -1))
            ones = torch.ones(1, 1, batch_size*nb_voxels, device='cuda')

        if self.use_norm:
            spatial_norm = self.pl(bilateral_coords, ones)
            spatial_norm = 1.0 / torch.sqrt(spatial_norm + 1e-20)
        
        if not self.is_da:
            U = torch.reshape(U, [batch_size, n_ch, -1])
        else:
            U = torch.reshape(U, [1, n_ch, -1])
        
        if not self.is_da:
            H1 = torch.nn.Softmax(1)(U)
        else:
            H1 = U
        
        if self.use_norm:
            Q1 = self.pl(features, H1 * spatial_norm)
            Q1 = Q1 * spatial_norm
        else:
            Q1 = self.pl(features, H1)

        Q1 = torch.reshape(Q1, [-1, n_ch])
        H1 = torch.reshape(torch.nn.Softmax(1)(U), [-1, n_ch])

        loss = torch.matmul(Q1.T, 1-H1)
        loss = torch.trace(loss)
        return loss

