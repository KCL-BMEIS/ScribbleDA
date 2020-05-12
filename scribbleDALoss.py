import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time

from Permutohedral_attention_module.PAM_cuda.pl import PermutohedralLattice as pl

class CRFLoss(nn.Module):
    def __init__(self,
                 alpha=5.,
                 beta=5.,
                 gamma=5.,
                 w=[1.0,0.0],
                 is_da=False):
        super(CRFLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._w = w
        self.pl = pl.apply
        self.is_da = is_da

    def forward(self, I, U):
        """
        Compute `T` iterations of mean field update given a dense CRF.
        This layer maintains has NO trainable parameters
        (neither a compatibility function nor `m` kernel weights).
        :param I: feature maps used in the dense pairwise term of CRF
        :param U: activation maps used in the unary term of CRF (before Softmax)
        :return: Maximum a posteriori labeling (before Softmax)
        """
        batch_size, n_feat, x_shape, y_shape, z_shape = I.size()
        nb_voxels = x_shape*y_shape*z_shape 
        spatial_rank = 3
        n_ch = U.size(1)
        
        spatial_x, spatial_y, spatial_z = torch.meshgrid(torch.arange(x_shape).cuda(), 
                                                         torch.arange(y_shape).cuda(), 
                                                         torch.arange(z_shape).cuda())

        

        if self._alpha>0:
            spatial_coords = torch.stack([spatial_x, spatial_y, spatial_z], 0)
            spatial_coords = spatial_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            spatial_coords = spatial_coords.type(torch.cuda.FloatTensor).detach()
            bilateral_coords = torch.cat([spatial_coords / self._alpha, I / self._beta], 1)

        else:
            bilateral_coords = I / self._beta
            
        if not self.is_da:
            if self._alpha>0:
                bilateral_coords = torch.reshape(bilateral_coords, (batch_size, n_feat + spatial_rank, -1))
            else:
                bilateral_coords = torch.reshape(bilateral_coords, (batch_size, n_feat, -1))
            ones = torch.ones(batch_size, 1, nb_voxels, device='cuda') 
            
        else:
            bilateral_coords = torch.reshape(bilateral_coords, (1, n_feat, -1))
            ones = torch.ones(1, 1, batch_size*nb_voxels, device='cuda')



        features = [bilateral_coords]


        norms = []
        
        for idx, feat in enumerate(features):
            spatial_norm = self.pl(feat, ones)
            spatial_norm = 1.0 / torch.sqrt(spatial_norm + 1e-20)
            norms.append(spatial_norm)
        
        if not self.is_da:
            U = torch.reshape(U, [batch_size, n_ch, -1])
        else:
            U = torch.reshape(U, [1, n_ch, -1])
        
        if not self.is_da:
            H1 = torch.nn.Softmax(1)(U)
        else:
            H1 = U
        Q1 = 0
        for idx, feat in enumerate(features):
            Q = self.pl(feat, H1 * norms[idx])
            Q1 += Q * norms[idx]

        Q1 = torch.reshape(Q1, [-1, n_ch])
        H1 = torch.reshape(torch.nn.Softmax(1)(U), [-1, n_ch])

        loss = torch.matmul(Q1.T, 1-H1)
        loss = torch.trace(loss)
        return loss

