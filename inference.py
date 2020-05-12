#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 

from network.unet import UNet2D5
from tqdm import tqdm
import SimpleITK as sitk
import sys
import torchio
from torchio import ImagesDataset, Image, Subject, Queue, DATA


import multiprocessing as mp

from torchvision.transforms import Compose

from torchio.data.sampler import ImageSampler
from torchio.transforms import (    
        ZNormalization,
        RandomMotion, 
        CenterCropOrPad,
        Rescale,    
        RandomNoise,    
        RandomFlip, 
        RandomAffine,   
        ToCanonical,    
        Resample    
    )   

from utilities.sampling import GridSampler, GridAggregator  
import nibabel as nib   
import pandas as pd 
from torch import nn 
#from apex import amp    
# Define training and patches sampling parameters   
patch_size = (128,128,128)
NB_CLASSES = 2

MODALITIES = ['t2']

def inference_padding(paths_dict, 
                      model,
                      transformation,
                      device, 
                      pred_path,
                      cp_path,
                      opt):

    model.load_state_dict(torch.load(cp_path))
    model.to(device)
    model.eval()

    subjects_dataset_inf = ImagesDataset(
        paths_dict, 
        transform=transformation)

   # batch_loader_inf = DataLoader(subjects_dataset_inf, batch_size=1)
    #window_size = (256,256,256)
    window_size = patch_size
    border = (0,0,0)
    for batch in tqdm(subjects_dataset_inf):
        batch_pad = CenterCropOrPad((288,128,48))(batch)
        mod_used = MODALITIES[-1]

        data = batch_pad[mod_used][DATA].cuda().unsqueeze(0)


        reference = torchio.utils.nib_to_sitk(batch[mod_used][DATA].numpy(), batch[mod_used]['affine'])

        
        affine_pad = batch_pad[mod_used]['affine']
        name = batch[mod_used]['stem']

        with torch.no_grad():
            logits, _ = model(data, 'source')
            labels = logits.argmax(dim=1, keepdim=True)
            labels = labels[0,0,...].cpu().numpy()
        output = labels

        output = torchio.utils.nib_to_sitk(output.astype(float), affine_pad)
        output = sitk.Resample(
            output,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
        )
        sitk.WriteImage(output, pred_path.format(name))



def main():
    opt = parsing_data()


    
    print("[INFO] Reading data.")
    # Dictionary with data parameters for NiftyNet Reader
    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(
            "[INFO] No GPU found or Wrong gpu id, please run without --cuda")
        

    
    # FOLDERS
    fold_dir = opt.model_dir
    checkpoint_path = os.path.join(fold_dir,'models', './CP_{}.pth')
    checkpoint_path = checkpoint_path.format(opt.epoch_infe)
    assert os.path.isfile(checkpoint_path), 'no checkpoint found'
    
    if opt.output_dir is None:
        output_path = os.path.join(fold_dir,'output')
    else:
        output_path = opt.output_dir

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path,'output_{}.nii.gz')
    
    # SPLITS
    split_path = opt.dataset_split
    assert os.path.isfile(split_path), 'split file not found'
    print('Split file found: {}'.format(split_path))
    

    # Reading csv file
    df_split = pd.read_csv(split_path,header =None)
    list_file = dict()
    list_split = ['inference', 'validation']
    for split in list_split:
        list_file[split] = df_split[df_split[1].isin([split.lower()])][0].tolist()
 
    # filing paths
    paths_dict = {split:[] for split in list_split}
    for split in list_split:
        for subject in list_file[split]:
            subject_data = []
            for modality in MODALITIES:
                subject_modality = opt.path_file+subject+modality+'.nii.gz'
                if os.path.isfile(subject_modality):
                    subject_data.append(Image(modality, subject_modality, torchio.INTENSITY))
                    #subject_data.append(Image(modality, path_file[domain]+subject+modality+'.nii.gz', torchio.INTENSITY))
            if len(subject_data)>0:
                paths_dict[split].append(Subject(*subject_data))
    
        
    transform_inference = (
            ToCanonical(),
            ZNormalization(),      
        )   
    transform_inference = Compose(transform_inference) 
     
    # MODEL 
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}  
    dropout_op_kwargs = {'p': 0, 'inplace': True}   
    net_nonlin = nn.LeakyReLU   
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}   

    print("[INFO] Building model.")  
    model= UNet2D5(input_channels=1,   
                base_num_features=16,   
                num_classes=NB_CLASSES,     
                num_pool=4,   
                conv_op=nn.Conv3d,    
                norm_op=nn.InstanceNorm3d,    
                norm_op_kwargs=norm_op_kwargs,  
                nonlin=net_nonlin,  
                nonlin_kwargs=net_nonlin_kwargs)
  

    paths_inf = paths_dict['inference']+paths_dict['validation']
    inference_padding(paths_inf, model, transform_inference, device, output_path, checkpoint_path, opt)



def parsing_data():
    parser = argparse.ArgumentParser(
        description='3D Segmentation Using PyTorch and NiftyNet')

    parser.add_argument('-epoch_infe',
                        type=str,
                        default = 'best')

    parser.add_argument('-model_dir',
                        type=str)

    parser.add_argument('-dataset_split',
                        type=str,
                        default='dataset_split.csv')

    parser.add_argument('-path_file',
                        type=str,
                        default='../data/VS_T1/target/')

    parser.add_argument('-add_sym',
                        type=int,
                        default=0,
                        choices=[0,1])

    parser.add_argument('-output_dir',
                        type=str,
                        default=None)

    parser.add_argument('-nb_classes',
                        type=int,
                        default=10)

    parser.add_argument('-modalities',
                        default=MODALITIES)

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()



