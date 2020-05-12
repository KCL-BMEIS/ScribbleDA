#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
import sys
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
import nibabel as nib   
import pandas as pd
import SimpleITK as sitk 

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler 
from torch import nn 
from torchvision.transforms import Compose


import torchio
from torchio import ImagesDataset, Image, Subject, Queue, DATA
from torchio.data.sampler import ImageSampler
from torchio.transforms import (    
        ZNormalization, 
        CenterCropOrPad,
        Rescale,    
        RandomNoise,    
        RandomFlip, 
        RandomAffine,   
        ToCanonical,    
        Resample    
    )    


from utilities.loss_function import DC_CE
from utilities.sampling import GridSampler, GridAggregator
from scribbleDALoss import CRFLoss  
from network.unet import UNet2D5


#from apex import amp    
# Define training and patches sampling parameters   
num_epochs_max = 10000  
patch_size = {'source':(288,128,48), 'target':(288,128,48)}

nb_voxels = {d:np.prod(v) for d,v in patch_size.items()}
queue_length = 16
samples_per_volume = 1
batch_size = 2

NB_CLASSES = 2


# Training parameters
val_eval_criterion_alpha = 0.95
train_loss_MA_alpha = 0.95
nb_patience = 10
patience_lr = 5
weight_decay = 1e-5

MODALITIES_SOURCE = ['t1']
MODALITIES_TARGET = ['t2']

MODALITIES = {'source':MODALITIES_SOURCE, 'target':MODALITIES_TARGET}


def onehot(gt,shape):
    with torch.no_grad():
        shp_y = gt.shape
        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, gt, 1)
    return y_onehot

def scribble_loss(outputs, scribbles, criterion):
    nb_target = outputs.shape[0]
    loss_target = 0.0
    for i in range(nb_target):
        outputs_i = outputs[i,...].reshape(NB_CLASSES, -1).unsqueeze(0)
        scribbles_i = scribbles[i,...].reshape(-1)
        
        outputs_i = outputs_i[:,:,scribbles_i<12]
        nb_inf_12 = outputs_i.shape[-1]
        outputs_i= outputs_i.reshape(1,NB_CLASSES,1,1,nb_inf_12)
        scribbles_i = scribbles_i[scribbles_i<12].reshape(1,1,1,1,nb_inf_12)
        loss_target += criterion(outputs_i, scribbles_i.type(torch.cuda.IntTensor))
    return loss_target


def infinite_iterable(i):
    while True:
        yield from i

def train(paths_dict, model, transformation, criterion,
        device, save_path, opt):
    
    since = time.time()

    dataloaders = dict()
    # Define transforms for data normalization and augmentation
    for domain in ['source', 'target']:
        subjects_domain_train = ImagesDataset(
            paths_dict[domain]['training'], 
            transform=transformation['training'][domain])

        subjects_domain_val = ImagesDataset(
            paths_dict[domain]['validation'], 
            transform=transformation['validation'][domain])

        # Number of workers
        workers = 10
        
        batch_loader_domain_train = infinite_iterable(DataLoader(subjects_domain_train, batch_size=batch_size))
        batch_loader_domain_val = infinite_iterable(DataLoader(subjects_domain_val, batch_size=batch_size))

        dataloaders_domain = dict()
        dataloaders_domain['training'] = batch_loader_domain_train
        dataloaders_domain['validation'] = batch_loader_domain_val
        dataloaders[domain] = dataloaders_domain

    
    # Training parameters are saved 
    df_path = os.path.join(opt.model_dir,'log.csv')
    if os.path.isfile(df_path): # If the training already started
        df = pd.read_csv(df_path, index_col=False)
        epoch = df.iloc[-1]['epoch']
        best_epoch = df.iloc[-1]['best_epoch']

        val_eval_criterion_MA = df.iloc[-1]['MA']
        best_val_eval_criterion_MA = df.iloc[-1]['best_MA']

        initial_lr = df.iloc[-1]['lr']

        model.load_state_dict(torch.load(save_path.format('best')))

    else: # If training from scratch
        df = pd.DataFrame(columns=['epoch','best_epoch', 'MA', 'best_MA', 'lr'])
        val_eval_criterion_MA = None
        best_epoch = 0
        epoch = 0
        initial_lr = opt.learning_rate


    model = model.to(device)

    # Optimisation policy
    optimizer = torch.optim.Adam(model.parameters(), initial_lr, weight_decay=weight_decay, amsgrad=True)
    lr_s = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                           patience=patience_lr,
                                                           verbose=True, 
                                                           threshold=1e-3,
                                                           threshold_mode="abs")

    


    # Loop parameters
    continue_training = True
    ind_batch_train = np.arange(0, samples_per_volume*len(paths_dict['source']['training']), batch_size)
    ind_batch_val = np.arange(0, samples_per_volume*max(len(paths_dict['source']['validation']),len(paths_dict['target']['validation'])), batch_size)
    ind_batch= dict()
    ind_batch['training'] = ind_batch_train
    ind_batch['validation'] = ind_batch_val


    # Loss initialisation
    crf_l = CRFLoss(alpha=opt.alpha, beta=opt.beta, is_da=False)
    crf_l_da = CRFLoss(alpha=0, beta=opt.beta_da, is_da=True)


    while continue_training:
        epoch+=1
        print('-' * 10)
        print('Epoch {}/'.format(epoch))
        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group['lr']))
            
        
        # Each epoch has a training and validation phase
        for phase in ['training','validation']:
            print(phase)
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_loss_target = 0.0
            running_loss_source = 0.0
            epoch_samples = 0

            # Iterate over data
            for _ in tqdm(ind_batch[phase]):
                # Next source batch
                batch_source = next(dataloaders['source'][phase])
                labels_source = batch_source['label'][DATA].to(device).type(torch.cuda.IntTensor)
                inputs_source = torch.cat([batch_source[k][DATA] for k in MODALITIES_SOURCE],1).to(device)

                # Next target batch
                batch_target= next(dataloaders['target'][phase])
                scribbles_target = batch_target['scribble'][DATA].to(device)
                inputs_target = torch.cat([batch_target[k][DATA] for k in MODALITIES_TARGET],1).to(device)
                


                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    
                    outputs, features = model(torch.cat([inputs_source,inputs_target],0), 'source')

                    outputs_source, features_source = outputs[:batch_size,...], features[:batch_size,...]
                    outputs_target, features_target = outputs[batch_size:,...], features[batch_size:,...]

                    # Loss Source with full Labels
                    loss_source = criterion(outputs_source, labels_source)

                    # Loss Target on Scribbles
                    loss_target = scribble_loss(outputs_target, scribbles_target, criterion)

                    # Within scans regularisation (target only)
                    if (opt.beta>0 or opt.alpha>0) and phase == 'training':
                        reg_target = opt.weight_crf/nb_voxels['target']*crf_l(inputs_target, outputs_target)
                    else:
                        reg_target = 0.0

                    # Pairwise scans regularisation (DA)
                    if opt.beta_da>0 and phase == 'training' and opt.warmup>epoch:
                        index = torch.LongTensor(2).random_(0, features_source.shape[1])

                        features_crf = [features_source[:,index,...], features_target[:,index,...]]
                        features_crf = torch.cat(features_crf,0).detach().cuda()

                        prob = [onehot(labels_source,outputs_source.shape), torch.nn.Softmax(1)(outputs_target)]
                        prob = torch.cat(prob,0)

                        reg_da = opt.weight_crf/nb_voxels['target']*crf_l_da(
                                    I=features_crf, 
                                    U=prob)
                    else:
                        reg_da = 0.0

                    
                    if phase == 'training':
                        loss = loss_source + loss_target  + reg_target + reg_da
                    else:
                        loss = loss_source + loss_target
                    
                        
                    
                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()
                        

                # statistics
                epoch_samples += 1
                running_loss += loss.item()
                running_loss_source += loss_source.item()
                running_loss_target += loss_target.item()  

            epoch_loss = running_loss / epoch_samples
            epoch_loss_source = running_loss_source / epoch_samples
            epoch_loss_target = running_loss_target / epoch_samples

            
            print('{}  Loss Seg Source: {:.4f}'.format(
                phase, epoch_loss_source))
            print('{}  Loss Seg Target: {:.4f}'.format(
                phase, epoch_loss_target))
                    

            if phase == 'validation':
                if val_eval_criterion_MA is None: # first iteration
                    val_eval_criterion_MA = epoch_loss
                    best_val_eval_criterion_MA = val_eval_criterion_MA

                else: #update criterion
                    val_eval_criterion_MA = val_eval_criterion_alpha * val_eval_criterion_MA + (
                                1 - val_eval_criterion_alpha) * epoch_loss

                df = df.append({'epoch':epoch,
                                'best_epoch':best_epoch,
                                'MA':val_eval_criterion_MA,
                                'best_MA':best_val_eval_criterion_MA,  
                                'lr':param_group['lr']}, ignore_index=True)
                df.to_csv(df_path, index=False)

                lr_s.step(val_eval_criterion_MA)

                if val_eval_criterion_MA < best_val_eval_criterion_MA:
                    best_val_eval_criterion_MA = val_eval_criterion_MA
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path.format('best'))
               
                else:
                    if epoch-best_epoch>nb_patience:
                        continue_training=False

                if epoch==opt.warmup:
                    torch.save(model.state_dict(), save_path.format('warmup'))

    
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best epoch is {}'.format(best_epoch))


def main():
    opt = parsing_data()

    print("[INFO] Reading data")
    # Dictionary with data parameters for NiftyNet Reader
    if torch.cuda.is_available():
        print('[INFO] GPU available.')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise Exception(
            "[INFO] No GPU found or Wrong gpu id, please run without --cuda")
        
    # FOLDERS
    fold_dir = opt.model_dir
    fold_dir_model = os.path.join(fold_dir,'models')
    if not os.path.exists(fold_dir_model):
        os.makedirs(fold_dir_model)
    save_path = os.path.join(fold_dir_model,'./CP_{}.pth')

    output_path = os.path.join(fold_dir,'output')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path,'output_{}.nii.gz')
    
    
    # LOGGING
    orig_stdout = sys.stdout
    if os.path.exists(os.path.join(fold_dir,'out.txt')):
        compt = 0
        while os.path.exists(os.path.join(fold_dir,'out_'+str(compt)+'.txt')):
            compt+=1
        f = open(os.path.join(fold_dir,'out_'+str(compt)+'.txt'), 'w')
    else:
        f = open(os.path.join(fold_dir,'out.txt'), 'w')
    #sys.stdout = f

    print("[INFO] Hyperparameters")
    print('Alpha: {}'.format(opt.alpha))
    print('Beta: {}'.format(opt.beta))
    print('Beta_DA: {}'.format(opt.beta_da))
    print('Weight Reg: {}'.format(opt.weight_crf))



    # SPLITS
    split_path_source = opt.dataset_split_source
    assert os.path.isfile(split_path_source), 'source file not found'

    split_path_target = opt.dataset_split_target
    assert os.path.isfile(split_path_target), 'target file not found'

    split_path = dict()
    split_path['source'] = split_path_source
    split_path['target'] = split_path_target

    path_file = dict()
    path_file['source'] = opt.path_source
    path_file['target'] = opt.path_target

    list_split = ['training', 'validation', 'inference']
    paths_dict = dict()

    for domain in ['source','target']:
        df_split = pd.read_csv(split_path[domain],header =None)
        list_file = dict()
        for split in list_split:
            list_file[split] = df_split[df_split[1].isin([split])][0].tolist()
        
        list_file['inference'] += list_file['validation']

        paths_dict_domain = {split:[] for split in list_split}
        for split in list_split:
            for subject in list_file[split]:
                subject_data = []
                for modality in MODALITIES[domain]:
                    subject_data.append(Image(modality, path_file[domain]+subject+modality+'.nii.gz', torchio.INTENSITY))
                if split in ['training', 'validation']:
                    if domain =='source':
                        subject_data.append(Image('label', path_file[domain]+subject+'t1_seg.nii.gz', torchio.LABEL))
                    else:
                        subject_data.append(Image('scribble', path_file[domain]+subject+'t2scribble_cor.nii.gz', torchio.LABEL))
                    #subject_data[] = 
                paths_dict_domain[split].append(Subject(*subject_data))
            print(domain, split, len(paths_dict_domain[split]))
        paths_dict[domain] = paths_dict_domain
            


    # PREPROCESSING
    transform_training = dict()
    transform_validation = dict()

    for domain in ['source', 'target']:
        transformations = (
            ToCanonical(),
            ZNormalization(),
            CenterCropOrPad((288,128,48)),
            RandomAffine(scales=(0.9, 1.1), degrees=10),
            RandomNoise(std_range=(0, 0.10)),
            RandomFlip(axes=(0,)), 
            )   
        transform_training[domain] = Compose(transformations)   

    for domain in ['source', 'target']:
        transformations = (
            ToCanonical(),
            ZNormalization(),
            CenterCropOrPad((288,128,48))
            )   
        transform_validation[domain] = Compose(transformations) 
     
    transform = {'training': transform_training, 'validation':transform_validation}   
    
    # MODEL 
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}  
    net_nonlin = nn.LeakyReLU   
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}   
    
    
    print("[INFO] Building model")  
    model= UNet2D5(input_channels=1,   
                base_num_features=16,   
                num_classes=NB_CLASSES,     
                num_pool=4,   
                conv_op=nn.Conv3d,    
                norm_op=nn.InstanceNorm3d,    
                norm_op_kwargs=norm_op_kwargs,  
                nonlin=net_nonlin,  
                nonlin_kwargs=net_nonlin_kwargs)
  
    
    print("[INFO] Training")
    #criterion = DC_and_CE_loss({}, {})
    criterion = DC_CE(NB_CLASSES)
    

    train(paths_dict, 
        model, 
        transform, 
        criterion, 
        device, 
        save_path,
        opt)

    #sys.stdout = orig_stdout
    #f.close()


def parsing_data():
    parser = argparse.ArgumentParser(
        description='3D Segmentation Using PyTorch and NiftyNet')

    parser.add_argument('-model_dir',
                        type=str)


    parser.add_argument('-weight_crf',
                    type=float,
                    default=0.1)

    parser.add_argument('-alpha',
                    type=float,
                    default=0)

    parser.add_argument('-beta',
                    type=float,
                    default=0.1)

    parser.add_argument('-beta_da',
                    type=float,
                    default=0)

    parser.add_argument('-dataset_split_target',
                        type=str,
                        default='./split/split_t2_training_30.csv')

    parser.add_argument('-dataset_split_source',
                        type=str,
                        default='./split/dataset_split_source.csv')

    parser.add_argument('-path_source',
                        type=str,
                        default='../data/VS_T1/source/')

    parser.add_argument('-path_target',
                        type=str,
                        default='../data/VS_T1/target/')


    parser.add_argument('-learning_rate',
                    type=float,
                    default=5*1e-4)


    parser.add_argument('-warmup',
                    type=int,
                    default=60)


    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()



