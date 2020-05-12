#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:55:35 2018

@author: reubendo
"""
import numpy as np
import nibabel 
import os

import sys
import pandas as pd
import argparse

from  medpy import metric


def dice_score(gt,pred):
    if np.sum(gt)>0 and np.sum(pred)>0:
        true_pos = np.float(np.sum(gt * pred ))
        union = np.float(np.sum(gt) + np.sum(pred))
        dice = true_pos * 2.0 / union
    elif np.sum(gt)==0 and np.sum(pred)==0:
        dice = 1
    else:
        dice = 0 
    return dice



def total_score(pred_path,gt_path, file_names, metric_name):
    print(pred_path)
    
    
    gt_to_pred = {k:k for k in [0,1]}
    list_labels = sorted(gt_to_pred.keys())
    score = dict()
    thre = 400
    score['names'] = []
    score['lesion'] = []
    
    
    for name in file_names:
        ground_truth = gt_path.format(name)
        ground_truth = os.path.expanduser(ground_truth)
        image_gt = nibabel.load(ground_truth)
        image_gt= nibabel.funcs.as_closest_canonical(image_gt).get_data()
        image_gt = image_gt.reshape(image_gt.shape[:3])
        
        
        pred = pred_path.format(name)
        pred = os.path.expanduser(pred)
        image_pred = nibabel.load(pred)
        affine = image_pred.affine
        voxel = [affine[0,0],affine[1,1],affine[2,2]]
        image_pred = image_pred.get_data()
        image_pred = image_pred.reshape(image_pred.shape[:3])


        score['names'].append(name)
        if metric_name=='assd':
            score['lesion'].append(metric.assd(image_gt,image_pred,voxel))
        elif metric_name=='rve':
            score['lesion'].append(metric.ravd(image_gt,image_pred))
        else:
            score['lesion'].append(metric.dc(image_gt,image_pred))


    print('Sample size: {}'.format(len(list(score.values())[0])))
    for label in score.keys():
        
        if label != 'names':
            print('Label: {}, {} mean: {}'.format(label, metric_name, round(np.mean(score[label]),2)))
            print('Label: {}, {} std: {}'.format(label, metric_name, round(np.std(score[label]),2)))
        

    return score
    #return [[round(100*np.mean(score[label]),1) for label in score.keys()], [round(100*np.std(score[label]),1) for label in score.keys()]]

gt_path = {'source':'../data/VS_T1/source/{}t2_seg.nii.gz',
'target': '../data/VS_T1/target/{}t2_seg.nii.gz'
}





def main():
    opt = parsing_data()

    path = opt.model_dir
    datasplit_path = opt.dataset_split
    # print command line arguments
    try:
        data_mean = []
        data_std = []
        #list_file = [k for k in os.listdir(path) if 'output_' in k]
        
        df_split = pd.read_csv(datasplit_path,header =None)
        list_file = ['output']
        for typ in np.unique(df_split[1].tolist()):
            print('---- score {} ----'.format(typ))
            file_names = df_split[df_split[1].isin([typ])][0].tolist()
            #file_names = [k for k in file_names]
            for k in list_file:
                
                path_k = os.path.join(path, k)
                
                #path_k = str(path_k) + '/output_30000/{}_niftynet_out.nii.gz'
                path_k = str(path_k) + '/output_{}t2.nii.gz'
                file_names_folder =  [name for name in file_names if os.path.exists(path_k.format(name))]
                if len(file_names_folder)>0:
                    if 'source' in typ:
                        dom = 'source'
                    else:
                        dom = 'target'
                    scores_k = total_score(path_k,gt_path[dom], file_names_folder, opt.metric)
                    df_k = pd.DataFrame(scores_k)
                    #print(df_k)
                    df_k.to_csv(path+'scores_mean_'+typ+'.csv')
             
    except Exception as e:
        print(e)


def parsing_data():
    parser = argparse.ArgumentParser(
        description='3D Segmentation Using PyTorch and NiftyNet')


    parser.add_argument('--model_dir',
                        type=str)

    parser.add_argument('--dataset_split',
                        type=str,
                        default='./splits/dataset_split_0.csv')

    parser.add_argument('--metric',
                        type=str,
                        default='dice',
                        choices=('dice', 'rve', 'assd'))

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":


    try:
        main()
    except Exception as e:
        print(e)






