# Scribble-based Domain Adaptation via Co-segmentation

Public pytorch implementation for our paper [Scribble-based Domain Adaptation 
via Co-segmentation](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_47), 
which was accepted for presentation at [MICCAI 2020](https://www.miccai2020.org). 

If you find this code useful for your research, please cite the following paper:

```
@article{ScribDA2020Dorent,
         author={Dorent, Reuben and Joutard, Samuel and
         Shapey, Jonathan and Bisdas, Sotirios and
         Kitchen, Neil and Bradford, Robert and Saeed, Shakeel and
         Modat, Marc and Ourselin, S\'ebastien and Vercauteren, Tom},
         title={Scribble-based Domain Adaptation via Co-segmentation},
         journal={MICCAI},
         year={2020},
}
```

## Method overview


## Virtual Environment Setup

The code is implemented in Python 3.6 using using the PyTorch library. 
Requirements:

 * Set up a virtual environment (e.g. conda or virtualenv) with Python 3.6
 * Install all requirements using:
  
  ````pip install -r requirements.txt````
 * Install the cuda implementation of the permutohedral lattice.
````
cd ./Permutohedral_attention_module/PAM_cuda/
python3 setup.py build
python3 setup.py install 
````
  

## Data

The dataset (images, annotations, scribbles) used for our experiments will be released soon.

## Running the code
`train.py` is the main file for training the model.
```` python3 train.py \
-model_dir ./models/$ALPHA/$BETA/$GAMMA/$WARMUP/ \
-alpha $ALPHA \
-beta $BETA \
-gamma $GAMMA \
-warmup $WARMUP \
-path_source $SOURCE \
-path_target $TARGET \
-dataset_split_source $SPLIT_SOURCE \
-dataset_split_target $SPLIT_TARGET \
````
Where the hyperparameters are defined such as:
 * `$ALPHA`: Image specific spatial kernel for smoothing differences in coordinates. Typical value: 15.
 * `$BETA`: Image specific spatial kernel for smoothing differences in intensities. Typical value: 0.05
 * `$GAMMA`: Cross-domain spatial kernel for smoothing differences in feature values. Typical value: 0.1
 * `$WARMUP`: Define the number of initialisation epochs without the cross-modality regularisation.
 * `$SOURCE`: Path to the source data
 * `$TARGET`: Path to the target data
 * `$SPLIT_SOURCE`: CSV file with source image IDs and group ('inference' or 'training')
 * `$SPLIT_TARGET`: CSV file with target image IDs and group ('inference' or 'training')
 
## Using the code with your own data

If you want to use your own data, you just need to change the source and target paths, 
the splits and potentially the modality used.
