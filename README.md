# Multilayer-Attention-Multiple-Instance-Deep-Learning-Methods
Multilayer Attention-based Multiple Instance Deep Learning Methods in Application to Tumor Mutational Burden Assessment and Cancer Subtyping from H&amp;E-stained Whole Slide Images of Endometrial Cancer Samples

## Setup

#### Requirerements
- ubuntu 18.04
- RAM >= 16 GB
- GPU Memory >= 12 GB
- GPU driver version >= 510.39
- CUDA version >= 11.6
- Python (3.7.7), h5py (2.10.0), matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), openslide-python (1.1.1), pandas (1.2.4), pillow (6.1.0), PyTorch (1.13.1+cu116), scikit-learn (0.22.1), scipy (1.4.1), tensorflow (1.14.0), tensorboardx (2.6), torchvision (0.14.1+cu116), pixman(0.38.0).

#### Download
Execution file, configuration file, and models are download from the [zip](https://drive.google.com/file/d/1stRXbUX2nyTspVJoNsZvL5_0xKi20ExU/view?usp=sharing) file.  (For reviewers, "..._cwlab" is the password to decompress the file.)

## Steps

#### 1. Tissue Segmentation and Patching

Place the Whole slide image in ./DATA
```
./DATA/
├── slide_1.svs
├── slide_2.svs
│        ⋮
└── slide_n.svs
  
```

Then in a terminal run:
```
python create_patches_fp.py --source DATA --save_dir RESULTS_DIRECTORY/ --patch_size 256 --preset tcga.csv --seg --patch --stitch

```

After running in a terminal, the result will be produced in folder named 'RESULTS_DIRECTORY/', which includes the masks and the sticthes in .jpg and the coordinates of the patches will stored into HD5F files (.h5) like the following structure.
```
RESULTS_DIRECTORY/
├── masks/
│   ├── slide_1.jpg
│   ├── slide_2.jpg
│   │       ⋮
│   └── slide_n.jpg
│
├── patches/
│   ├── slide_1.h5
│   ├── slide_2.h5
│   │       ⋮
│   └── slide_n.h5
│
├── stitches/
│   ├── slide_1.jpg
│   ├── slide_2.jpg
│   │       ⋮
│   └── slide_n.jpg
│
└── process_list_autogen.csv
```


#### 2. Feature Extraction

In the terminal run:
```
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir RESULTS_DIRECTORY/ --data_slide_dir DATA --csv_path RESULTS_DIRECTORY/process_list_autogen.csv --feat_dir FEATURES_DIRECTORY/ --batch_size 512 --slide_ext .svs

```
After running in a terminal, the extracted features will be produced in folder named 'FEATURES_DIRECTORY/', like the following structure.

```
FEATURES_DIRECTORY/
├── h5_files/
│   ├── slide_1.h5
│   ├── slide_2.h5
│   │       ⋮
│   └── slide_n.h5
│
└── pt_files/
    ├── slide_1.pt
    ├── slide_2.pt
    │       ⋮
    └── slide_n.pt
```
Open the models/resnet_custom.py to modify the backbone for the feature extraction part:
```
def resnet50_baseline(pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
          # for 18 layers = Bottleneck_Baseline, [2, 2, 2, 2])
    # for 50 layers = Bottleneck_Baseline, [3, 4, 6, 3])
    # for 101 layers = Bottleneck_Baseline, [3, 4, 23, 3])
    # for 152 layers = Bottleneck_Baseline, [3, 8, 36, 3])
    
    """
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model
```

#### 3. Training Splits
Prepare the training and the testing list containing the labels of the files and put it into ./dataset_csv folder

TMB_endometrial_train.csv
| slide_id       | case_id     | label   | sex | 
| :---           |  :---       | :---:   |:---:| 
| slide_1        | slide_1     | TMBH   |   F | 
| slide_2        | slide_2     | TMBL   |   F |
| ...            | ...         | ...     | ... |
| slide_n        | slide_n     | TMBL   |   F |


To create a splits for training, validation, and evaluation set automatically, do stratified sampling by open the create_splits.py, and change this related code with the specific task we want to investigates
```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_WSI_MTL_Dataset(csv_path = 'dataset_csv/TMB_endometrial_train.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'TMBL':0, 'TMBH':1}, {'F':0, 'M':1}],
                            label_cols = ['label', 'sex'],
                            patient_strat= False)
```
In the terminal run:
```
python create_splits.py --task dummy_mtl_concat --seed 1 --k 1

```

#### 3. Training
Open the "main_mtl_concat.py" and set the task and the training list location ("TMB_endometrial_train.csv").

```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/TMB_endometrial_train.csv',
                            data_dir= os.path.join(args.data_root_dir,'pt_files'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'TMBL':0, 'TMBH':1}, {'F':0, 'M':1}],
                            label_cols = ['label', 'sex'],
                            patient_strat= False)
else:
    raise NotImplementedError
```

Then in a terminal run:
```
CUDA_VISIBLE_DEVICES=0 python main_mtl_concat.py --drop_out --early_stopping --lr 2e-4 --k 1 --exp_code folder_1  --task dummy_mtl_concat  --log_data  --data_root_dir FEATURES_DIRECTORY

```

#### 4. Evaluation
After the training is completed, open the "eval_mtl_concat.py" and set the task and the evaluation list location ("TMB_endometrial_test.csv").

```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/TMB_endometrial_test.csv',
                            data_dir= os.path.join(args.data_root_dir,'pt_files'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'TMBL':0, 'TMBH':1}, {'F':0, 'M':1}],
                            label_cols = ['label', 'sex'],
                            patient_strat= False)
else:
    raise NotImplementedError
```

Then in a terminal run:
```
CUDA_VISIBLE_DEVICES=0 python eval_mtl_concat.py --drop_out --k 1 --models_exp_code folder_1_s1 --save_exp_code folder_1_s1_eval --split all --task dummy_mtl_concat  --results_dir results --data_root_dir FEATURES_DIRECTORY

```

## License
This extension to the Caffe library is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology

