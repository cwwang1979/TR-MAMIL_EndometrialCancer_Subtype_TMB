# Deep Learning Can Predict Key Biomarkers for Tumor Mutational Burden and Classify Aggressive and Non-aggressive Endometrial Cancer directly from Histopathology Whole Slide Images
Multilayer Attention-based Multiple Instance Deep Learning Methods in Application to Tumor Mutational Burden Assessment and Cancer Subtyping from H&amp;E-stained Whole Slide Images of Endometrial Cancer Samples

## Datasets
In this study, we utilized 918 anonymized whole section H\&E-stained WSIs collected from 529 patients of TCGA cohort. All EC images from TCGA (759 frozen, 144 diagnostic formalin-fixed paraffin-embedded and 15 error WSIs) were determined for TMB status by NGS [TMB-H: TMB $\geq$ 10  (186 patients); TMB-low: TMB < 10 (331 patients); NA: TMB data not available (12 patients).
The TCGA cohort was collected from 29 tissue source sites available in the public repositories at the National Institutes of Health, USA ([TCGA Link](https://portal.gdc.cancer.gov/))


## Setup

#### Requirerements
- ubuntu 18.04
- RAM >= 16 GB
- GPU Memory >= 12 GB
- GPU driver version >= 510.39
- CUDA version >= 11.6
- Python (3.7.7), h5py (2.10.0), matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), openslide-python (1.1.1), pandas (1.2.4), pillow (6.1.0), PyTorch (1.13.1+cu116), scikit-learn (0.22.1), scipy (1.4.1), tensorflow (1.14.0), tensorboardx (2.6), torchvision (0.14.1+cu116), pixman(0.38.0).

#### Download
Source code file, configuration file, and models are download from the [zip](https://drive.google.com/open?id=1h0tFM_OQ3Ls1ZC7JoyIemnIcJWG3sAER&usp=drive_copy) file.  (For reviewers, the password of the file is in the implementation section of the associated manuscript.)

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
python create_patches.py --source DATA --save_dir RESULTS_DIRECTORY/ --patch_size 256 --preset tcga.csv --seg --patch --stitch

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
For the classification of aggressive and non-aggressive, use resnet50 as the backbone, and for the TMB prediction and TP53, use resnet152 as the backbone by following this instruction:

Open the models/resnet_custom.py to modify the backbone for the feature extraction part:

For the classification of the aggressive and non-aggressive EC:
```
def resnet50_baseline(pretrained=False):
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model
```

For the TMB prediction and TP53 prediction in EC samples:
```
def resnet50_baseline(pretrained=False):
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 8, 36, 3])
    if pretrained:
        model = load_pretrained_weights(model, 'resnet152')
    return model
```

In the terminal run:
```
CUDA_VISIBLE_DEVICES=0,1 python extract_features.py --data_h5_dir RESULTS_DIRECTORY/ --data_slide_dir DATA --csv_path RESULTS_DIRECTORY/process_list_autogen.csv --feat_dir FEATURES_DIRECTORY_RESNETxxx/ --batch_size 512 --slide_ext .svs

```
change "--feat_dir FEATURES_DIRECTORY_RESNETxx/" with the specified backbone to save the features.


After running in the terminal, the extracted features will be produced as .pt file for each slide in folder named 'FEATURES_DIRECTORY_RESNETxx/' with specific backbone (e.g. "./FEATURES_DIRECTORY_RESNET50" for the classification of aggressive and non-aggressive and "./FEATURES_DIRECTORY_RESNET152" for the TMB prediction and TP53).

example features results for the feature extraction step:
```
FEATURES_DIRECTORY_RESNET152/
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

#### 3. Training and Testing List
Prepare the training and the testing list containing the labels of the files and put it into ./dataset_csv folder. (We provides the csv sample training and testing list in named "TMB_endometrial_train.csv" and "TMB_endometrial_test.csv")

example of the "TMB prediction for the aggressive subtype" training set list:
| slide_id       | case_id     | label   | covariate | 
| :---           |  :---       | :---:   |:---:| 
| slide_1        | slide_1     | TMBH   |   F | 
| slide_2        | slide_2     | TMBL   |   F |
| ...            | ...         | ...     | ... |
| slide_346        | slide_346     | TMBL   |   F |

 example of the"TMB prediction for the aggressive subtype" testing set list:
| slide_id       | case_id     | label   | covariate | 
| :---           |  :---       | :---:   |:---:| 
| slide_1        | slide_1     | TMBH   |   F | 
| slide_2        | slide_2     | TMBL   |   F |
| ...            | ...         | ...     | ... |
| slide_173       | slide_173     | TMBL   |   F |


#### 4. Inference and Evaluation

For inference, open the "eval_mtl_concat.py" and set the number of the classes, the label for each class and the testing list location ("TMB_endometrial_test.csv").
```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/TMB_endometrial_test.csv',
                            data_dir= os.path.join(args.data_root_dir,'pt_files'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'TMBL':0, 'TMBH':1}, {'F':0, 'M':1}],
                            label_cols = ['label', 'covariate'],
                            patient_strat= False)
else:
    raise NotImplementedError
```
Then run this code in the terminal:
```
CUDA_VISIBLE_DEVICES=0 python eval_mtl_concat.py --drop_out --k 1 --models_exp_code models --save_exp_code model_prediction --split all --task dummy_mtl_concat  --results_dir results --data_root_dir FEATURES_DIRECTORY_RESNETxxx
```

To assess the proposed methods for the classification of aggressive and non-aggressive: 
1. Load the saved models in the ./results folder by changing "--models_exp_code models" with "--models_exp_code TR-MAMIL_Subtyping"
2. change "--save_exp_code model_prediction" with the "--save_exp_code TR-MAMIL_Subtyping_prediction"
3. change the "--data_root_dir FEATURES_DIRECTORY" with the "FEATURES_DIRECTORY_RESNET50", 

To assess the proposed methods for the TMB prediction in aggressive : 
1. Load the saved models in the ./results folder by changing "--models_exp_code models" with "--models_exp_code TR-MAMIL_Aggressive"
2. change "--save_exp_code model_prediction" with the "--save_exp_code TR-MAMIL_Aggressive_prediction"
3. change the "--data_root_dir FEATURES_DIRECTORY" with the "FEATURES_DIRECTORY_RESNET152",

To assess the proposed methods for the TMB prediction in  non-aggressive: 
1. Load the saved models in the ./results folder by changing "--models_exp_code models" with "--models_exp_code TR-MAMIL_NonAggressive"
2. change "--save_exp_code model_prediction" with the "--save_exp_code TR-MAMIL_NonAggressive_prediction"
3. change the "--data_root_dir FEATURES_DIRECTORY" with the "FEATURES_DIRECTORY_RESNET152",

To assess the proposed methods for the Kplain-Meier survival analysis: 
1. Load the saved models in the ./results folder by changing "--models_exp_code models" with "--models_exp_code TR-MAMIL_KM"
2. change "--save_exp_code model_prediction" with the "--save_exp_code TR-MAMIL_KM_prediction"
3. change the "--data_root_dir FEATURES_DIRECTORY" with the "FEATURES_DIRECTORY_RESNET50",

Example for the proposed  in application to the prediction of TMB status for aggressive EC, run this in the terminal:
```
CUDA_VISIBLE_DEVICES=0 python eval_mtl_concat.py --drop_out --k 1 --models_exp_code TR-MAMIL_Aggressive --save_exp_code TR-MAMIL_Aggressive_prediction --split all --task dummy_mtl_concat  --results_dir results --data_root_dir FEATURES_DIRECTORY_RESNET152
```


These inference part will create a folder named TR-MAMIL_x_prediction in ./eval_results folder (e.g. ./eval_results/EVAL_TR-MAMIL_x_prediction) with this following structure:
```
./eval_results/EVAL_TR-MAMIL_x_prediction/
├── eval_experiment_TR-MAMIL_x_prediction.txt
├── fold0.txt  
└── summary.txt
```
the file "eval_experiment_TR-MAMIL_x_prediction.txt" will contain the configuration of the proposed method, the file "fold0.txt" will contain the probability and the prediction for each slides and for the evaluation part, access the file "summary.txt"

## Training
#### Preparing Training Splits

To create a splits for training and validation set from the training list automatically. The default proportion for the training:validation splits used in this study is 9:1. Do the stratified sampling by open the create_splits.py, and change this related code with the directory of the training csv, the number of classess and the labels we want to investigates. 
```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_WSI_MTL_Dataset(csv_path = 'dataset_csv/TMB_endometrial_train.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'TMBL':0, 'TMBH':1}, {'F':0, 'M':1}],
                            label_cols = ['label', 'covariate'],
                            patient_strat= False)
```

In the terminal run:
```
python create_splits.py --task dummy_mtl_concat --seed 1 --k 1

```

#### Training

Open the "main_mtl_concat.py" and  and change this related code with the directory of the training csv, the number of classess and the labels we want to investigates. 
```
if args.task == 'dummy_mtl_concat':
    args.n_classes=2
    dataset = Generic_MIL_MTL_Dataset(csv_path = 'dataset_csv/TMB_endometrial_train.csv',
                            data_dir= os.path.join(args.data_root_dir,'pt_files'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dicts = [{'TMBL':0, 'TMBH':1}, {'F':0, 'M':1}],
                            label_cols = ['label', 'covariate'],
                            patient_strat= False)
else:
    raise NotImplementedError
```

Opening the "core_utils_mtl_concat.py" and modify this related part:
1. For the classification of aggressive and non-aggressive and TMB prediction in non-aggressive, modified the model selection and the early stopping part with F1-Score.
2. For the TMB prediction in aggressive part, modified the model selection and the early stopping part with Cross Entropy.
```
"""F1-SCORE"""
def __call__(self, epoch, val_f1score, model, ckpt_name = 'checkpoint.pt'):

        score = val_f1score[0]
        print('score:', score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_f1score, model, ckpt_name)

        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_f1score, model, ckpt_name)

            self.counter = 0

    def save_checkpoint(self, val_f1score, model, ckpt_name):

        '''Saves model when F1-score increased.'''
        if self.verbose:
            print(f'F1-Score increased ({self.val_f1score_max} --> {val_f1score}).  Saving model ...')

        torch.save(model.state_dict(), ckpt_name)
        self.val_f1score_max = val_f1score
                  .
                  .
                  .

  if early_stopping:
        assert results_dir

        """Cross Entropy""" ###for the TMB prediction of aggressive subtype  
        early_stopping(epoch, cls_val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        """F1-score"""
        # early_stopping(epoch, cls_val_f1score, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

```
Then in a terminal run:
```
CUDA_VISIBLE_DEVICES=0 python main_mtl_concat.py --drop_out --early_stopping --lr 2e-4 --k 1 --exp_code saved_model  --task dummy_mtl_concat  --log_data  --data_root_dir FEATURES_DIRECTORY_RESNETxx
```
change "--exp_code saved_model" with the model name (e.g."--exp_code Proposed_Method_for_subtype") and the "--data_root_dir FEATURES_DIRECTORY_RESNETxx" with the features with the specified bacbone (e.g. --data_root_dir FEATURES_DIRECTORY_RESNET50)

## License
This Python source code is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology

