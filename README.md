# CcBHLA-pan-specific-peptide-HLA-class-I-binding-prediction-via-Convolutional-and-BiLSTM features
This is the code for "CcBHLA: pan-specific peptide-HLA class I binding prediction via Convolutional and BiLSTM features" paper. 

Conda Environemt Setup
tensorflow==2.0

Step 1. Train the model 
Run the following command in the terminal

python train.py --task fold0 --test_data val_data_fold0

Step 2. Test the model 
Run the following command in the terminal

python test.py --ckpt_path ckpt/fold0.h5 --test_file independent_set
