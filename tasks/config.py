import os
import json
import numpy as np

######################### CONFIG START #########################
GOOGLE_API_KEY=''
dataset_dir = '../data'

train_input_dir = os.path.join(dataset_dir, 'dataset', 'Train')
val_input_dir = os.path.join(dataset_dir, 'dataset', 'Val')
test_input_dir = os.path.join(dataset_dir, 'dataset', 'Test')

global_vis_path = os.path.join(dataset_dir, 'visualization')
global_ans_path = os.path.join(dataset_dir, 'answers')

mgh_train_label = '../data/dataset/outcome/train.npy'
mgh_test_label = '../data/dataset/outcome/test.npy'
mgh_label = np.load(mgh_train_label)
mgh_label_test = np.load(mgh_test_label)

######################### CONFIG END #########################

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f,indent=4)
    f.close()
