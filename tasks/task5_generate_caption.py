import os
import json
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer


######################### CONFIG START #########################
from config import *

label_mask_dir = '../data/visualization/ROI'
global_dataset_path = '../data/dataset'
label_txt = '../data/dataset/ADCriolabel/62ROIs_for_Children.txt'
grading_gt_path = '../data/visualization/atlases/grading_gt.json'
area_and_mri_score_gt_path = '../data/visualization/atlases/roi_gt_0.225_with_injury_score_new_more.json'

mode = '62'

task1_pattern = 'example_flash_1_5_flash_001_task1_'
task2_pattern = 'example_flash_1_5_flash_001_task2_'
task2_5_pattern = f'rare_location_report_{task2_pattern[:-1]}'
task3_pattern = f'example_flash_1_5_flash_001_task3_{mode}_clinic_v_lesion_grading_'
task5_pattern = f'example_flash_1_5_flash_001_task4_{mode}_math_v_outcome_'

######################### CONFIG END #########################



ori_nib_label_dict = {}
with open(label_txt, 'r') as l_t:
    content = l_t.readlines()
    for line in content:
        line = line.strip()
        nib_idx = int(line.split('\t')[0])
        nib_label = line.split('\t')[1]
        ori_nib_label_dict[nib_idx] = nib_label


single_or_both = 'both'
task1_dir = os.path.join(global_ans_path, 'task1', single_or_both)
task2_dir = os.path.join(global_ans_path, 'task2', single_or_both)
task2_5_dir = os.path.join(global_ans_path, 'task2', single_or_both)
task3_dir = os.path.join(global_ans_path, 'task3', single_or_both)
task5_dir = os.path.join(global_ans_path, 'task5', single_or_both)
os.makedirs(task5_dir, exist_ok=True)


def load_gt(gt_path):
    gt = None
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    return gt

grading_gt = load_gt(grading_gt_path)
area_and_mri_score_gt = load_gt(area_and_mri_score_gt_path)

def load_pattern_file(task_dir, task_pattern):
    task_files = os.listdir(task_dir)
    cur_task_file_list = [file for file in task_files if task_pattern in file]
    cur_ans = {}
    for ans_file in cur_task_file_list:
        cur_split_ans = None
        with open(os.path.join(task_dir, ans_file), 'r') as f:
            cur_split_ans = json.load(f)
            cur_ans.update(cur_split_ans)
    if len(cur_ans.keys()) == 0:
        print(f'[Error] cur pattern {task_pattern} has error!')
        import pdb; pdb.set_trace()

    return cur_ans

task1_ans = load_pattern_file(task1_dir, task1_pattern)
task2_ans = load_pattern_file(task2_dir, task2_pattern)
task2_5_ans = load_pattern_file(task2_dir, task2_5_pattern)
task3_ans = load_pattern_file(task3_dir, task3_pattern)


def generate_caption(case_id, task1, task2, task2_5, task3):
    sentence1_pos = f'The {case_id} MRI shows a <task1> of the brian volume injured.'
    sentence1_neg = f'The {case_id} MRI shows no brian volume injured.'
    
    sentence2_pos = 'The affected regions include <task2>.'
    sentence2_neg = 'No regions are affected.'

    sentence3_pos = 'For uncommon areas, <task2_5> are not typical for common HIE injury regions.' 
    sentence3_neg = 'All lesioned regions are typical for common HIE injury regions.'
    
    sentence4 = 'The MRI injury score for this MRI is <task3>.'

    if task1 > 0.:
        caption = ' '.join([
            sentence1_pos.replace('<task1>', f'{round(task1 * 100, 2)}%'),
            sentence2_pos.replace('<task2>', ', '.join(task2)) if len(task2) > 0 else sentence2_neg,
            sentence3_pos.replace('<task2_5>', ', '.join(task2_5)) if len(task2_5) > 0 else sentence3_neg,
            sentence4.replace('<task3>', task3)
        ])
    else:
        caption = sentence1_neg
    
    return caption


def calculate_pair_rouge(ans_caption, gt_caption):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = scorer.score(gt_caption, ans_caption)

    rouge1 = [scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure]
    rouge2 = [scores['rouge2'].precision, scores['rouge2'].recall, scores['rouge2'].fmeasure]
    rougeL = [scores['rougeL'].precision, scores['rougeL'].recall, scores['rougeL'].fmeasure]

    # print("ROUGE-1:", scores['rouge1'])
    # print("ROUGE-2:", scores['rouge2'])
    # print("ROUGE-L:", scores['rougeL'])
    
    return rouge1, rouge2, rougeL
    
cnt = 0
total_rouge1 = np.zeros(3)
total_rouge2 = np.zeros(3)
total_rougeL = np.zeros(3)

caption_dict = {}

for task1_key, task1_value in tqdm(task1_ans.items()):
    task1 = float(task1_value['model_answer'][0].split(',')[1])

    task2_key = f"HIE_{task1_key.split('-')[0].split('_')[1]}"
    try:
        task2_ans_id = task2_ans[task2_key]['ans']
    except:
        print(f'[Error] task2_key {task2_key} not found!')
        continue
    try:
        task2 = [ori_nib_label_dict[cur_ans_id] for cur_ans_id in task2_ans_id if cur_ans_id in list(ori_nib_label_dict.keys())] if len(task2_ans_id) > 0 else []
    except:
        import pdb; pdb.set_trace()

    task2_5_ans_id = task2_5_ans[task2_key]['ans']
    task2_5 = [ori_nib_label_dict[cur_ans_id] for cur_ans_id in task2_5_ans_id if cur_ans_id in list(ori_nib_label_dict.keys())] if len(task2_5_ans_id) > 0 else []

    task3 = task3_ans[task1_key]['model_answer'][0].split(',')[0].split(':')[1].strip('.')
    
    caption = generate_caption(task2_key, task1, task2, task2_5, task3)

    gt_task1 = float(grading_gt[task1_key.split('-')[0]][1])
    area_and_mri_score_gt_tasks = area_and_mri_score_gt[task2_key]
    gt_task2_list = area_and_mri_score_gt_tasks['primary_region'] + area_and_mri_score_gt_tasks['uncommon_region']
    gt_task2 = [item.split(': ')[1] for item in gt_task2_list] if len(gt_task2_list) > 0 else []

    gt_task2_5 = [item.split(': ')[1] for item in area_and_mri_score_gt_tasks['uncommon_region']] if len(area_and_mri_score_gt_tasks['uncommon_region']) > 0 else []
    
    gt_task3 = f"Score{area_and_mri_score_gt_tasks['mri_injury_score']}"
    gt_caption = generate_caption(task2_key, gt_task1, gt_task2, gt_task2_5, gt_task3)

    # [precision, recall, f1]
    rouge1, rouge2, rougeL = calculate_pair_rouge(caption, gt_caption)

    total_rouge1 += np.array(rouge1)
    total_rouge2 += np.array(rouge2)
    total_rougeL += np.array(rougeL)
    cnt += 1


    caption_dict[task2_key] = {
        'gt': gt_caption,
        'ans': caption,
        'rouge1_f1': str(rouge1[2]),
        'rouge2_f1': str(rouge2[2]),
        'rougeL_f1': str(rougeL[2]),
    }

    if cnt % 2 == 0:
        jsondump(os.path.join(task5_dir, task5_pattern), caption_dict)

jsondump(os.path.join(task5_dir, task5_pattern), caption_dict)

total_rouge1 /= cnt
total_rouge2 /= cnt
total_rougeL /= cnt

print(f"Average ROUGE-1: precision = {total_rouge1[0]}, recall = {total_rouge1[1]}, f1 = {total_rouge1[2]}")
print(f"Average ROUGE-2: precision = {total_rouge2[0]}, recall = {total_rouge2[1]}, f1 = {total_rouge2[2]}")
print(f"Average ROUGE-L: precision = {total_rougeL[0]}, recall = {total_rougeL[1]}, f1 = {total_rougeL[2]}")