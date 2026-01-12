import os
import json

######################### CONFIG START #########################

grading_gt_path = '../data/visualization/atlases/grading_gt.json'
task1_ans_dir = '../data/answers/task1/both/'
cur_ans_pattern = 'example_flash_1_5_flash_001_task1_'

task1_ans_list = os.listdir(task1_ans_dir)
cur_ans_task1_ans_list = [file for file in task1_ans_list if cur_ans_pattern in file]
######################### CONFIG END #########################

gt_content = None
with open(grading_gt_path, 'r') as gt:
    gt_content = json.load(gt)


ans_content = {}

for ans_file in cur_ans_task1_ans_list:
    cur_split_ans = None
    with open(os.path.join(task1_ans_dir, ans_file), 'r') as f:
        cur_split_ans = json.load(f)
        ans_content.update(cur_split_ans)

correct = 0
cnt = 0
abs_diff = 0.


import random

for key, value in ans_content.items():
    gt_key = key.split('-')[0]
    if 'Zmap_' in gt_key:
        gt_key = gt_key.replace('Zmap_', '')
    
    gt_level = gt_content[gt_key][0]
    gt_percent = gt_content[gt_key][1]

    real_ans = value['task1'].replace('task1:', '').replace(' ', '')
    ans_level = real_ans.split(',')[0]
    try:
        ans_percent = real_ans.split(',')[1]
    except:
        print('skip')
        continue
    
    print(f'gt: {gt_level}, {gt_percent}, ans: {ans_level}, {ans_percent}')

    if gt_level == ans_level:
        correct += 1
    abs_diff += abs(float(gt_percent) - float(ans_percent))
    cnt += 1

print(f'correct: {correct}, total: {len(ans_content)}, acc: {correct / len(ans_content)}, mae: {abs_diff / cnt}')
