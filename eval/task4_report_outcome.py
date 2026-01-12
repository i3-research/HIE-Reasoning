import json
import os
import numpy as np

######################### CONFIG START #########################

roi_gt_path = '../data/visualization/atlases/roi_gt_0.225_with_injury_score_new_more.json'
mode = '62'

task4_ans_dir = '../data/answers/task4/both/'
cur_ans_pattern = f'example_flash_1_5_flash_001_task4_{mode}_math_v_outcome_'
mgh_train_label = '../data/dataset/outcome/train.npy'
mgh_test_label = '../data/dataset/outcome/test.npy'

task4_ans_list = os.listdir(task4_ans_dir)
cur_ans_task4_ans_list = [file for file in task4_ans_list if cur_ans_pattern in file and 'report' not in file]
######################### CONFIG END #########################


mgh_label = np.load(mgh_train_label)
mgh_label_test = np.load(mgh_test_label)

labels = np.concatenate((mgh_label, mgh_label_test), axis=0).tolist()

cur_ans = {}
for ans_file in cur_ans_task4_ans_list:
    cur_split_ans = None
    with open(os.path.join(task4_ans_dir, ans_file), 'r') as f:
        cur_split_ans = json.load(f)
        cur_ans.update(cur_split_ans)

correct = 0
total = 0

correct_0 = 0
correct_1 = 0


gt_0 = 0
gt_1 = 0
ans_0 = 0
ans_1 = 0
fail = 0
ans_correct = []


for item in labels:
    item_name = item[0]
    item_gt = item[1]
    
    ans_name = f'{item_name}-VISIT_01-ADC_ss'
    if 'Zmap' in list(cur_ans.keys())[0]:
        ans_name = f'Zmap_{item_name}-VISIT_01-ADC_smooth2mm_clipped10'
    try:
        item_ans = cur_ans[ans_name]['model_answer'][0].split(':')[1]
        # item_ans = cur_ans[ans_name]['task4'].replace('[ans]:', '').replace(' ', '')
    except KeyError as e:
        fail += 1
        continue
        print(e)
        import pdb; pdb.set_trace()

    if item_ans == item_gt:
        correct += 1
        if item_ans == '0':
            correct_0 += 1
        else:
            correct_1 += 1
        ans_correct.append([item_name, item_gt])
    if item_gt == '0':
        gt_0 += 1
    if item_gt == '1':
        gt_1 += 1
    if item_ans == '0':
        ans_0 += 1
    if item_ans == '1':
        ans_1 += 1
    total += 1


print(f'correct: {correct}, total: {total + fail}, acc:{correct / (total + fail)}')
print(f'gt_0: {gt_0}, gt_1: {gt_1}, ans_0: {ans_0}, ans_1: {ans_1}, fali: {fail}')
print(f'correct_0: {correct_0 / gt_0}, correct_1: {correct_1 / gt_1}, inter-class acc: {(correct_0 / gt_0 + correct_1 / gt_1) / 2}')
