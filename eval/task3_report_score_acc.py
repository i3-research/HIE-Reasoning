import json
import os

######################### CONFIG START #########################
roi_gt_path = '../data/visualization/atlases/roi_gt_0.225_with_injury_score_new_more.json'
mode = '62'
task3_ans_dir = '../data/answers/task3/both/'

task3_ans_list = os.listdir(task3_ans_dir)
cur_ans_pattern = f'example_flash_1_5_flash_001_task3_{mode}_clinic_v_lesion_grading_'
cur_ans_task3_ans_list = [file for file in task3_ans_list if cur_ans_pattern in file]
######################### CONFIG END #########################


ori_roi = None
with open(roi_gt_path, 'r') as f:
    ori_roi = json.load(f)

cur_ans = {}
for ans_file in cur_ans_task3_ans_list:
    cur_split_ans = None
    with open(os.path.join(task3_ans_dir, ans_file), 'r') as f:
        cur_split_ans = json.load(f)
        cur_ans.update(cur_split_ans)

correct = 0
total = 0

for ans_key, ans_value in cur_ans.items():
    if 'Zmap' in ans_key:
        cur_gt_key = f"HIE_{ans_key.split('-')[0].split('_')[2]}"
    else:
        cur_gt_key = f"HIE_{ans_key.split('-')[0].split('_')[1]}"
    cur_gt_value = ori_roi[cur_gt_key]['mri_injury_score']

    cur_ans_value = int(ans_value['task3'].replace(' ', '').split('Score')[1].strip('.'))
    print(f'gt: {cur_gt_value}, ans: {cur_ans_value}')
    if cur_ans_value == cur_gt_value:
        correct += 1
    total += 1

print(f'correct: {correct}, total: {total}, acc: {correct / total}')

