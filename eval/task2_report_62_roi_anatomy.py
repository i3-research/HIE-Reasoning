import json
import os

######################### CONFIG START #########################
roi_gt_path = '../data/visualization/atlases/roi_gt_0.225_with_injury_score_new_more.json'
task2_ans_dir = '../data/answers/task2/both/'
cur_ans_pattern = 'example_flash_1_5_flash_001_task2_'

task2_ans_list = os.listdir(task2_ans_dir)
cur_ans_task2_ans_list = [file for file in task2_ans_list if cur_ans_pattern in file and 'report' not in file]
report_path = os.path.join(task2_ans_dir, f'report_{cur_ans_pattern[:-1]}_no_7voxel_limit.json')

######################### CONFIG END #########################




def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f,indent=4)
    f.close()

ori_roi = None
with open(roi_gt_path, 'r') as f:
    ori_roi = json.load(f)


cur_ans = {}
for ans_file in cur_ans_task2_ans_list:
    cur_split_ans = None
    with open(os.path.join(task2_ans_dir, ans_file), 'r') as f:
        cur_split_ans = json.load(f)
        cur_ans.update(cur_split_ans)


def f1_score(ans, gt):
    # Convert to sets for easier intersection and union calculations
    set_ans, set_gt = set(ans), set(gt)
    
    # Compute intersection size
    intersection = len(set_ans & set_gt)
    
    # Compute precision and recall
    precision = intersection / len(set_ans) if set_ans else 0
    recall = intersection / len(set_gt) if set_gt else 0
    
    # Compute F1 score
    if precision + recall == 0:
        return 0.0  # Avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def jaccard_similarity(ans, gt):
    set1, set2 = set(ans), set(gt)
    intersection_list = list(set1 & set2)
    union_list = list(set1 | set2)
    intersection = len(set1 & set2)  
    union = len(set1 | set2)

    return intersection / union, intersection_list, union_list

global_inter = 0
global_union = 0
global_set_ans = 0
global_set_gt = 0


report = {}

for ans_key, ans_val in cur_ans.items():
    if 'Zmap' in ans_key:
        cur_gt_key = f"HIE_{ans_key.split('-')[0].split('_')[2]}"
    else:
        cur_gt_key = f"HIE_{ans_key.split('-')[0].split('_')[1]}"
    cur_gt_value = ori_roi[cur_gt_key]['primary_region'] + ori_roi[cur_gt_key]['uncommon_region']
    cur_gt_value = [int(item.split(':')[0]) for item in cur_gt_value]
    
    cur_ans_value = ans_val['task2'].replace('[ans]', '').replace(' ', '').split(',')
    try:
        cur_ans_value = [int(cur_item.split('_')[0]) for cur_item in cur_ans_value]
    except:
        continue

    cur_case_f1 = f1_score(cur_ans_value, cur_gt_value)
    cur_case_jac, cur_case_inter, cur_case_union = jaccard_similarity(cur_ans_value, cur_gt_value)

    global_inter += len(cur_case_inter)
    global_union += len(cur_case_union)
    global_set_ans += len(cur_ans_value)
    global_set_gt += len(cur_gt_value)

    report[cur_gt_key] = dict(
        gt = sorted(cur_gt_value),
        ans = sorted(cur_ans_value),
        f1_score = cur_case_f1,
        jaccard_similarity = cur_case_jac
    )

jsondump(report_path, report)

global_precision = global_inter / global_set_ans
global_recall = global_inter / global_set_gt
global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall)
global_jac = global_inter / global_union

print(f'global precision: {global_precision}')
print(f'global recall: {global_recall}')
print(f'global f1: {global_f1}')

# jac_sim = f1 / (2 - f1)
print(f'global jac sim: {global_jac}')

