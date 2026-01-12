import json
import os

######################### CONFIG START #########################

primary_location_path = '../data/visualization/atlases/primary_area_0.225.json'
roi_gt_path = '../data/visualization/atlases/roi_gt_0.225_with_injury_score_new_more.json'
# roi_gt_path = '../data/visualization/atlases/roi_gt_0.225_with_injury_score_new_more_7voxel.json'

task2_ans_dir = '../data/answers/task2/both/'
task2_ans_list = os.listdir(task2_ans_dir)
cur_ans_pattern = 'example_flash_1_5_flash_001_task2_'

cur_ans_task2_ans_list = [file for file in task2_ans_list if cur_ans_pattern in file and 'report' not in file]
report_path = os.path.join(task2_ans_dir, f'rare_location_report_{cur_ans_pattern[:-1]}.json')
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

primary_location = {}
with open(primary_location_path, 'r') as f:
    primary_location = json.load(f)

primary_location_list = list(primary_location.keys())
primary_location_list = [int(item) for item in primary_location_list]


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

    if union == 0:
        return 0, intersection_list, union_list

    return intersection / union, intersection_list, union_list

global_inter = 0
global_union = 0
global_set_ans = 0
global_set_gt = 0

yes_or_no = 0

total = 0

report = {}


for ans_key, ans_val in cur_ans.items():
    # cur_gt_key = f"HIE_{ans_key.split('-')[0].split('_')[1]}"
    if 'Zmap' in ans_key:
        cur_gt_key = f"HIE_{ans_key.split('-')[0].split('_')[2]}"
    else:
        cur_gt_key = f"HIE_{ans_key.split('-')[0].split('_')[1]}"
    cur_gt_value = ori_roi[cur_gt_key]['uncommon_region']
    cur_gt_value = [int(item.split(':')[0]) for item in cur_gt_value]

    if 'model_answer' not in ans_val.keys():
        # cur_ans_value_full = ans_val['ans']
        
        cur_ans_value_full = ans_val['task2'].replace('[ans]', '').replace(' ', '').split(',')
        try:
            cur_ans_value_full = [int(cur_item.split('_')[0]) for cur_item in cur_ans_value_full]
        except:
            continue
        
    else:
        cur_ans_value_full = ans_val['model_answer'][0].split(':')[1].split(',')
        try:
            cur_ans_value_full = [int(cur_item.split('_')[0]) for cur_item in cur_ans_value_full]
        except:
            continue
    
    

    cur_ans_value = sorted(list(set(cur_ans_value_full) - set(primary_location_list)))
    
    cur_case_f1 = f1_score(cur_ans_value, cur_gt_value)
    cur_case_jac, cur_case_inter, cur_case_union = jaccard_similarity(cur_ans_value, cur_gt_value)

    global_inter += len(cur_case_inter)
    global_union += len(cur_case_union)
    global_set_ans += len(cur_ans_value)
    global_set_gt += len(cur_gt_value)

    if len(cur_gt_value) == 0 and len(cur_ans_value) == 0 \
        or len(cur_gt_value) > 0 and len(cur_ans_value) > 0:
        yes_or_no += 1
    
    total += 1

    # import pdb; pdb.set_trace()

    report[cur_gt_key] = dict(
        gt = sorted(cur_gt_value),
        ans = sorted(cur_ans_value),
        f1_score = cur_case_f1,
        jaccard_similarity = cur_case_jac
    )

jsondump(report_path, report)

yes_or_no_acc = yes_or_no / total
global_precision = global_inter / global_set_ans
global_recall = global_inter / global_set_gt
global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall)
global_jac = global_inter / global_union

print(f'lesion in rare location yes or no acc: {yes_or_no_acc}')
print(f'global precision: {global_precision}')
print(f'global recall: {global_recall}')
print(f'global f1: {global_f1}')

# jac_sim = f1 / (2 - f1)
print(f'global jac sim: {global_jac}')
