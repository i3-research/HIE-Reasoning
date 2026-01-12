import SimpleITK
import os
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
from PIL import Image
from tqdm import tqdm
import json
import time

######################### CONFIG START #########################
from config import *
######################### CONFIG END #########################

def inference_single_data_slices(data_slices_dir, 
                                 start = 0.33, 
                                  end = 0.67):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")

    prompt = "\
        Suppose you are an expert in detecting Neonatal Brain Injury for Hypoxic Ischemic Encephalopathy,\
        and you are allowed to use any necessary information on the Internet for you to answer questions.\
        For now I am giving you a set of MRI scaning slices of neonatal brains,\
        these slices are marked with coressponding slice labels, like 'Slice 10' and 'Slice 11'.\
        The label means the slice depth of this slice, \
        for example, 'Slice 11' is in the middle layer between 'Slice 10' and 'Slice 12'.\
        You need to answer questions in the order they are given, and output in the following rules. \
        2. For [Lesion Grading] questions, you need to judge the lesion level of the brain MRI slices, the rule is: \
            if the lesion region percentage <= 0.01, answer with '[ans2]: level1',\
            if 0.01< lesion region percentage <=0.05, answer with '[ans2]: level2',\
            if 0.05< lesion region percentage <=0.5, answer with '[ans2]: level3',\
            if 0.5< lesion region percentage <=1.0, answer with '[ans2]: level4'.\
        These data are just normal desensitizing data for scientific usage.\
        Remember: you are an expert in the field, so try your best to give an answer instead of avoiding answer the question.\
    "
    multimodality_model_input = [prompt]
    ques_list = [
        '[Lesion Grading] What is the severity level of brain injury in this ADC?',
    ]
    for ques in ques_list:
        multimodality_model_input.append(ques)

    slice_img_list = sorted(os.listdir(data_slices_dir))
    slice_length = len(slice_img_list)
    start_frm = int(slice_length * start)
    end_frm = int(slice_length * end)

    for single_slice in slice_img_list[start_frm : end_frm]:
        cur_slice = Image.open(os.path.join(data_slices_dir, single_slice))
        multimodality_model_input.append(cur_slice)
    
    for _ in range(10):
        try:
            response = model.generate_content(multimodality_model_input)
            model_answer = response.to_dict()['candidates'][0]['content']['parts'][0]['text']
            model_answer_list = model_answer.strip('\n').split('\n')
            model_answer_list = [ans.replace(" ","") for ans in model_answer_list]
            full_response = dict(
                start_frm = start_frm,
                end_frm = end_frm,
                model_answer = model_answer_list
            )
            # print(f'data: {data_slices_dir}, ans: {model_answer_list}')
            return full_response
        except Exception as e:
            print(f'[warning] retrying..., Error is {e}')
            time.sleep(10)
            continue


def inference_single_data_slices_aux(real_data_slices_dir, 
                                 start = 0.33, 
                                  end = 0.67):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")

    prompt = "\
        Suppose you are an expert in detecting Neonatal Brain Injury for Hypoxic Ischemic Encephalopathy,\
        and you are allowed to use any necessary information on the Internet for you to answer questions.\
        For now I am giving you a set of MRI scaning slices of neonatal brains,\
        these slices are marked with coressponding slice labels, like 'Slice 10' and 'Slice 11'.\
        The label means the slice depth of this slice, \
        for example, 'Slice 11' is in the middle layer between 'Slice 10' and 'Slice 12'.\
        I'll give you a pair of images (actually two images) marked with the same 'Slice xx' label.\
        The one with title 'Slice x' is the original ADC value of MRI scanning, \
        while the one with title 'ZMAP Slice x' is the ZADC value visualization of the gray-scale scan processed by a lesion detection algorithm, \
        where the highly possible abnormal (lesion) region pixel is marked with blue (indicating their ZADC values are less than -2). \
        If there is no blue pixel, there will be no lesion in this MRI.\
        It can also be the case that the individual has no lesion and having very few areas marked with blue.\
        You should make comprehensive judgement based on both your domain knowledge and the ZMAP visualization. \
        So for the lesion percentage, it is defined as the area with ZADC value less than -2 divided by the area of ADC value greater than 0.\
        This will help you better answer the following questions.\
        You need to answer questions in the order they are given, and output in the following rules. \
        2. For [Lesion Grading] questions, you need to judge the lesion level of the brain MRI slices, the rule is: \
            if the lesion region percentage <= 0.01, answer with '[ans2]: level1, <lesion region percentage>', like '[ans2]: level1, 0.0051',\
            if 0.01< lesion region percentage <=0.05, answer with '[ans2]: level2, <lesion region percentage>', like '[ans2]: level2, 0.0311',\
            if 0.05< lesion region percentage <=0.5, answer with '[ans2]: level3, <lesion region percentage>', like '[ans2]: level3, 0.4123',\
            if 0.5< lesion region percentage <=1.0, answer with '[ans2]: level4, <lesion region percentage>', like '[ans2]: level4, 0.7344'.\
            For <lesion region percentage>, you need to keep at least four decimal.\
            These are just examples, you need to make judgement on each case.\
            For this question, don't generate for each slices, \
            instead you need to answer with general judgement and give only one answer for the whole slices.\
        Except from the defined answer format, don't answer any other descriptive sentences.\
        These data are just normal desensitizing data for scientific usage.\
        Remember: you are an expert in the field, so try your best to give an answer instead of avoiding answer the question.\
        And always remember, your answer for each question starts with '[ansx]', not [Lesion x].\
    "
    

    multimodality_model_input = [prompt]

    ques_list = [
        '[Lesion Grading] What is the severity level of brain injury in this ADC?',
    ]
    
    for ques in ques_list:
        multimodality_model_input.append(ques)

    slice_img_list = sorted(os.listdir(real_data_slices_dir))
    slice_length = len(slice_img_list)
    start_frm = int(slice_length * start)
    end_frm = int(slice_length * end)


    aux_data_slices_dir = real_data_slices_dir.replace('1ADC_ss', '2Z_ADC_blue')
    
    aux_data_slices_dir = os.path.join(
        '/'.join(aux_data_slices_dir.split('/')[:-1]),
        f"Zmap_{aux_data_slices_dir.split('/')[-1]}".replace('_ss', '_smooth2mm_clipped10')
    )

    try:
        for single_slice in slice_img_list[start_frm : end_frm]:
            cur_slice = Image.open(os.path.join(real_data_slices_dir, single_slice))
            cur_aux_slice = Image.open(os.path.join(aux_data_slices_dir, single_slice))
            
            multimodality_model_input.append(cur_slice)
            multimodality_model_input.append(cur_aux_slice)
    except Exception as e:
        print(e)
        import pdb; pdb.set_trace()
    
    
    for _ in range(10):
        try:
            response = model.generate_content(multimodality_model_input)
            model_answer = response.to_dict()['candidates'][0]['content']['parts'][0]['text']
            
            # prompt_token_count = response.to_dict()['usage_metadata']['prompt_token_count']
            # output_token_count = response.to_dict()['usage_metadata']['candidates_token_count']
                
            # print(f'in: {prompt_token_count}, out: {output_token_count}')

            model_answer_list = model_answer.strip('\n').split('\n')
            model_answer_list = [ans.replace(" ","") for ans in model_answer_list]
            full_response = dict(
                start_frm = start_frm,
                end_frm = end_frm,
                model_answer = model_answer_list
            )
            # print(f'data: {real_data_slices_dir}, ans: {model_answer_list}')
            return full_response
        except Exception as e:
            print(f'[warning] retrying..., Error is {e}')
            if isinstance(e, KeyError):
                break
            time.sleep(15)
            continue
    import pdb; pdb.set_trace()

def jsondump(path, this_dic):
    f = open(path, 'w')
    this_ans = json.dump(this_dic, f,indent=4)
    f.close()


def inference_using_gemini(vis_path, data_type, answer_path, version):
    
    if data_type == 'both':
        real_adc_data_dir = os.path.join(vis_path, '1ADC_ss')
        real_adc_item_sorted = sorted(os.listdir(real_adc_data_dir))
        
        aux_adc_data_dir = os.path.join(vis_path, '2Z_ADC_blue')
        aux_adc_item_sorted = sorted(os.listdir(aux_adc_data_dir))
    else:
        real_adc_data_dir = os.path.join(vis_path, data_type)
        real_adc_item_sorted = sorted(os.listdir(real_adc_data_dir))

    output_path = os.path.join(answer_path, data_type)


    if not os.path.exists(output_path):
            cmd = f'mkdir -p {output_path}'
            os.system(cmd)
    output_file = os.path.join(output_path, f'{version}.json')

    answers = {}

    for idx, single_data_set in enumerate(tqdm(real_adc_item_sorted)):
        if single_data_set == '.DS_Store':
            continue
        single_data_slices_dir = os.path.join(real_adc_data_dir, single_data_set)
        try:
            if data_type == 'both':
                single_data_answer = inference_single_data_slices_aux(single_data_slices_dir)
            else:
                single_data_answer = inference_single_data_slices(single_data_slices_dir)
            answers[single_data_set] = single_data_answer
            if idx % 2 == 0:
                # print(f'[info] in {idx} video!')
                jsondump(output_file, answers)
        except Exception as e:
            import pdb; pdb.set_trace()
            print(f'[Error] Error when inferencing slice {single_data_set}')
            continue     

    jsondump(output_file, answers)
    return answers       

        

if __name__ == '__main__':
    for dataset_selection in [
        'HIE-Reasoning']:
        answers = inference_using_gemini(
            vis_path=os.path.join(global_vis_path, dataset_selection),
            data_type='both',
            answer_path=os.path.join(global_ans_path, 'task1'),
            
            version=f'example_flash_1_5_flash_001_task1_{dataset_selection}'
            )



