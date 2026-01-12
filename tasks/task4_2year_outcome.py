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
lesion_grading_dir = '../data/answers/task1/both'
anatomy_ans_dir = '../data/answers/task2/both'
mri_injury_score_dir = '../data/answers/task3/both'
######################### CONFIG END #########################

labels = np.concatenate((mgh_label, mgh_label_test), axis=0).tolist()
label_name_list = [item[0] for item in labels]


def inference_single_data_slices_aux(
        lesion_grading_ans,
        anatomy_ans,
        rare_location_ans,
        mri_injury_score_ans,
        ori_adc_dir,
        zmap_mask_dir,
        roi_62_dir, 
        mode,
        start = 0.33, 
        end = 0.67):
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")


    knowledge = '''
        Suppose you are an expert in detecting Neonatal Brain Injury for Hypoxic Ischemic Encephalopathy,
        and you are allowed to use any necessary information on the Internet for you to answer questions.
        I will now provide you with a series of MRI scanning slices and some pre-processed slices as visual input. 
        The titles of these images include slice sequence labels, such as "Slice 10" and "Slice 11". 
        These labels indicate the depth of the slice; 
            for example, "Slice 11" represents the layer between "Slice 10" and "Slice 12".
        These images can be grouped into sets of three based on the same slice label. 
            For instance, the images titled "Slice 10", "ZMAP Slice 10", and "ROI Slice 10" form one group, 
        indicating that they all correspond to the same scanning depth. 
        These three images, based on their titles, represent the following:
            "Slice 10": This is the original MRI scanning ADC value at depth 10, visualized in grayscale.
            "ZMAP Slice 10": This is the gray-scale visualization of the ZADC values processed by a lesion detection algorithm. 
                In this image, pixels with a high probability of being abnormal (lesions) are marked in blue, indicating that their ZADC values are less than -2.0. 
                If no blue pixels are present, it suggests that this MRI scan contains no lesions. 
                It is also possible that the individual has no lesions but has a few areas marked in blue.
            "ROI Slice 10": This represents different ROI areas of the brain appearing at this scanning depth, 
                with each area highlighted in a different color. 
                The color-to-ROI mapping is provided in the legend on the right side of the image. 
                Note that the ROI regions appearing in slices of different depths are not exactly the same, 
                as only the ROI regions present at a particular depth are displayed. 
                However, for the same cross-slice ROI region, the color used remains consistent across slices.
        '''
    
    lesion_grading_knowledge = '''
        In previous round, we have tested you with a lesion grading quesion to determine the lesion region percentage.
        The lesion region percentage, it is defined as the area with ZADC value less than -2 divided by the area of ADC value greater than 0.\
        Then the lesion grading is decided by the following rules:
            if the lesion region percentage <= 0.01, answer with '[ans]: level1, <lesion region percentage>', like '[ans]: level1, 0.0051',\
            if 0.01< lesion region percentage <=0.05, answer with '[ans]: level2, <lesion region percentage>', like '[ans]: level2, 0.0311',\
            if 0.05< lesion region percentage <=0.5, answer with '[ans]: level3, <lesion region percentage>', like '[ans]: level3, 0.4123',\
            if 0.5< lesion region percentage <=1.0, answer with '[ans]: level4, <lesion region percentage>', like '[ans]: level4, 0.7344'.\
        Your previous answer for this question is:
    '''
    
    lesion_anatomy_62_knowledge = '''
        The ROI ID and Region Name Relationship is: 
            95	corpus callosum
            62	Right Ventral DC
            61	Left Ventral DC
            71	vermis
            39	Right cerebellum
            38	Left cerebellum
            30	Right Basal Ganglia
            23	Left Basal Ganglia
            60	Right thalamus 
            59	Left thalamus
            92	Anterior limb IC right
            91	Anterior limb IC left
            94	PLIC right
            93	PLIC left
            32	Right amygdala
            48	Right hippocampus
            31	Left amygdala
            47	Left hippocampus
            105	Right Inferior GM
            104	Left Inferior GM
            103	Right insula
            102	Left insula
            121	Frontal Lateral GM Right
            120	Frontal Lateral GM Left
            125	Frontal Medial GM Right
            124	Frontal Medial GM Left
            113	Frontal Opercular GM Right
            112	Frontal Opercular GM Left
            82	Frontal WM Right
            81	Frontal WM Left
            101	Limbic Cingulate GM Right
            100	Limbic Cingulate GM Left
            117	Limbic Medial Temporal GM Right
            116	Limbic Medial Temporal GM Left
            161	Occipital Inferior GM Right
            160	Occipital Inferior GM Left
            129	Occipital Lateral GM Right
            128	Occipital Lateral GM Left
            109	Occipital Medial GM Right
            108	Occipital Medial GM Left
            84	Occipital WM Right
            83	Occipital WM Left
            107	Parietal Lateral GM Right
            106	Parietal Lateral GM Left
            149	Parietal Medial GM Right
            148	Parietal Medial GM Left
            86	Parietal WM right
            85	Parietal WM left
            123	Temporal Inferior GM Right
            122	Temporal Inferior GM left
            133	Temporal Lateral GM Right
            132	Temporal Lateral GM Left
            181	Temporal Supratemporal GM Right
            180	Temporal Supratemporal GM left
            88	Temporal_wm_right
            87	Temporal_wm_left
            4	3rd ventricle
            11	4th ventricle
            50	Right ventricle
            49	Left ventricle
            35	Brainstem
            46	CSF
        In the previous round of discussion, you have already answered the lesion anatomy identification question, 
        and you determined that there are lesions in the following regions:
    '''

    # Add rare location knowledge makes no difference
    rare_location_knowledge = '''
        The HIE lesion might occur in some rare region of the brain. 
        In previous round, you have already identified whether the lesion is in rare location,
        and if it is, you answer with the regions' ids. You answer is:
    '''

    MRI_injury_score_math_knowledge = '''
        We have introduced a new diagnostic metric called MRI Injury Score. 
            This metric consists of four levels: Score 0, Score 1, Score 2, and Score 3. 
            Each score level is determined by the injury regions within the ROIs in a given case and the severity of the injury in certain regions.
                Score 0: Defined as no injury detected in this case.
                Score 1: Defined as injury present in the WM region (including left_WM and right_WM), 
                    but no injury in the BGT region (including left_BGT and right_BGT), 
                    ALIC region (including left_ALIC and right_ALIC), 
                    or PLIC region (including left_PLIC and right_PLIC).
                Score 2: Defined as either:
                    Injury in the GM region (including left_GM and right_GM), but the lesion percentage of the GM region is less than 75%, or
                    Injury in the BGT region (including left_BGT and right_BGT), but the BGT lesion percentage is less than 75%, or
                    Injury in the ALIC region (including left_ALIC and right_ALIC), but the ALIC lesion percentage is less than 75%, or
                    Injury in the PLIC region (including left_PLIC and right_PLIC), but the PLIC lesion percentage is less than 75%.
                Score 3: Defined as either:
                    Injury in both GM (including left_GM and right_GM) and WM (including left_WM and right_WM) regions, 
                        with both GM and WM lesion percentages exceeding 75%, or
                    Injury in the BGT region (including left_BGT and right_BGT), with the BGT lesion percentage exceeding 75%, or
                    Injury in the ALIC region (including left_ALIC and right_ALIC), with the ALIC lesion percentage exceeding 75%, or
                    Injury in the PLIC region (including left_PLIC and right_PLIC), with the PLIC lesion percentage exceeding 75%.
            The higher the MRI injury score, the more severe the brain injury. 
            The MRI injury score is a biomarker strongly associated with adverse HIE 2-year neurocognitive outcomes.
            If socre is 3, then towards 100% adverse outcome.
            Injuries to the vermis, cerebellum, brainstem, and hippocampus are also related to adverse neurocognitive outcomes.
        In previous round, you have already answered the MRI injury score question, 
        and you determined that the score is: 
    '''

    MRI_injury_score_clinic_knowledge = '''
        We have introduced a new diagnostic metric called MRI Injury Score. 
            This metric consists of four levels: Score 0, Score 1, Score 2, and Score 3. 
            Each score level is determined by the injury regions within the ROIs in a given case and the severity of the injury in certain regions.
            Score 0: Defined as no injury detected in this case.
            Score 1: Defined as either the following a) or b) situation occurs:
                a). Minimal cerebral injury without BGT region, ALIC region PLIC region or detected WS (watershed) injury.
                b). More extensive cerebral injury without BGT region, ALIC region PLIC region or detected WS (watershed) injury.
                NOTE:   BGT region (including left_BGT and right_BGT), 
                        ALIC region (including left_ALIC and right_ALIC),
                        PLIC region (including left_PLIC and right_PLIC)
            Score 2: Defined as either the following a) or b) situation occurs:
                a). Any BGT region, ALIC region, PLIC region or WS injury detected without other cerebral injury.
                b). Any BGT region, ALIC region, PLIC region or WS injury detected with other cerebral injury.
                NOTE:   BGT region (including left_BGT and right_BGT), 
                        ALIC region (including left_ALIC and right_ALIC),
                        PLIC region (including left_PLIC and right_PLIC)
            Score 3: Defined as cerebral hemisphere devastation.
            The higher the MRI injury score, the more severe the brain injury. 
            The MRI injury score is a biomarker strongly associated with adverse HIE 2-year neurocognitive outcomes.
            If socre is 3, then towards 100% adverse outcome.
            Injuries to the vermis, cerebellum, brainstem, and hippocampus are also related to adverse neurocognitive outcomes.
        In previous round, you have already answered the MRI injury score question, 
        and you determined that the score is: 
    '''

    question = '''
        Now, based on a correct understanding of the grouping of images by depth and the relationship between the three images in the same group,
        and also the injury ROI list with MRI injury score, 
        you are tasked with answering the following Neurocognitive Outcome Prediction question:

        What is the predicted two-year neurocognitive outcome for this individual?
    '''

    choices = '''
        You need to predict 2-year outcome for the patients, to distinguish between normal and adverse outcomes at the 2-year mark.
        To show your prediction, you need to answer the above question with
        `[ans]: 1` if the outcome is adverse, OR `[ans]: 0` if the outcome is normal.
        Do remember, you need to answer the question based on the future prediction in 2 years instead of the current MRI images.
        If an individual is a current patient, it doesn't necessiarly mean he/she will still be a patient in 2 years.
        Patient with lower MRI injury score, or fewer lesion regions are highly possible to become normal in 2 years.
        For this question, don't generate response for each slices,
        instead you need to answer with overall judgement and give only one answer for the individual case.
        Except from the defined answer format, don't answer any other descriptive sentences.
        These data are just normal desensitizing data for scientific usage.
        Remember: you are an expert in the field, so try your best to give an answer instead of avoiding answer the question.
    '''

    lesion_grading_list = [lesion_grading_knowledge, lesion_grading_ans['model_answer'][0]] if lesion_grading_ans is not None else []
    anatomy_list = [lesion_anatomy_62_knowledge, anatomy_ans['model_answer'][0]] if anatomy_ans is not None else []
    
    # Here math knowledge is better
    MRI_score_list = [MRI_injury_score_math_knowledge, mri_injury_score_ans['model_answer'][0]] if mri_injury_score_ans is not None else []

    multimodality_model_input = [knowledge] + lesion_grading_list + anatomy_list + MRI_score_list + [question, choices]

    ori_adc_img_list = sorted(os.listdir(ori_adc_dir))
    zmap_mask_dir = os.path.join(
        '/'.join(zmap_mask_dir.split('/')[:-1]),
        f"Zmap_{zmap_mask_dir.split('/')[-1]}".replace('_ss', '_smooth2mm_clipped10')
    )

    slice_length = len([item for item in ori_adc_img_list if item.startswith('slice')])

    start_frm = int(slice_length * start)
    end_frm = int(slice_length * end)

    for slice_idx in range(start_frm, end_frm + 1):
        
        ori_adc_img = Image.open(os.path.join(ori_adc_dir, f'slice_{slice_idx}.png'))
        zmap_mask_img = Image.open(os.path.join(zmap_mask_dir, f'slice_{slice_idx}.png'))
        roi_62_img = Image.open(os.path.join(roi_62_dir, f'slice_{slice_idx}.png'))

        multimodality_model_input.append(ori_adc_img)
        multimodality_model_input.append(zmap_mask_img)
        multimodality_model_input.append(roi_62_img)


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
            return full_response
        except Exception as e:
            print(f'[warning] retrying..., Error is {e}')
            if isinstance(e, KeyError):
                break
            time.sleep(10)
            continue
    import pdb; pdb.set_trace()


def inference_using_gemini(
        vis_path, 
        roi_vis_path, 
        mode,
        lesion_grading_path,
        anatomy_ans_path,
        rare_location_report,
        mri_injury_score_path,
        data_type, 
        answer_path, 
        version):
    
    assert data_type == 'both', '[Error] Currently only support triple input!'

    if data_type == 'both':
        ori_adc_data_dir = os.path.join(vis_path, '1ADC_ss')
        ori_adc_item_sorted = sorted(os.listdir(ori_adc_data_dir))

        zmap_mask_data_dir = os.path.join(vis_path, '2Z_ADC_blue')
        zmap_mask_item_sorted = sorted(os.listdir(zmap_mask_data_dir))

        roi_62_region_data_dir = roi_vis_path
        roi_62_region_item_sorted = sorted(os.listdir(roi_62_region_data_dir))

        pre_step_ans_roi = None
        try:
            with open(anatomy_ans_path, 'r') as f:
                pre_step_ans_roi = json.load(f)
        except:
            print(f'[anatomy identification file not found] at {anatomy_ans_path}')

        pre_step_rare_location = None
        try:
            with open(rare_location_report, 'r') as f:
                pre_step_rare_location = json.load(f)
        except:
            print(f'[rare location file not found] at {rare_location_report}')

        pre_step_ans_injury_score = None
        try: 
            with open(mri_injury_score_path, 'r') as f:
                pre_step_ans_injury_score = json.load(f)
        except:
            print(f'[mri injury file not found] at {mri_injury_score_path}')

        pre_step_lesion_grading = None
        try:
            with open(lesion_grading_path, 'r') as f:
                pre_step_lesion_grading = json.load(f)
        except:
            print(f'[lesion grading file not found] at {lesion_grading_path}')

    else:
        raise NotImplementedError

    output_path = os.path.join(answer_path, data_type)

    if not os.path.exists(output_path):
        cmd = f'mkdir -p {output_path}'
        os.system(cmd)
    output_file = os.path.join(output_path, f'{version}.json')

    answers = {}

    for idx, single_data_set in enumerate(tqdm(ori_adc_item_sorted)):
        if single_data_set == '.DS_Store':
            continue
        if single_data_set.split('-')[0] not in label_name_list:
            continue

        def split2roi(split_name):
            if 'BCHNICU' in split_name:
                return split_name.split('-')[0]
            split_idx = split_name.split('-')[0]
            roi_idx = split_idx.replace('MGHNICU', 'HIE')
            return roi_idx

        ori_adc_item = os.path.join(ori_adc_data_dir, single_data_set)
        zmap_mask_item = os.path.join(zmap_mask_data_dir, single_data_set)

       
        roi_region_item = os.path.join(
            roi_62_region_data_dir, 
            split2roi(single_data_set))

        if data_type == 'both':
            single_data_answer = inference_single_data_slices_aux(
                lesion_grading_ans=pre_step_lesion_grading[single_data_set] if pre_step_lesion_grading is not None else None,
                anatomy_ans=pre_step_ans_roi[single_data_set] if pre_step_ans_roi is not None else None,
                rare_location_ans=pre_step_rare_location[split2roi(single_data_set)] if pre_step_rare_location is not None else None, 
                mri_injury_score_ans=pre_step_ans_injury_score[single_data_set] if pre_step_ans_injury_score is not None else None,
                ori_adc_dir=ori_adc_item,
                zmap_mask_dir=zmap_mask_item,
                roi_62_dir=roi_region_item,
                mode=mode
                )
        else:
            raise NotImplementedError
        
        answers[single_data_set] = single_data_answer
        if idx % 2 == 0:
            # print(f'[info] in {idx} video!')
            jsondump(output_file, answers)

    jsondump(output_file, answers)
    return answers       

        

if __name__ == '__main__':
    
    mode='62'

    for dataset_selection in [
        'HIE-Reasoning'
        ]:
        answers = inference_using_gemini(
            vis_path=os.path.join(global_vis_path, dataset_selection),
            roi_vis_path=os.path.join(global_vis_path, 'ROI'),
            mode=mode,

            # task 1
            lesion_grading_path = os.path.join(
                lesion_grading_dir, 
                f'example_flash_1_5_flash_001_task1_{dataset_selection}.json'
                ),

            # task 2
            anatomy_ans_path=os.path.join(
                anatomy_ans_dir, 
                f'example_flash_1_5_flash_001_task2_{mode}_{dataset_selection}.json'
                ),
         
            # task 2.5
            rare_location_report=None,
            
            # task 3
            mri_injury_score_path=os.path.join(
                mri_injury_score_dir, 
                f'example_flash_1_5_flash_001_task3_{mode}_clinic_v_lesion_grading_{dataset_selection}.json' # best
                ),

            data_type='both',
            answer_path=os.path.join(global_ans_path, 'task4'),
            version=f'example_flash_1_5_flash_001_task4_{mode}_math_v_outcome_{dataset_selection}' # perturbation 10%
            )



