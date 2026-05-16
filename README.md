# Visual and Domain Knowledge for Professional-level Graph-of-Thought Medical Reasoning (ICML 2025 Spotlight)

## HIE-Reasoning Benchmark Download

## Zenodo Link

Please download the dataset from 

For downloading the dataset, please first fill out the form:
HIEReasoning_Dataset_Agreement.pdf

Then, send the completed form to rina.bao@childrens.harvard.edu or yangming.ou@childrens.harvard.edu.

The file requires a password to unzip the dataset. Please complete the acknowledgment form and send it to us. We will then reply with the password.



## Usage

### Data Layout

Arrange raw data and auxiliary files under `data/dataset/`. The evaluation scripts also expect ground-truth files under `data/visualization/` and experimental outputs under `data/answers/`.

```text
data/
├── dataset/
│   ├── HIEReasoning_Train/
│   │   ├── 1ADC_ss/
│   │   ├── 2Z_ADC/
│   │   
│   ├── HIEReasoning_Val/
│   │   ├── 1ADC_ss/
│   │   ├── 2Z_ADC/
│   │   
│   ├── HIEReasoning_Test/
│   │   ├── 1ADC_ss/
│   │   ├── 2Z_ADC/
│   │   
│   ├── ROI/
│   ├── ADCriolabel/
│   │   └── 62ROIs_for_Children.txt
│   ├── atlases/
│   │   ├── normal_atlases/
│   │   └── lesion_atlases/
│   ├── mgh_train.npy
│   └── mgh_test.npy
├── visualization/
├── answers/
└── evaluation_results/
```

All saved experimental results used by the evaluation scripts are in `data/answers/`. 

```text
data/answers/
├── task1_lesion_grading/
├── task2_lesion_anatomy/
├── task2.5_rare_location/
├── task3_MRI_injury_score/
├── task4_2year_outcome/
└── output_by_case/
```

Derived evaluation JSON files are written to `data/evaluation_results/`.

### Preprocess Data

Run the preprocessing utilities in order if you need to rebuild the processed data and ground-truth files:

```bash
python data_utils/step1_preprocess_data.py
python data_utils/step2_visualization_adc_roi.py
python data_utils/step3_prepare_lesion_gt.py
python data_utils/step4_vis_atlas_n_generate_primary_area.py
python data_utils/step5_generate_roi_gt.py
python data_utils/step6_generate_gt_grading.py
```

## CGoT Pipeline

- CGoT pipeline: follow `tasks/` task1-task5, inference each case using CGoT pipeline.

The repository includes saved experimental outputs in `data/answers/`, so evaluation can be run directly. To regenerate answers, run the task scripts in `src/tasks/`:

```bash
python src/tasks/task1_inference_lesion_grading.py
python src/tasks/task2_inference_anatomy.py
python src/tasks/task3_mri_injury_score.py
python src/tasks/task4_2year_outcome.py
python src/tasks/task5_generate_caption.py
```

Task outputs should be written under the corresponding `data/answers/` subdirectory.

## Evaluation

- Task evaluation: follow `eval/` task1-task4, evaluate each task immediately.

Install the ROUGE dependency used by task 5:

```bash
python -m pip install rouge-score
```

Run evaluation scripts from `src/eval/`:

```bash
cd src/eval

python task1_report_lesion_grading.py
python task2_report_62_roi_anatomy.py
python task2-5_report_rare_location.py
python task3_report_score_acc.py
python task4_report_outcome.py
python task5_generate_caption.py
```

The scripts read existing experiment outputs from `data/answers/`. Scripts that create derived JSON reports write them to `data/evaluation_results/`:

```text
data/evaluation_results/task2_report_flash_ver_task2_anatomy_v2_62_no_7voxel_limit.json
data/evaluation_results/task2_5_rare_location_report_flash_ver_task2_anatomy_v2_62.json
data/evaluation_results/caption_report_flash_ver_task5_generate_caption_v_62.json
```

Recommended order for full evaluation:

```text
task1 -> task2 -> task2.5 -> task3 -> task4 -> task5
```

Task 5 uses outputs from task 1, task 2, task 2.5, and task 3 to generate captions and report ROUGE scores.





