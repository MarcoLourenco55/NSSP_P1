## NSSP_P1
# NSSP Project — Data Structure Overview

This project analyzes fMRI data from the Human Connectome Project as part of the **NX-421: Neural Signals and Signal Processing** course.  
The dataset is divided into two main parts: **Structural** and **Functional** data, each stored in separate folders under the `Preprocessing` directory.

Before running the pipeline, make sure that the **raw data** are correctly placed inside the `Preprocessing/Structural` and `Preprocessing/Functional` folders as shown below.  

---

## Folder Organization

### Preprocessing/Structural/
This folder contains the **anatomical (T1-weighted)** MRI data used for structural preprocessing.  
It is organized as follows:

Preprocessing/
└── Structural/
    ├── T1w.nii.gz                        # Anatomical T1-weighted image
    ├── Skull_striping/                   # Results of skull stripping (BET)
    └── Segmentation/                     # Results of tissue segmentation (FAST)

---

### Preprocessing/Functional/
This folder contains the **functional MRI (fMRI)** data used for motion correction, smoothing, and GLM/ICA analyses.

Preprocessing/
└── Functional/
    ├── tfMRI_MOTOR_LR.nii               # Raw fMRI data (Left→Right)
    ├── tfMRI_MOTOR_RL.nii               # Raw fMRI data (Right→Left)
    ├── subj_concat_var1_mc_s6mm.nii     # Example of preprocessed functional data
    └── other intermediate outputs...

---

### fMRI/ (Task Events)
This folder contains the **task event timing files** used in the GLM stage.

fMRI/
├── tfMRI_MOTOR_LR/
│   └── events_LR.csv
└── tfMRI_MOTOR_RL/
    └── events_RL.csv
---

## Outputs and Results

All preprocessing results are stored within their respective subfolders:
- Structural preprocessing results → Preprocessing/Structural/Skull_striping and Preprocessing/Structural/Segmentation
- Functional preprocessing results → Preprocessing/Functional

Final visual results and figures are saved under the main results/ GLM_output folder.

---

Make sure to keep the folder structure intact when running the pipeline, as file paths are hardcoded relative to the project root.
