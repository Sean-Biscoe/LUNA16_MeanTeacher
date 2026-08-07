# 3D Medical Imaging Nodule Segmentation with Semi-Supervised Learning (LUNA16)

## Dissertation Project Overview

This repository contains the full source code and analysis pipeline for a Year 3 dissertation project. The project developed a sophisticated semi-supervised deep learning framework for the automated segmentation of pulmonary nodules in standard CT scans. We contrasted two distinct network architectures:

1.  **Fully Supervised Baseline:** A 3D U-Net (LUNA16_UNet) trained exclusively on labeled data[cite: 12, 13].
2.  **Semi-Supervised Mean Teacher (MT):** An advanced MT framework featuring a Student-Teacher architecture where the Teacher’s parameters represent an Exponential Moving Average (EMA) of the Student’s weights[cite: 13].

**Key Finding:** The Mean Teacher framework successfully incorporated unlabeled volumes, achieving a peak validation Dice Similarity Coefficient (DSC) of $0.6617$ (Epoch 87), demonstrating the powerful data-efficiency benefits of semi-supervised techniques for medical image segmentation tasks[cite: 14].

---

## Technical Specifications & Environment

*   **Deep Learning Framework:** PyTorch v2.1.2 (with CUDA 12.1 acceleration)[cite: 13]
*   **Operating System Compatibility:** Linux/Ubuntu (as defined in the dissertation workflow)[cite: 13]
*   **Medical Imaging Toolkits:** SimpleITK (v2.3.1) and nibabel for advanced handling of `.mhd` and `.nii.gz` file formats[cite: 12, 13]
*   **Core Metrics (Validation):** Dice Similarity Coefficient (DSC) and combined BCE+Dice loss[cite: 13]

---

## Core Mathematics of the Pipeline

### Model Metric (Dice Coefficient)
The evaluation metric used across all experiments to quantify spatial overlap between the target mask ($Y$) and predicted probability map ($\hat{Y}$)[cite: 13]:

$$\text{DSC}(Y, \hat{Y}) = \frac{2 \sum Y \cdot \hat{Y}}{\sum Y + \sum \hat{Y}}$$

### Multi-Loss Objective Function
The loss function minimized by the student network is a combined Binary Cross-Entropy (BCE) and Dice Loss[cite: 13]:

$$\mathcal{L}_{\text{sup}} = \text{BCE}(y, \hat{y}) + (1 - \text{DSC}(y, \hat{y}))$$

### Consistency Ramp-Up Strategy
In the Mean Teacher setup, the consistency weight $w_{\text{cons}}(t)$ is applied to the unlabeled data loss, ramping up over the first $T_{\text{ramp}} = 20$ epochs via a Gaussian curve[cite: 13]:

$$w_{\text{cons}}(t) = \exp\left( -5 \left( 1 - \frac{t}{T_{\text{ramp}}} \right)^2 \right)$$

---

## Project Structure & Data References

The repository structure matches the final organization defined in the dissertation documentation:

```text
/Year 3 project
├── .gitignore                   # Standard file exclusion rules (cache/logs/raw Zips)
├── README.md                    # This documentation file
├── requirements.txt             # Python dependency manifest
├── baseline_pred_730.nii.gz     # Qualitative visualization: 3D prediction map from Supervised Baseline (Scan 730)
├── mask_730.nii.gz              # Qualitative visualization: Centered ground-truth binary mask (Scan 730)
├── mt_pred_730.nii.gz           # Qualitative visualization: 3D prediction map from Mean Teacher framework (Scan 730)
├── scan_730.nii.gz              # Qualitative visualization: normalized example input CT volume (Scan 730)
│
├── data/                        # Contains essential metadata for dataset reconstruction
│   ├── annotations.csv          # Metadata: nodule coordinates and series UIDs[cite: 12, 13]
│   ├── candidates.csv           # Metadata: nodule classification labels
│   └── sampleSubmission.csv     # Metadata: submission formatting placeholder
│
└── models/                      # Core implementation logic
    └── unet3d.py                # Definition of the LUNA16_UNet architecture and standard Mean Teacher framework (EMA logic)[cite: 12, 13]
```

---

## Installation & Environment Setup

1.  **Clone this repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a dedicated Python 3.10 virtual environment:**
    ```bash
    python3.10 -m venv dissertation_env
    source dissertation_env/bin/activate  # On Linux/macOS
    # .\dissertation_env\Scripts\activate # On Windows
    ```

3.  **Install the specific project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## External Data Acquisition

**The raw CT scan volumes and full segmentation masks used for model training are not hosted in this repository due to size constraints.**

You must acquire the primary dataset and preprocess it using the scripts mentioned in the dissertation text.

1.  **Base Dataset:** Download the original LUNA16 grand challenge dataset from its primary archive[cite: 13].
2.  **Preprocessing:** Use appropriate masking and normalization tools (e.g., SimpleITK) to generate compatible `.npy` or `.nii.gz` training arrays. The required preprocessing steps are defined in the dissertation text (Window HU $[-1200, 600]$, scaled $[0, 1]$)[cite: 15].

---

## Step-by-Step Pipeline Execution

*Note: Paths and absolute file positions referenced below must align with your local system layout of the LUNA16 dataset and labeled metadata.*

### Step 1: Preprocessing and Data Structuring
Generate pre-processed crops for centered masks and background validation.
```bash
# Execute scripts that use the annotations.csv and raw LUNA16 volumes
python export_centered_mask.py  # (Example script name)
python generate_labels.py         # Generate required training indices
```
*   *Output:* Preprocessed Hounsfield windowed CT volumes ($128 \times 128 \times 128$ crops) and binary masks (0/1)[cite: 12, 13, 15].

### Step 2: Training the Fully Supervised Baseline (LUNA16_UNet)
Train the model using *only* labeled data on two GPUs.
```bash
# Configured for standard cross-validation or specific hold-out fold[cite: 13]
python train.py --config supervised_baseline.yaml --gpus 2
```
*   *Validation Reference:* The quantitative validation DSC history (e.g., the standard baseline performance referenced in Results chart[cite: 14]) is generated during this step.

### Step 3: Training the Semi-Supervised Mean Teacher Model
Train using a mixed-batch (labeled+unlabeled data) and consistency ramp-up (epochs 0-20)[cite: 13].
```bash
# Example setup: batch_size=2 labeled + 2 unlabeled samples per GPU[cite: 13]
python train.py --config mean_teacher_semisup.yaml --gpus 2 --resume checkpoints/pretrain.pt
```
*   *Best Model (Ref Dissertation):* The network achieves peak quantitative performance (DSC $0.6617$) around Epoch 87[cite: 14].

### Step 4: Verification and Model Persistence
```bash
# Persist the final teacher weights and output example predictions
python save_predictions.py --model checkpoints/best_teacher_epoch87.pt --scan 730
```

---

## Qualitative Dissertation Summary (Scan 730 Comparison)

The four `.nii.gz` visualization files provided in the repository create a complete comparative case study using Scan ID 730. They demonstrate the advantage of Mean Teacher over standard supervision when handling low-resource or challenging nodule segmentations.

### Visualization Workflow
Load the provided NIfTI volumes into a standard 3D viewer (e.g., ITK-SNAP or 3D Slicer) in the following configuration:

| Layer Type | File Name | Context |
| :--- | :--- | :--- |
| **Main Image** | `scan_730.nii.gz` | Input: normalized Hounsfield window crop[cite: 15] |
| **Ground Truth Mask (Overlay 1)**| `mask_730.nii.gz` | The intended perfect segmentation target (binary 1/0)[cite: 12, 13] |
| **Baseline Prediction (Overlay 2)**| `baseline_pred_730.nii.gz`| 3D prediction map showing final supervised network output[cite: 13, 14] |
| **Mean Teacher Pred (Overlay 3)**| `mt_pred_730.nii.gz`| 3D prediction map showing final output of proposed semi-supervised model (v8)[cite: 13, 14] |

### Synthesis and Interpretation
By overlaying all three result volumes simultaneously, you can directly evaluate performance:

1.  **Mean Teacher vs. Ground Truth:** Observe how `mt_pred_730.nii.gz` adheres closely to the structure of `mask_730.nii.gz`, demonstrating high spatial overlap.
2.  **Mean Teacher vs. Baseline:** By toggling visible layers between `mt_pred_730.nii.gz` and `baseline_pred_730.nii.gz`, you can identify where the supervised baseline failed or generated false positives.
3.  **Core Contribution:** This visual result demonstrates the qualitative advantage provided by the semi-supervised framework, leveraging information from the unlabeled volumes[cite: 13, 14].

---

## Dissertation Project Citation

If using this implementation as part of subsequent research or in a technical review, please cite the original dissertation document:

> **[cite: YOUR_NAME], (2024).** *Investigating Semi-Supervised Learning (Mean Teacher) for 3D Pulmonary Nodule Segmentation.* Year 3 Dissertation Project, Loughborough University.

---
