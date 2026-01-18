## Usage Instructions for ../bash_script_card0.sh
This shell script implements the complete knowledge distillation pipeline for **CIFAR100 image classification** task, based on MVQ/LSMVQ distillation strategy. It supports knowledge distillation training for RegNetX and ResNet series backbone networks, the whole process is executed in a stage-by-stage manner. No extra command line parameters are required, modify the two core variables in the script header then run directly.

### Prerequisites
1. Configure the required Python environment and all dependent libraries for model training and feature extraction.
2. Ensure the CIFAR100 dataset is placed in the fixed path: `./data/cifar100/`.
3. The script is fixed to run on single GPU with configuration: `export CUDA_VISIBLE_DEVICES=0`.
4. All trained teacher/student model weights and experimental intermediate files are automatically saved to the preset checkpoint directories in the script.

### Stage-by-Stage Execution
The script is divided into 3 independent sequential stages with fixed execution order: 1 → 2 → 3. Each stage is triggered by the condition: [ $stage -le N ] && [ $stop_stage -ge N ]

### Stage 1: Train Teacher Baseline Models

- Trigger condition: [ $stage -le 1 ] && [ $stop_stage -ge 1 ]
- Train three official baseline teacher models for CIFAR100 classification: RegNetX_400MF, resnet32x4, resnet56.
- Call train_baseline.py for training, all teacher model weights are saved to ./data/[model_name]/.

### Stage 2: Extract Codebook Index from Pre-trained Teacher Models

- Trigger condition: [ $stage -le 2 ] && [ $stop_stage -ge 2 ]
- Extract vector quantization codebook index from pre-trained teacher models by running ./QUAD/extract_codebook_index_for_ic.py.
- Process three groups of teacher-student model pairs separately, extract index for two key embedding layers (layer 3 for feature layer, layer 5 for logits layer): RegNetX_400MF→RegNetX_200MF, resnet32x4→resnet56, resnet56→resnet8.
- Fixed hyperparameters: matching num-codebooks (8/16), embedding-dim based on model feature/logits dimension, num-batch-data (100/500).
- All extracted codebook index files are saved to QUAD/exp directory.

### Stage 3: Train Student Models with Multiple Distillation Strategies

- Trigger condition: [ $stage -le 3 ] && [ $stop_stage -ge 3 ]
- Core stage, train lightweight student models with multiple distillation training strategies, fully cover the three teacher-student model pairs.
- All training tasks use unified configuration: --disabled-shuffle-and-augmentation True.
- Two core training scripts are called: train_baseline_student.py for baseline training without distillation, train_student_lsmvq.py for MVQ/LSMVQ distillation training.
- All trained student model weights are saved to ./data/second/[strategy]/[model_name]/.

### How to Run the Script
Modify the stage and stop_stage values in the header part of bash_script_card0.sh, then execute the script directly.
```bash
./bash_script_card0.sh
```
