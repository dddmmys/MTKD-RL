#!/usr/bin/env bash

stage=3
stop_stage=3
export CUDA_VISIBLE_DEVICES=0

# train teacher models
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

# python train_baseline.py --model RegNetY_400MF      --data-folder ./data/cifar100     --checkpoint-dir ./data/RegNetY_400MF/
# python train_baseline.py --model RegNetX_400MF      --data-folder ./data/cifar100     --checkpoint-dir ./data/RegNetX_400MF/
# python train_baseline.py --model resnet32x4      --data-folder ./data/cifar100     --checkpoint-dir ./data/resnet32x4/
# python train_baseline.py --model wrn_28_4      --data-folder ./data/cifar100     --checkpoint-dir ./data/wrn_28_4/

python train_baseline.py --model RegNetX_200MF      --data-folder ./data/cifar100     --checkpoint-dir ./data/RegNetX_200MF/
python train_baseline.py --model MobileNetV2      --data-folder ./data/cifar100     --checkpoint-dir ./data/MobileNetV2/
python train_baseline.py --model ShuffleV2      --data-folder ./data/cifar100     --checkpoint-dir ./data/ShuffleV2/
python train_baseline.py --model resnet56      --data-folder ./data/cifar100     --checkpoint-dir ./data/resnet56/

fi

# this is an example
# python  ./QUAD/example.py  --data ./data/cifar100  --arch RegNetX_200MF  --checkpoint-dir ./data/checkpoint/RegNetX_200MF/  --teacher-name-list RegNetY_400MF   --rank 0 
# teachers dim list  features -- logits  (3 -- 5)
# RegNetY_400MF [64, 384, 8, 8] -- [64, 100]
# RegNetX_400MF [64, 384, 8, 8] -- [64, 100]
# resnet32x4 [64, 256, 8, 8] -- [64, 100]
# wrn_28_4 [64, 256, 8, 8] -- [64, 100]

# students dim list  features -- logits  (3 -- 5)
# RegNetX_200MF [64, 368, 8, 8] -- [64, 100]

# extract codebook index from teacher
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

# python ./QUAD/extract_codebook_index_for_ic.py \
#     --data ./data/cifar100 \
#     --arch resnet56 \
#     --checkpoint-dir ./data/checkpoint/resnet56/ \
#     --teacher-name-list resnet32x4 \
#     --rank 0 \
#     --embedding-layer 5 \
#     --num-batch-data 500 \
#     --kd-exp-dir QUAD/exp \
#     --num-codebooks 8 \
#     --embedding-dim 100

# python ./QUAD/extract_codebook_index_for_ic.py \
#     --data ./data/cifar100 \
#     --arch resnet56 \
#     --checkpoint-dir ./data/checkpoint/resnet56/ \
#     --teacher-name-list resnet32x4 \
#     --rank 0 \
#     --embedding-layer 3 \
#     --num-batch-data 100 \
#     --kd-exp-dir QUAD/exp \
#     --num-codebooks 16 \
#     --embedding-dim 256


python ./QUAD/extract_codebook_index_for_ic.py \
    --data ./data/cifar100 \
    --arch RegNetX_200MF \
    --checkpoint-dir ./data/checkpoint/RegNetX_200MF/ \
    --teacher-name-list RegNetX_400MF \
    --rank 0 \
    --embedding-layer 5 \
    --num-batch-data 500 \
    --kd-exp-dir QUAD/exp \
    --num-codebooks 8 \
    --embedding-dim 100

python ./QUAD/extract_codebook_index_for_ic.py \
    --data ./data/cifar100 \
    --arch RegNetX_200MF \
    --checkpoint-dir ./data/checkpoint/RegNetX_200MF/ \
    --teacher-name-list RegNetX_400MF \
    --rank 0 \
    --embedding-layer 3 \
    --num-batch-data 100 \
    --kd-exp-dir QUAD/exp \
    --num-codebooks 16 \
    --embedding-dim 384

fi

# train student models
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then

# resnet32x4 ------ resnet56 -------------------------------------------------------------------------------------------
# baseline no distillation 62.84
python train_baseline_student.py \
    --model resnet56 \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/baseline-student/240epoch/resnet56/  \
    --disabled-shuffle-and-augmentation True

# baseline kd logits distillation 63.99
python train_student_avg.py \
    --data ./data/cifar100 \
    --arch resnet56 \
    --checkpoint-dir ./data/second/kd-student/240epoch/resnet56/  \
    --teacher-name-list resnet32x4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --kd-feat-enable False \
    --rank 0 

# baseline kd-feat distillation 64.81
python train_student_avg.py \
    --data ./data/cifar100 \
    --arch resnet56 \
    --checkpoint-dir ./data/second/kd-feat-student/240epoch/resnet56/  \
    --teacher-name-list resnet32x4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --kd-feat-enable True \
    --rank 0 

# mvq single layer 64.86
python train_student_lsmvq.py \
    --model resnet56 \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/ls-mvq/240epoch/stsl5/resnet56_8/ \
    --num-codebooks "8" \
    --middle-output-layers "5" \
    --teacher-name-list resnet32x4 \
    --disabled-shuffle-and-augmentation True

# mvq multi layers 66.08
python train_student_lsmvq.py \
    --model resnet56 \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/ls-mvq/240epoch/stml35/resnet56_16_8/ \
    --num-codebooks "16,8" \
    --middle-output-layers "3,5" \
    --teacher-name-list resnet32x4 \
    --layer-avg True \
    --disabled-shuffle-and-augmentation True

# lsmvq single layer 64.47
python train_student_lsmvq.py \
    --model resnet56 \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/ls-mvq/240epoch/mtsl5/resnet56_8/ \
    --num-codebooks "16,8" \
    --middle-output-layers "3,5" \
    --teacher-name-list resnet32x4 RegNetX_400MF \
    --layer-avg True \
    --multi-teacher True \
    --ls-sl True \
    --disabled-shuffle-and-augmentation True

# lsmvq multi layers 66.64
python train_student_lsmvq.py \
    --model resnet56 \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/ls-mvq/240epoch/mtml35/resnet56_16_8/ \
    --num-codebooks "16,8" \
    --middle-output-layers "3,5" \
    --teacher-name-list resnet32x4 RegNetX_400MF \
    --layer-avg True \
    --multi-teacher True \
    --disabled-shuffle-and-augmentation True

# RegNetX_400MF ------ RegNetX_200MF -----------------------------------------------------------------------------
# baseline no distillation 69.17
python train_baseline_student.py \
    --model RegNetX_200MF \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/baseline-student/240epoch/RegNetX_200MF/  \
    --disabled-shuffle-and-augmentation True

# baseline kd logits distillation 74.02
python train_student_avg.py \
    --data ./data/cifar100 \
    --arch RegNetX_200MF \
    --checkpoint-dir ./data/second/kd-student/240epoch/RegNetX_200MF/  \
    --teacher-name-list RegNetX_400MF \
    --dist-backend 'nccl' \
    --world-size 1 \
    --kd-feat-enable False \
    --rank 0 

# baseline kd-feat distillation 74.79
python train_student_avg.py \
    --data ./data/cifar100 \
    --arch RegNetX_200MF \
    --checkpoint-dir ./data/second/kd-feat-student/240epoch/RegNetX_200MF/  \
    --teacher-name-list RegNetX_400MF \
    --dist-backend 'nccl' \
    --world-size 1 \
    --kd-feat-enable True \
    --rank 0 

# mvq single layer 71.67
python train_student_lsmvq.py \
    --model RegNetX_200MF \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/ls-mvq/240epoch/stsl5/RegNetX_200MF_8/ \
    --num-codebooks "8" \
    --middle-output-layers "5" \
    --teacher-name-list RegNetX_400MF \
    --disabled-shuffle-and-augmentation True

# mvq multi layers 74.58
python train_student_lsmvq.py \
    --model RegNetX_200MF \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/ls-mvq/240epoch/stml35/RegNetX_200MF_16_8/ \
    --num-codebooks "16,8" \
    --middle-output-layers "3,5" \
    --teacher-name-list RegNetX_400MF \
    --layer-avg True \
    --disabled-shuffle-and-augmentation True

# lsmvq multi layers 74.89
python train_student_lsmvq.py \
    --model RegNetX_200MF \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/ls-mvq/240epoch/mtml35/RegNetX_200MF_16_8/ \
    --num-codebooks "16,8" \
    --middle-output-layers "3,5" \
    --teacher-name-list RegNetX_400MF resnet32x4 \
    --layer-avg True \
    --multi-teacher True \
    --disabled-shuffle-and-augmentation True

# mvq single layer 72.86
python train_student_lsmvq.py \
    --model RegNetX_200MF \
    --data-folder ./data/cifar100 \
    --checkpoint-dir ./data/second/ls-mvq/240epoch/mtsl5/RegNetX_200MF_8/ \
    --num-codebooks "16,8" \
    --middle-output-layers "3,5" \
    --teacher-name-list RegNetX_400MF resnet32x4 \
    --layer-avg True \
    --multi-teacher True \
    --ls-sl True \
    --disabled-shuffle-and-augmentation True


fi