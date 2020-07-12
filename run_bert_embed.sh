#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1


for layer_num in 1 2 3 4 5 6 7 8 9 10
do
python run_comp_on_GAT_stack.py \
    --pretrained_transformer_model_type bert \
    --pretrained_transformer_model_name_or_path bert-base-uncased \
    --do_lower_case \
    --input_size=768 \
    --model_name="GAT" \
    --task_name="comp_GAT" \
    --feature_creation="original_bert_model_using_original_text" \
    --dataset_name="uncased_bert_GAT_stack_new_run" \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --data_dir="./data/basicDependencies" \
    --max_seq_length=256 \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=8 \
    --stack_layer_num=$layer_num \
    --embed_dropout=0.3 \
    --att_dropout=0 \
    --learning_rate=0.0005 \
    --weight_decay=1e-4 \
    --num_train_epochs=30 \
    --logging_steps=100 \
    --output_dir="./result" |& tee log.txt
done



