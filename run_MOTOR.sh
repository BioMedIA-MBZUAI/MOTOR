#!/bin/bash
top_k=10
root_dir="/datasets" # Should have MIMIC-CXR-JPG, CXR-PRO, Medical-Diff-VQA and the generated reports
dataset_name="med-diff-vqa"
model_name="llava-med-v1.5-mistral-7b"
embedding_model="dino"
generated_texts_folder="grounded_reports" # Path to the generated grounded reports by MAIRA-2
use_caption="enabled" # Whether to include the caption in the prompt
ot="enabled" # Whether to enable re-ranking by Optimal Transport
ot_alpha=0.2 # Question-Report relevance weight
ot_beta=0.3 # Text-Text relevance weight
ot_delta=0.5 # Image-Image relevance weight
outputs_dir="OT_outputs_ot_${ot}_${ot_alpha}_${ot_beta}_${ot_delta}" # Path to save results after running the script

# Run the main.py script
python main.py \
  --top_k $top_k \
  --root_dir $root_dir \
  --dataset_name $dataset_name \
  --model_name $model_name \
  --embedding_model $embedding_model \
  --search_modality image \
  --query_modality image \
  --use_caption $use_caption \
  --ot $ot \
  --ot_alpha $ot_alpha \
  --ot_beta $ot_beta \
  --ot_delta $ot_delta \
  --outputs_dir $outputs_dir \
  --subset_size 5000 \
  --generated_texts_folder $generated_texts_folder

# Run the evaluation script
python eval_nli.py --output_file "${outputs_dir}/${dataset_name}/${embedding_model}/${model_name}_search_image_query_image_top_${top_k}"
