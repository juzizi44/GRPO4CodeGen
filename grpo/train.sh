export TOKENIZERS_PARALLELISM=false

deepspeed --master_port=28519 --include localhost:2,3 train.py \
    --lora_enable True \
    --freeze_llm False \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_namespan_exclude "['score', 'rm_head', 'embed_tokens']" \
    --bf16 True \
    --torch_dtype "bfloat16" \
    --num_lora_modules -1 \
    --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --meta_data "/data/ytan089/GRPO4CodeGen_v2/dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-train_sorting.jsonl" \
    --execute_code_url "http://localhost:5000/api/execute_code" \
    --output_dir output_model/20250612_weightnet_sorting_1e-5_lora_r64_128_gen8_iter3 \
    --output_dim 1 \
    --use_special_tokens False \
    --reward_token "special" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 72 \
    --num_generations 8 \
    --num_iterations 3 \
    --learning_rate 1e-5 \
    --special_token_lr 1e-5 \
    --report_to tensorboard \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --eval_strategy "epoch" \
    --logging_steps 10 \
    --eval_epochs 5 \
    --save_epochs 5 \
    --max_length 4800 \
    --gradient_checkpointing True \
    --deepspeed ds_config/zero0.json \
    --save_only_model True \
    --save_full_model False \
    --dataloader_num_workers 8 \
    --max_prompt_length 1800 \
    --max_completion_length 3000 \
    --reward_dimensions correctness efficiency comment maintainability \
    --use_weight_net True \
    --fixed_weights "{\"correctness\": 0.5, \"efficiency\": 0.2, \"comment\": 0.2, \"maintainability\": 0.1}"

# 如果是从checkpoint加载模型，一定记得加下面两个参数！！！
# --load_from_pretrained "./output_model/train_20250529_weightnet_sort/" \
# --load_from_pretrained_step 222 \



# ==================================================
# ===> Selected 2376 samples for training.
# ===> Selected 265 samples for testing.
# ===> Using 2 GPUs.
# ===> Total Batch Size: 32
# ===> Training Epochs: 72.0
# ===> Total Steps: 42768.0
# ===> Save Steps: 74
# ===> Eval Steps: 371
# ===> Logging Steps: 10
# ===> Reward Dimensions: ['correctness', 'efficiency', 'comment', 'maintainability']
# ===> Use Weight Net: True
# ===> Fixed Weights: {'correctness': 0.5, 'efficiency': 0.2, 'comment': 0.2, 'maintainability': 0.1}
