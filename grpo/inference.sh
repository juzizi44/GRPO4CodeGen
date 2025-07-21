# 设置模型路径
MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
# 输入和输出路径
INPUT_FILE="../dataset/LeetCodeDataset_postprocessed/LeetCodeDataset-v0.3.1-test.jsonl"
OUTPUT_FILE="../dataset/grpo_inference_data/20250702_all_data.jsonl"

CUDA_VISIBLE_DEVICES=7 python inference.py \
  --model_name_or_path "$MODEL_PATH" \
  --checkpoint_path "output_model/20250624_all_data" \
  --checkpoint_step 8295 \
  --input_file "$INPUT_FILE" \
  --output_file "$OUTPUT_FILE" \
  --max_length 3000 \
  --torch_dtype "bfloat16"
