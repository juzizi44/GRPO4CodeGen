
---

# 🧠 GRPO4CodeGen

**GRPO4CodeGen** is a code generation alignment project based on **GRPO (Gradient Reward Preference Optimization)**.
It fine-tunes models using multi-dimensional rewards such as **correctness**, **efficiency**, **comment quality**, and **maintainability**.

We train on the **LeetCodeDataset** and extend the [ExecEval](https://github.com/ntunlp/ExecEval) framework to better capture execution-level signals.

---

## 📦 Dataset: LeetCodeDataset

We use the [**LeetCodeDataset**](https://huggingface.co/datasets/newfacade/LeetCodeDataset), released in **April 2025**, designed specifically for training and evaluating code generation models.

### Dataset Details

| Set       | Problems | Release Time            | Content Included                        | Notes                                            |
| --------- | -------- | ----------------------- | --------------------------------------- | ------------------------------------------------ |
| 🏋️ Train | \~2,631  | Before **July 1, 2024** | Description, examples, difficulty, tags | Used for supervised fine-tuning                  |
| 🧪 Test   | 238      | After **July 1, 2024**  | Description, examples, difficulty, tags | Evaluation only; no data leakage from the future |

---

## 🎯 Reward Dimensions

We use four types of reward signals:

| Reward Type         | Description                                      | Method                                                                                                                             |
| ------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| ✔️ Correctness      | Pass rate of test cases                          | https://arxiv.org/pdf/2105.09938                                                                    |
| ⚡ Efficiency        | Measures runtime, memory usage, CPU instructions | Following approaches from https://dl.acm.org/doi/10.1145/3715727 and https://arxiv.org/abs/2503.15242, using cirron and tracemalloc for measurement |
| 💬 Comments         | Comment clarity, relevance, and helpfulness      | Combines static rules and LLM-as-a-judge scoring                                                                                   |
| 🛠️ Maintainability | Code complexity and maintainability index        | Computed via [`radon`](https://radon.readthedocs.io/en/latest/api.html#radon.metrics.mi_visit)                                     |

---

## ⚖️ Reward Aggregation

GRPO4CodeGen supports multiple reward strategies:

* **Fixed Weights**
  Manually assign weights, e.g. `0.4` for correctness, `0.3` for efficiency
  → Enable with: `--fixed_weights`

* **Dynamic Weighting**
  Uses a `RewardWeightNet` to learn weights automatically
  → Enable with: `--use_weight_net`

* **Fallback Mode**
  If no reward is active, a constant reward (e.g., `0.5 ± ε`) is used.
  → For testing only; **not recommended for training**

---

## 🔍 Runtime Evaluation: Extended ExecEval

We extend [ExecEval](https://github.com/ntunlp/ExecEval) to collect runtime metrics through a custom Docker setup for Python code execution.

### Additional Logged Metrics

| Metric                     | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| ✅ `exec_outcome`           | Final result (`PASSED`, `WRONG_ANSWER`, `RUNTIME_ERROR`, etc.) |
| ⏱️ `time_consumed`         | Total wall-clock execution time (seconds)                      |
| 🧮 `cpu_instruction_count` | Total CPU instructions used during execution                   |
| 📉 `tracemalloc_current`   | Current memory usage (in bytes)                                |
| 📈 `tracemalloc_peak`      | Peak memory usage (in bytes)                                   |

---



## 🚀 Quick Start

### 1. Create Conda Environment

```bash
conda env create -f environment.yaml
conda activate grpo4codegen
```

> 📌 Make sure `conda` is installed. The environment name is defined in `environment.yaml`.

> 📌 `pip install flash-attn==2.5.8 --no-build-isolation`
---



### 2. Start Code Execution Docker (ExecEval)

During training, Docker is used to run and evaluate code based on test cases for reward calculation.

```bash
# Step into the Docker project directory
cd ExecEval_for_reward
```

Follow the instructions in `ExecEval_for_reward/README.md` to build and start the container. This service is required during training to provide reward feedback.

---

### 3. Prepare the Dataset

Use the following notebooks to download and preprocess the dataset, ensuring the generated code can be executed with test cases via ExecEval::

* `process_data_LeetCodeDataset_train.ipynb`
* `process_data_LeetCodeDataset_test.ipynb`

You may need to adjust:

* File names and save paths

---

### 4. (Optional) Enable LLM-based Comment Scoring

In `trainer.py`, around **line 163**, modify the logic to enable or disable **LLM-as-a-judge** scoring for comments.

If enabling it, don’t forget to set your API key in:

```bash
grpo/reward/api_key.py
```

---

### 5. Start Training

Run the training script:

```bash
bash train.sh
```

Key training parameters:

```bash
--model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
# Model path (can be Hugging Face model or local checkpoint)

--meta_data "/path/to/LeetCodeDataset-v0.3.1-train_sorting.jsonl" \
# Path to preprocessed training data

--execute_code_url "http://localhost:5000/api/execute_code" \
# URL for ExecEval Docker service

--output_dir output_model/your_experiment_name \
# Directory to save checkpoints and logs

--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
# 🔧 Adjust these according to the number of GPUs

--num_train_epochs 72 \
# Number of training epochs (you can change this)

--num_generations 8 \
# Number of completions generated per prompt

--num_iterations 3 \
# How many times to repeat each prompt-task during GRPO

--learning_rate 1e-5 \
--eval_strategy "epoch" \
--logging_steps 10 \
--eval_epochs 5 \
--save_epochs 5 \
--max_length 4800 \
--max_prompt_length 1800 \
--max_completion_length 3000 \

--reward_dimensions correctness efficiency comment maintainability \
# Enabled reward types

--use_weight_net True \
# Use learned reward weights

--fixed_weights "{\"correctness\": 0.5, \"efficiency\": 0.2, \"comment\": 0.2, \"maintainability\": 0.1}" \
# Used only if `--use_weight_net` is set to False
```

---
当然可以，以下是将你提供的说明翻译并改写为自然流畅的英文版，适合放在 GitHub README 中：

---

### 🔄 Resuming from Checkpoint

If you want to resume training from a previous checkpoint, be sure to include:

```bash
--load_from_pretrained "./output_model/train_20250529_weightnet_sort/" \
--load_from_pretrained_step 222
```

---

### 📊 Visualize with TensorBoard

To monitor training logs and metrics using TensorBoard:

1. Navigate to the model output directory:

   ```bash
   cd grpo/output_model
   ```

2. Start TensorBoard:

   ```bash
   tensorboard --logdir [your_model_folder]
   ```

> Replace `[your_model_folder]` with the actual folder name where your checkpoints and logs are saved.


---

## 📈 Inference and Analysis

### 1. Run Inference with Trained Model

Use the script below to generate predictions with your fine-tuned model:

```bash
bash grpo/inference.sh
```

> ⚠️ Be sure to modify paths in the script (e.g., model path, test dataset path).

---

### 2. Run Inference with External LLMs (e.g., GPT)

To evaluate using an external model like OpenAI's GPT:

```bash
python eval_process/openai_generator.py
```

> ⚠️ Update the model name, parameters, and set your API key in `eval_process/api_key.py`.

---

### 3. Score Inference Results

After inference, score the outputs using:

```bash
python eval_process/score.py
```

> ⚠️ Edit the script around **line 560** to configure file paths and the URL of the ExecEval Docker service.

---

### 4. Analyze the Results

To analyze the scored results:

```bash
python eval_process/analysis.py
```

> ⚠️ Adjust input/output file paths and analysis settings as needed.

---
