# Final SPFT

DFT/SPFT + AdamW, adapted from upstream DFT `11e395d4`. Source and license details: `UPSTREAM.md`.

## Math Reasoning

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
bash verl/download_datasets.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash verl/sweep_dft_1gpu.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash verl/sweep_spft_1gpu.sh
python -m pytest -q tests
```

- Default training dataset: 100,000 Numina train examples. Numina test is used as validation. Set `VAL_FILE` to use another parquet validation file with `extra_info.question/answer`.
- Global batch is 256 and micro-batch/GPU is 8 by default. Override with `TRAIN_BATCH_SIZE` and `MICRO_BATCH_SIZE_PER_GPU`. Global batch must be divisible by `N_GPUS * MICRO_BATCH_SIZE_PER_GPU`.
- Sweeps: `OPTIM_LRS`, `OPTIM_WEIGHT_DECAYS`, `EPOCHS_LIST`, `SPFT_LAMBDAS`. Change model with `MODEL_NAME`; set `DRY_RUN=1` to print the launch command. Extra Hydra overrides can be appended to the command.
- Liger is off by default. `USE_LIGER=true` requires `liger-kernel`; `USE_WANDB=true` requires `wandb`. If FlashAttention is unavailable, append `model.attention=sdpa`.
- Full fine-tuning uses FSDP1. SPFT keeps a fixed reference model, one replica per GPU, with optional `SPFT_REFERENCE_CPU_OFFLOAD=true`. Saved checkpoints are Hugging Face checkpoints and do not include optimizer state/resume state.
- Gradient accumulation and token normalization are computed over the full global batch. Validation covers every example exactly once across ranks. `weight_threshold` is metric-only.

Math eval, preferably in a separate environment:

```bash
python -m pip install -r requirements-eval.txt
bash verl/download_datasets.sh --eval-only
MODEL_NAME_OR_PATH=/path/to/checkpoint EVAL_CUDA_VISIBLE_DEVICES=0,1 bash verl/eval_dft.sh
```

Supported math eval sets: `math`, `math_oai`, `minerva_math`, `olympiadbench`, `aime24`, `amc23`. Defaults: qwen-boxed prompt, `n=16`, temperature 1, top-p 1, max tokens 4096. Override with `EVAL_DATASETS`, `EVAL_N_SAMPLING`, `EVAL_TEMPERATURE`, `EVAL_TOP_P`, `EVAL_MAX_TOKENS`, `EVAL_SEED`, `OUTPUT_DIR`. ANTLR 4.11.1 is required. Keep GPU count fixed when comparing sampling runs; `mean_acc` is not pass@16.

## Code Generation

This adds the exploratory setup from paper section 4.3: UltraFeedback SFT data, one epoch, learning rate `5e-5`, warmup ratio `0.05`, global batch size `16`, and HumanEval/HumanEval+/MultiPL-E eval scripts.

```bash
python -m pip install -r requirements.txt
bash verl/download_codegen_dataset.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 CODEGEN_MODELS=Qwen/Qwen2.5-Coder-3B bash verl/sweep_codegen_dft.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 CODEGEN_MODELS=Qwen/Qwen2.5-Coder-3B bash verl/sweep_codegen_spft.sh
```

- Dataset output: `verl/data/ultrafeedback_codegen/{train,test}.parquet`.
- Downloader source defaults to `openbmb/UltraFeedback`. It samples usable rows, selects the response with the highest average numeric score, and writes 10,000 train rows plus 500 validation rows by default.
- Downloader knobs: `ULTRAFEEDBACK_SOURCE`, `ULTRAFEEDBACK_SPLIT`, `ULTRAFEEDBACK_REVISION`, `CODEGEN_TRAIN_SIZE`, `CODEGEN_VAL_SIZE`, `SEED`.
- Training wrappers set `DATASET_TYPE=ultrafeedback`, `TRAIN_BATCH_SIZE=16`, `MICRO_BATCH_SIZE_PER_GPU=1`, `WARMUP_STEPS_RATIO=0.05`, `TOTAL_EPOCHS=1`, and `OPTIM_LR=5e-5` by default.
- Model sweep knob: `CODEGEN_MODELS`, for example `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-Coder-3B`, or `Qwen/Qwen2.5-Coder-7B`.

HumanEval/HumanEval+ eval uses EvalPlus:

```bash
python -m pip install -r requirements-eval.txt
MODEL_NAME_OR_PATH=/path/to/checkpoint EVAL_CUDA_VISIBLE_DEVICES=0,1 bash verl/eval_codegen.sh
```

MultiPL-E generation on the GPU server:

```bash
git clone https://github.com/nuprl/MultiPL-E /home/datpv/SPFT_code/MultiPL-E

MODEL_NAME_OR_PATH=/path/to/checkpoint \
CODEGEN_EVALS=multiple \
MULTIPLE_REPO=/home/datpv/SPFT_code/MultiPL-E \
MULTIPLE_RUN_TESTS=false \
OUTPUT_DIR=/home/datpv/SPFT_code/verl/codegen_eval_outputs/spft_codegen_multipl_e_20 \
bash verl/eval_codegen.sh
```

`MULTIPLE_LANGS` defaults to `py,cpp,java,php,ts,cs,sh,js`, matching the eight-language MultiPL-E reporting setup. MultiPL-E defaults to `MULTIPLE_TEMPERATURE=0.2`, `MULTIPLE_TOP_P=0.95`, `MULTIPLE_COMPLETION_LIMIT=200`, `MULTIPLE_MAX_TOKENS=1024`, and `MULTIPLE_BATCH_SIZE=20`; `automodel.py` requires a strictly positive temperature. The script builds the Python reworded prompt JSONL automatically from `datasets/originals-with-cleaned-doctests`; override it with `MULTIPLE_PY_PROMPT=/path/to/humaneval-py-reworded.jsonl` if you already prepared one. Output run names use the checkpoint basename by default to avoid very long paths; set `MULTIPLE_NAME_OVERRIDE=short_name` to choose one explicitly. `MULTIPLE_RUN_TESTS=false` only generates completions, which is useful when the GPU server has no Docker daemon.

Copy generated completions from the server to a local machine with Docker:

```powershell
scp -r datpv@worker-0:/home/datpv/SPFT_code/verl/codegen_eval_outputs/spft_codegen_multipl_e_20 C:\Users\Admin\Downloads\spft_codegen_multipl_e_20
```

Evaluate the completions locally with Docker. Use short local paths before running `pass_k.py`; long Windows paths can make Python miss the result files.

```powershell
cd "C:\Users\Admin\Downloads\final spft\MultiPL-E"

$srcRoot = "C:\Users\Admin\Downloads\spft_codegen_multipl_e_20\multiple"
$dstRoot = "C:\meval20\multiple"
New-Item -ItemType Directory -Force $dstRoot | Out-Null

foreach ($lang in @("py","cpp","cs","java","js","php","sh","ts")) {
    $run = Get-ChildItem "$srcRoot\$lang" -Directory | Select-Object -First 1
    if ($null -eq $run) { Write-Host "Skipping missing $lang"; continue }
    $dst = "$dstRoot\$lang"
    New-Item -ItemType Directory -Force $dst | Out-Null
    robocopy $run.FullName $dst /E
    docker run --rm --network none -v "${dst}:/out:rw" ghcr.io/nuprl/multipl-e-evaluation --dir /out --output-dir /out
}

python pass_k.py `
  C:\meval20\multiple\py `
  C:\meval20\multiple\cpp `
  C:\meval20\multiple\java `
  C:\meval20\multiple\php `
  C:\meval20\multiple\ts `
  C:\meval20\multiple\cs `
  C:\meval20\multiple\sh `
  C:\meval20\multiple\js `
  > C:\meval20\multiple\pass_k.txt

Get-Content C:\meval20\multiple\pass_k.txt
```

The `Estimate` column times 100 is the paper-style percentage for each MultiPL-E language. The MultiPL-E average is the mean of Python, C++, Java, PHP, TS, C#, Bash, and JS. EvalPlus and MultiPL-E outputs default to `verl/codegen_eval_outputs/...`.
