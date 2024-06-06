#!/bin/bash

MODEL_ARGS="\
--variant=llama3.70b
--model_path="/gpfs/llama3/hf/70b_instruction_tuned"
--tokenizer="/gpfs/llama3/hf/70b_instruction_tuned"
--model_source=hf
--architecture=llama
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/llama3-70b-ropefixed-tie_wt-scalednorm-4node-backup/checkpoints/step_11186_ckp.pth"
--speculator_ckpt_singlefile
--top_k_tokens_per_head=4,3,2,2
--batch_input
"

MODEL_ARGS0="\
--variant=ibm.20b
--model_path="/gpfs/prangan/granite-20b-code-instruct"
--tokenizer="/gpfs/prangan/granite-20b-code-instruct"
--model_source=hf
--architecture=gpt_bigcode
--prompt_type="code"
--speculator_path="/gpfs/suneja/checkpoints/grantite-20b-code-instruct-v1-speculator/step_42001_ckp.pth"
--speculator_ckpt_singlefile
--top_k_tokens_per_head=4,3,2,2
"

MODEL_ARGS1="\
--variant="llama3.8b"
--model_path="/gpfs/llama3/hf/8b_instruction_tuned"
--tokenizer="/gpfs/llama3/hf/8b_instruction_tuned"
--model_source=hf
--architecture=llama
--speculator_path="/gpfs/suneja/models/hub/models--ibm-fms--llama3-8b-accelerator/snapshots/56660560945bc17b7958655bf15546bf8b11fced/"
--speculator_variant=3_2b
--speculator_source=hf
--top_k_tokens_per_head=4,3,2,2
"

MODEL_ARGS_GRANITE8B="\
--variant=calico.8b.code
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-8b-code-instruct/snapshots/8a0fc76e4d374188e0cc8794d2d7275aa5aa7e64"
--tokenizer="/gpfs/prangan/hub/models--ibm-granite--granite-8b-code-instruct/snapshots/8a0fc76e4d374188e0cc8794d2d7275aa5aa7e64"
--model_source=hf
--architecture=llama
--prompt_type="code"
--speculator_ckpt_singlefile
--top_k_tokens_per_head=5,4,3,2,2
"
#--speculator_path="/gpfs/prangan/backup-ckptx/checkpoints/step_15001_ckp.pth"

MODEL_ARGS_GRANITE3B="\
--variant=calico.3b.code
--model_path="/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/"
--tokenizer="/gpfs/prangan/hub/models--ibm-granite--granite-3b-code-instruct/snapshots/4420bfb5a3361ab4714bbd653848ef1a819d9f5b/"
--model_source=hf
--architecture=llama
--prompt_type="code"
--speculator_ckpt_singlefile
--top_k_tokens_per_head=5,4,3,2,2
"
#--speculator_path="/gpfs/prangan/ckpts/granite_3b_stage1/checkpoints/step_21001_ckp.pth"


export CUDA_VISIBLE_DEVICES=1

torchrun \
    --nproc_per_node=1 \
    scripts/paged_speculative_inference_new.py \
    ${MODEL_ARGS_GRANITE8B}
