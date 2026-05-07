#!/bin/bash
export LAYERNORM_TYPE=fast_layernorm
export PROTENIX_DATA_ROOT_DIR="./release_data"

checkpoint_path="./release_data/checkpoint/protenix_base_constraint_v0.5.0.pt"

# python3 ./runner/train.py \
torchrun --nproc_per_node=8 ./runner/train.py \
--model_name "protenix_base_constraint_v0.5.0" \
--run_name constraint \
--seed 42 \
--base_dir ./outputs/finetune/constraint \
--dtype bf16 \
--project protglycandock \
--use_wandb true \
--eval_first true \
--diffusion_batch_size 32 \
--eval_interval 100 \
--log_interval 100 \
--checkpoint_interval 100 \
--ema_decay 0.9999 \
--train_crop_size 640 \
--max_steps 100000 \
--warmup_steps 2000 \
--lr 0.0001 \
--sample_diffusion.N_step 20 \
--triangle_attention "triattention" \
--triangle_multiplicative "cuequivariance" \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.train_sets glyco_train_all_on \
--data.test_sets glyco_valid_all_on \
--loss.weight.alpha_bond 1.0 \
--loss.weight.smooth_lddt 0.0 \
--loss.diffusion_lddt_loss_dense false
