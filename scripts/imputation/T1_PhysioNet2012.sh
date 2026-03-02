export CUDA_VISIBLE_DEVICES=0

model_name=T1

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --model_id PhysioNet2012 \
  --model $model_name \
  --data PhysioNet2012 \
  --root_path ./dataset/benchpots/ \
  --features M \
  --seq_len 48 \
  --label_len 0 \
  --pred_len 48 \
  --enc_in 37 \
  --dec_in 37 \
  --c_out 37 \
  --n_heads 128 \
  --patch_size 2 \
  --patch_stride 1 \
  --n_blocks 2 2 \
  --kernel_size_large 35 15 \
  --kernel_size_small 5 \
  --ffn_ratio 1.0 \
  --drop_attn 0.1 \
  --drop_path 0.1 \
  --mask_rate 0.2 \
  --train_epochs 300 \
  --patience 30 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --lradj type3 \
  --precision bf16 \
  --data_source benchpots
