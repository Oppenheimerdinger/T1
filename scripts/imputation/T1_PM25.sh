export CUDA_VISIBLE_DEVICES=0

model_name=T1

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --model_id PM25 \
  --model $model_name \
  --data PM25 \
  --root_path ./dataset/pm25/ \
  --features M \
  --seq_len 36 \
  --label_len 0 \
  --pred_len 36 \
  --enc_in 36 \
  --dec_in 36 \
  --c_out 36 \
  --n_heads 128 \
  --patch_size 2 \
  --patch_stride 1 \
  --n_blocks 2 2 \
  --kernel_size_large 71 31 \
  --kernel_size_small 5 \
  --ffn_ratio 1.0 \
  --mask_rate 0.2 \
  --train_epochs 300 \
  --patience 30 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --lradj type3 \
  --precision bf16 \
  --data_source csdi
