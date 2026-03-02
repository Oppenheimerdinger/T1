export CUDA_VISIBLE_DEVICES=0

model_name=T1

for mask_rate in 0.1 0.3 0.5 0.7; do
python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --model_id ILI_mask_${mask_rate} \
  --model $model_name \
  --data ILI \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 128 \
  --patch_size 2 \
  --patch_stride 1 \
  --n_blocks 2 2 \
  --kernel_size_large 71 31 \
  --kernel_size_small 5 \
  --ffn_ratio 1.0 \
  --mask_rate $mask_rate \
  --train_epochs 300 \
  --patience 30 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --lradj type3 \
  --precision bf16 \
  --data_source tslib
done
