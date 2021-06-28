CUDA_VISIBLE_DEVICES=7 \
    python3 train.py \
    --model_name stage1_nyuv2rec_geo1e-1_smooth1e-4 \
    --use_geo_loss --geo_loss_weights 1e-1 \
    --split nyuv2rec --dataset nyuv2rec --data_path dataset/NYUv2_rectified/train --min_depth 0.1 --max_depth 10 --height 256 --width 320 \
    --disparity_smoothness 1e-4 \
    --log_dir log2 --png \
    --batch_size 12 \
    --num_workers 6 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --num_epochs 60
