wandb enabled
wandb offline


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python train_FKD.py \
    --batch-size 1024 \
    --model resnet101 \
    --cos \
    -j 4 --gradient-accumulation-steps 2 \
    -T 20 \
    --mix-type 'cutmix' \
    --wandb-api-key xxxxxxxxxxxxx \
    --output-dir ./save/final_rn18_fkd/imagenet_ipc10_label18_train101_v1 \
    --train-dir /home-ext/tbw/suduo/data/init_data/imagenet_ipc10_s0.7_g8.0_kmexpand1_v1 \
    --val-dir /home-ext/tbw/suduo/data/imagenet/val \
    --fkd-path /home-ext/tbw/suduo/SRe2L/relabel/imagenet_ipc10_label18_v1 \
    --wandb-project imagenet_ipc10_label18_repeat  \
