CUDA_VISIBLE_DEVICES=1 python gen_prototype.py \
    --batch_size 10 \
    --data_dir /home-ext/tbw/suduo/data/imagenet \
    --dataset imagenet \
    --diffusion_checkpoints_path /home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5 \
    --ipc 10 \
    --km_expand 1 \
    --label_file_path /home-ext/tbw/suduo/D3M/label-propmt/imagenet_classes.txt \
    --save_prototype_path /home-ext/tbw/suduo/D3M/prototypes