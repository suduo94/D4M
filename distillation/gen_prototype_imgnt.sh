CUDA_VISIBLE_DEVICES=0 python gen_prototype.py \
    --batch_size 10 \
    --data_dir ../data/imagenet \
    --dataset imagenet \
    --diffusion_checkpoints_path ../stablediffusion/checkpoints/stable-diffusion-v1-5 \
    --ipc 10 \
    --km_expand 1 \
    --label_file_path ./label-propmt/imagenet_classes.txt \
    --save_prototype_path ./prototypes