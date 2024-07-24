CUDA_VISIBLE_DEVICES=0 python gen_syn_image.py \
    --dataset imagenet \
    --diffusion_checkpoints_path ../stablediffusion/checkpoints/stable-diffusion-v1-5 \
    --guidance_scale 8 \
    --strength 0.7 \
    --ipc 10 \
    --km_expand 1 \
    --label_file_path ./label-propmt/imagenet_classes.txt \
    --prototype_path ./prototypes/imagenet-ipc10-kmexpand1.json \
    --save_init_image_path ../data/distilled_data/