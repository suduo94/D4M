


# CUDA_VISIBLE_DEVICES=0 python gen_syn_image_hjj.py \
#     --dataset imagenet \
#     --diffusion_checkpoints_path /home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5 \
#     --guidance_scale 8 \
#     --ipc 10 \
#     --km_expand 1 \
#     --label_file_path /home-ext/tbw/suduo/D3M/label-propmt/imagenet_classes.txt \
#     --strength 0.01 \
#     --prototype_path /home-ext/tbw/suduo/D3M/prototypes/imagenet-ipc10-kmexpand1.json \
#     --save_init_image_path /home-ext/tbw/suduo/data/init_data/new



CUDA_VISIBLE_DEVICES=7 python gen_syn_image_hjj.py \
    --dataset imagenet \
    --diffusion_checkpoints_path /home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5 \
    --guidance_scale 8 \
    --ipc 10 \
    --km_expand 1 \
    --label_file_path /home-ext/tbw/suduo/D3M/label-propmt/imagenet_classes.txt \
    --strength 0.7 \
    --prototype_path /home-ext/tbw/suduo/D3M/prototypes/imagenet-ipc10-kmexpand1.json \
    --save_init_image_path /home-ext/tbw/suduo/data/init_data/new






# CUDA_VISIBLE_DEVICES=7 python gen_syn_image_hjj.py \
#     --dataset imagenet \
#     --diffusion_checkpoints_path /home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5 \
#     --guidance_scale 8 \
#     --ipc 10 \
#     --km_expand 1 \
#     --label_file_path /home-ext/tbw/suduo/D3M/label-propmt/imagenet_classes.txt \
#     --strength 0.3 \
#     --prototype_path /home-ext/tbw/suduo/D3M/prototypes/imagenet-ipc10-kmexpand1.json \
#     --save_init_image_path /home-ext/tbw/suduo/data/init_data/new


# CUDA_VISIBLE_DEVICES=7 python gen_syn_image_hjj.py \
#     --dataset imagenet \
#     --diffusion_checkpoints_path /home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5 \
#     --guidance_scale 8 \
#     --ipc 10 \
#     --km_expand 1 \
#     --label_file_path /home-ext/tbw/suduo/D3M/label-propmt/imagenet_classes.txt \
#     --strength 0.5 \
#     --prototype_path /home-ext/tbw/suduo/D3M/prototypes/imagenet-ipc10-kmexpand1.json \
#     --save_init_image_path /home-ext/tbw/suduo/data/init_data/new

# CUDA_VISIBLE_DEVICES=7 python gen_syn_image_hjj.py \
#     --dataset imagenet \
#     --diffusion_checkpoints_path /home-ext/tbw/suduo/D3M/stablediffusion/checkpoints/stable-diffusion-v1-5 \
#     --guidance_scale 8 \
#     --ipc 10 \
#     --km_expand 1 \
#     --label_file_path /home-ext/tbw/suduo/D3M/label-propmt/imagenet_classes.txt \
#     --strength 0.9 \
#     --prototype_path /home-ext/tbw/suduo/D3M/prototypes/imagenet-ipc10-kmexpand1.json \
#     --save_init_image_path /home-ext/tbw/suduo/data/init_data/new


