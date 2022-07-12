python test.py \
--name \
deepfashion \
--dataset_mode \
deepfashion \
--dataroot \
[ROOT OF DEEPFAHION DATASET] \
--no_flip \
--video_like \
--gpu_ids \
0 \
--netG \
dynast \
--batchSize \
4 \
--load_size \
256 \
--crop_size \
256 \
--n_layers \
2 \
--max_multi \
8 \
--ngf \
64 


