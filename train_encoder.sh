# train dual-encoders with this bash

python -m torch.distributed.launch --nproc_per_node=3 train_dense_encoder.py \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_local \
output_dir=/anaconda3/envs/paper/DPR/output/simcse_ckpt
