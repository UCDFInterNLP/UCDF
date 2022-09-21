# generate simcse-based sentence embedding through optimal ckpt which is trained on 'train_encoder.sh'

python generate_dense_embeddings.py \
	--model_file=/anaconda3/envs/paper/DPR/output/simcse_ckpt/dpr_biencoder.32 \
	--ctx_src=dpr_wiki \
	--shard_id=0 num_shards=1 \
	--out_file=/anaconda3/envs/paper/DPR/output/DenseEmbedding/SimCSE_embedding