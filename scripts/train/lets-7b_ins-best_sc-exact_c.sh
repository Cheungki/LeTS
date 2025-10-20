# old code + 8192 + max turn + max gen 512

export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/opt/conda/envs/lets/lib:$LD_LIBRARY_PATH

conda activate lets

MODEL_NAME="lets-7b_ins-best_sc-exact_c"

bash train.sh \
    --train_batch_size 256 \
    --ppo_mini_batch_size 256 \
    --prompt_template_name lets_template_sys \
    --actor_model_path /input2/zhangqi.zq/models/Qwen2.5-7B-Instruct \
    --search_url http://0.0.0.0:8888 \
    --project_name lets \
    --experiment_name ${MODEL_NAME} \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 10 \
    --test_freq 10 \
    --total_epochs 2 \
    --save_path /input2/zhangqi.zq/experiments/LeTS/${MODEL_NAME} \
    --train_files /input2/zhangqi.zq/raw/musique/train.parquet \
    --test_files /input2/zhangqi.zq/raw/musique/test.parquet \
    --stepwise_reward True \
    --self_check_mode best \
    --self_check_std 0.1 \
    --contrastive_mode exact \
    --contrastive_std 0.1 \

# merge model
cd /input2/zhangqi.zq/LeTS
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python model_merger.py \
    --backend fsdp \
    --hf_model_path "/input2/zhangqi.zq/models/Qwen2.5-7B-Instruct" \
    --local_dir "/input2/zhangqi.zq/experiments/LeTS/${MODEL_NAME}/global_step_154/actor" \
    --target_dir "/input2/zhangqi.zq/experiments/LeTS/${MODEL_NAME}/global_step_154/actor/huggingface" \
    --tie-word-embedding

rm /input2/zhangqi.zq/experiments/LeTS/${MODEL_NAME}/global_step_154/actor/*.pt -f