export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LD_LIBRARY_PATH=/opt/conda/envs/lets/lib:$LD_LIBRARY_PATH

conda activate lets

bash train.sh \
    --train_batch_size 8 \
    --ppo_mini_batch_size 8 \
    --prompt_template_name lets_template_sys \
    --actor_model_path /input2/zhangqi.zq/models/Qwen2.5-7B-Instruct \
    --search_url http://0.0.0.0:8888 \
    --project_name step_research \
    --experiment_name debug \
    --nnodes 1 \
    --n_gpus_per_node 8 \
    --save_freq 1 \
    --test_freq 1 \
    --total_epochs 2 \
    --save_path /input2/zhangqi.zq/experiments/LeTS/debug \
    --train_files /input2/zhangqi.zq/raw/musique/train.parquet \
    --test_files /input2/zhangqi.zq/raw/musique/test.parquet \
    --stepwise_reward True \
    --self_check_mode best \
    --self_check_std 0.1 \
    --contrastive_mode similar \
    --contrastive_std 0.1 \
    --use_hybrid True \
    --sentence_level_std True \
    --is_reweight True \
    --rescale True \
    --gamma 0.1 \
    --val_before_train False