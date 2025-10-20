export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export MKL_SERVICE_FORCE_INTEL=1
export WANDB_MODE=offline

conda activate lets

PROMPT_KEY=question
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=256
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=8192
PROMPT_TEMPLATE_NAME=lets_template_sys
ACTOR_MODEL_PATH=/your/model/path
REWARD_MANAGER=lets
ROLLOUT_N=5
MAX_TURNS=5
SEARCH_URL=/your/search/url
PROJECT_NAME=project-name-on-wandb
EXPERIMENT_NAME=experiment-name-on-wandb
NNODES=1
N_GPUS_PER_NODE=4
SAVE_FREQ=10
TEST_FREQ=10
TOTAL_EPOCHS=2
WANDB_API_KEY=None
SAVE_PATH=/your/save/path
TRAIN_FILES=/your/train/file/path
TEST_FILES=/your/test/file/path
STEPWISE_REWARD=False
SELF_CHECK_MODE=best
SELF_CHECK_AGG=False
SELF_CHECK_STD=1.0
CONTRASTIVE_MODE=similar
CONTRASTIVE_AGG=False
CONTRASTIVE_STD=1.0
USE_HYBRID=False
SENTENCE_LEVEL_STD=False
FORWARD_ACC=False
IS_REWEIGHT=False
RESCALE=False
GAMMA=1.0
VAL_BEFORE_TRAIN=True

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt_key) PROMPT_KEY="$2"; shift 2;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2;;
        --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2;;
        --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2;;
        --prompt_template_name) PROMPT_TEMPLATE_NAME="$2"; shift 2;;
        --actor_model_path) ACTOR_MODEL_PATH="$2"; shift 2;;
        --max_turns) MAX_TURNS="$2"; shift 2;;
        --reward_manager) REWARD_MANAGER="$2"; shift 2;;
        --rollout_n) ROLLOUT_N="$2"; shift 2;;
        --search_url) SEARCH_URL="$2"; shift 2;;
        --project_name) PROJECT_NAME="$2"; shift 2;;
        --experiment_name) EXPERIMENT_NAME="$2"; shift 2;;
        --nnodes) NNODES="$2"; shift 2;;
        --n_gpus_per_node) N_GPUS_PER_NODE="$2"; shift 2;;
        --save_freq) SAVE_FREQ="$2"; shift 2;;
        --test_freq) TEST_FREQ="$2"; shift 2;;
        --total_epochs) TOTAL_EPOCHS="$2"; shift 2;;
        --wandb_api_key) WANDB_API_KEY="$2"; shift 2;;
        --save_path) SAVE_PATH="$2"; shift 2;;
        --train_files) TRAIN_FILES="$2"; shift 2;;
        --test_files) TEST_FILES="$2"; shift 2;;
        --stepwise_reward) STEPWISE_REWARD="$2"; shift 2;;
        --self_check_mode) SELF_CHECK_MODE="$2"; shift 2;;
        --self_check_agg) SELF_CHECK_AGG="$2"; shift 2;;
        --self_check_std) SELF_CHECK_STD="$2"; shift 2;;
        --contrastive_mode) CONTRASTIVE_MODE="$2"; shift 2;;
        --contrastive_agg) CONTRASTIVE_AGG="$2"; shift 2;;
        --contrastive_std) CONTRASTIVE_STD="$2"; shift 2;;
        --use_hybrid) USE_HYBRID="$2"; shift 2;;
        --sentence_level_std) SENTENCE_LEVEL_STD="$2"; shift 2;;
        --forward_acc) FORWARD_ACC="$2"; shift 2;;
        --is_reweight) IS_REWEIGHT="$2"; shift 2;;
        --rescale) RESCALE="$2"; shift 2;;
        --gamma) GAMMA="$2"; shift 2;;
        --val_before_train) VAL_BEFORE_TRAIN="$2"; shift 2;;
        *)
            echo "unknown argument '$1'" >&2
            exit 1;;
    esac
done

if [ "$WANDB_API_KEY" != "None" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

ROLLOUT_SAVE_PATH=${SAVE_PATH}/rollout
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=${GAMMA} \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TEST_FILES" \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.prompt_template_name=${PROMPT_TEMPLATE_NAME} \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm_with_search \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.search_url=${SEARCH_URL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_turns=${MAX_TURNS} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.enable_stepwise_reward=${STEPWISE_REWARD} \
    reward_model.stepwise_reward_config.self_check_mode=${SELF_CHECK_MODE} \
    reward_model.stepwise_reward_config.self_check_agg=${SELF_CHECK_AGG} \
    reward_model.stepwise_reward_config.self_check_std=${SELF_CHECK_STD} \
    reward_model.stepwise_reward_config.contrastive_mode=${CONTRASTIVE_MODE} \
    reward_model.stepwise_reward_config.contrastive_agg=${CONTRASTIVE_AGG} \
    reward_model.stepwise_reward_config.contrastive_std=${CONTRASTIVE_STD} \
    reward_model.stepwise_reward_config.use_hybrid=${USE_HYBRID} \
    reward_model.stepwise_reward_config.sentence_level_std=${SENTENCE_LEVEL_STD} \
    reward_model.stepwise_reward_config.is_reweight=${IS_REWEIGHT} \
    reward_model.stepwise_reward_config.rescale=${RESCALE} \
    reward_model.stepwise_reward_config.forward_acc=${FORWARD_ACC} \
    trainer.critic_warmup=0 \
    trainer.logger="[console,wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.rollout_save_path=${ROLLOUT_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs | tee ${SAVE_PATH}/run.log