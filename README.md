<div align="center">

# ***LeTS***: ***Le***arning to ***T***hink-and-***S***earch via Process-and-Outcome Reward Hybridization

</div>

We introduce ***LeTS***, a novel framework that trains LLMs to ***le***arn to ***t***hink-and-***s***earch via process-and-outcome reward hybridization—without requiring any supervised data on reasoning steps (both process- and outcome-level). This is a work in progress and we are actively working on it.

## 📦 Installation

We recommend using conda to manage the environment. First create a conda environment and activate it.
```bash
conda create -n lets python==3.10
conda activate lets
```
Then install dependencies, and the packages under ```src/``` will be installed in the editable mode.  Check out ```setup.py``` for details.
```bash
git clone https://github.com/Cheungki/LeTS.git
cd LeTS
pip3 install -e .
pip3 install flash-attn --no-build-isolation
```
If you want to host a Wikipedia RAG system based on FlashRAG, you need to install faiss-gpu as follow. As described in the [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#wrench-installation), due to the incompatibility when installing faiss using pip, we need to use the following conda command to install faiss-gpu.
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## 🚀 Quick Start

### Data Preparation

*LeTS* is trained on the training set of MuSiQue, and evaluated on the dev set of HotpotQA, 2WikiMultiHopQA, MuSiQue and Bamboogle. For downloading the datasets, please refer to the `data/download_dataset.sh` script.
```bash
cd data
bash download_dataset.sh
```

For preparing the training and validation data for following reinforcement learning, please run this script to parse the MuSiQue dataset to the parquet format.
```bash
cd data
python prepare_musique.py
```

### Retriever Serving

For training on MuSiQue data with a Wikipedia search tool, we provide a Wikipedia retriever service implemented using FlashRAG and FastAPI. Before starting the retriever serving, you need download the [pre-indexed wikipedia](https://github.com/RUC-NLPIR/FlashRAG?tab=readme-ov-file#index), [wikipedia corpus and corresponding retriever models](https://github.com/RUC-NLPIR/FlashRAG/blob/main/docs/original_docs/reproduce_experiment.md#preliminary). More details can be found in the documentation of FlashRAG.

For starting the retriever serving, you need to first fill the `scripts/serving/retriever_config.yaml` with the correct path to the retrieval model, index, and corpus, and available GPU ids. Then, you can run the following command to start the retriever serving:
```bash
cd scripts/serving
python retriever_serving.py \
    --config retriever_config.yaml \
    --num_retriever {num_retriever} \  
    --port {port}
```

### Training

Our training framework is based on [verl](https://github.com/volcengine/verl), a powerful reinforcement learning framework for LLMs. We deeply customize the verl code to fit our needs, and the modified version of verl is under the `src/verl` directory. The example of training scripts are under `scripts/train`.

Here is an example of training Qwen2.5-7B-Instruct with 4 GPUs locally. Note that the training script below **is just an example** for single-node training, using small batch size for quick start, and do not assure the training performance.
```bash
cd scripts/train
bash train.sh \
    --train_batch_size 8 \
    --ppo_mini_batch_size 4 \
    --use_lets True \
    --prompt_template_name lets_template_sys \
    --actor_model_path {model/path/to/qwen2.5-7b-instruct} \
    --search_url {your-hosted-retriever-url} \
    --sandbox_url {your-hosted-sandbox-url} \
    --project_name {wandb-project-name} \
    --experiment_name {wandb-experiment-name} \
    --nnodes 1 \
    --n_gpus_per_node 4 \
    --save_freq 5 \
    --test_freq 5 \
    --total_epochs 2 \
    --wandb_api_key {your-wandb-api-key} \
    --save_path {path/to/save} \
    --train_files "['train1.parquet', 'train2.parquet']" \
    --test_files "['test1.parquet', 'test2.parquet']"
```

### Inference
For model serving, we recommend using [SGLang](https://docs.sglang.ai/). You can either download our open-source models or train your own models to conduct the inference. Here is an example of how to launch the model service:
```bash
python3 -m sglang.launch_server \
        --served-model-name {trained/model/name} \
        --model-path {trained/model/path} \
        --tp 2 \
        --context-length 8192 \
        --enable-metrics \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --port 80 \
        --trust-remote-code \
        --disable-overlap \
        --disable-radix-cache
```

### Evaluation

#### Multi-hop QA

For the evaluation on multi-hop QA, we use [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) as the standard evaluation environment. For downloading the evaluation data, please run the following command:
```bash
cd data
bash download_dataset.sh
```
Here is an example of evaluating the performance of LeTS-Qwen-7B-Instruct on Bamboogle test set.
```bash
cd scripts/evaluation
python run_eval.py \
    --config_path eval_config.yaml \
    --method_name lets \
    --data_dir {root/path/to/evaluation/data} \
    --dataset_name bamboogle \
    --split test \
    --save_dir {your-save-dir} \
    --save_note lets_qwen7b_ins
    --sgl_remote_url {your-launched-sgl-url} \
    --remote_retriever_url {your-hosted-retriever-url} \
    --generator_model {your-local-model-path}
```
For more details about the configuration, please refer to the `scripts/evaluation/eval_config.yaml` file. 

## 🤝 Acknowledge

This training implementation is highly built on [verl](https://github.com/volcengine/verl) and [recall](https://github.com/Agent-RL/ReCall), and the evaluation is based on [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) and BFCL. The serving of sandbox and retriever is based on [FastAPI](https://github.com/fastapi/fastapi). The model serving is based on [SGLang](https://docs.sglang.ai/). *LeTS* models are trained based on [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/). We sincerely appreciate their contributions to the open-source community.

## 📚 Citation

If you find this work useful, please cite it as follows:
```bibtex
@misc{zhang2025lets,
      title={LeTS: Learning to Think-and-Search via Process-and-Outcome Reward Hybridization}, 
      author={Qi Zhang and Shouqing Yang and Lirong Gao and Hao Chen and Xiaomeng Hu and Jinglei Chen and Jiexiang Wang and Sheng Guo and Bo Zheng and Haobo Wang and Junbo Zhao},
      year={2025},
      eprint={2505.17447},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.17447}, 
}
```