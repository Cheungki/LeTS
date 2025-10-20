# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import torch
import numpy as np

from verl import DataProto
from scipy.optimize import linear_sum_assignment
from verl.utils.reward_score import _default_compute_score

def extract_search_content(text):
    try:
        text = text.split('<|im_end|>\n<|im_start|>assistant\n')[1]
    except Exception as e:
        text = text.split('Assistant:', 1)[1]
    # Define the pattern to match text between <user\n and <im_end>\n
    pattern = r'<result>(.*?)</result>'
    # Search for all occurrences of the pattern
    matches = re.findall(pattern, text, re.DOTALL)
    matches = [match.split("\n\n") for match in matches]
    matches = [[m.strip() for m in match] for match in matches]
    return matches

def check_content_sim(search_content_a, search_content_b):
    search_content_a = set(search_content_a)
    search_content_b = set(search_content_b)
    return len(search_content_a.intersection(search_content_b)) / len(search_content_a.union(search_content_b))

def self_check(search_contents):
    self_check_scores = []
    if len(search_contents) == 0:
        return self_check_scores
    else:
        self_check_scores.append(1.0)
    for i in range(1, len(search_contents)):
        contents_sim = []
        for j in range(0, i):
            contents_sim.append(check_content_sim(search_contents[i], search_contents[j]))
        self_check_scores.append(1 - max(contents_sim))

    return self_check_scores

def cross_chain_of_search_check(gold_search_contents, curr_search_contents):
    if len(gold_search_contents) == 0 or len(curr_search_contents) == 0:
        return [0.0 for _ in range(len(curr_search_contents))]
    
    contents_sims = []
    for curr_search_content in curr_search_contents:
        contents_sim = []
        for gold_search_content in gold_search_contents:
            contents_sim.append(check_content_sim(curr_search_content, gold_search_content))
        contents_sims.append(contents_sim)

    def select_maximal_elements(matrix):
        cost_matrix = -matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        selected_elements = matrix[row_indices, col_indices]
        
        return row_indices, selected_elements

    def select_best_match(matrix):
        n, _ = matrix.shape
        row_indices, selected_elements = select_maximal_elements(matrix)
        result_vector = np.zeros(n)
        
        for i, row_idx in enumerate(row_indices):
            result_vector[row_idx] = selected_elements[i]
        return result_vector

    return select_best_match(np.array(contents_sims)).tolist()


class LeTSRewardManagerWithSave():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, save_path=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.save_path = save_path

    def __call__(self, data: DataProto, return_dict=False, curr_save_path=None, config=None):
        """We will expand this function gradually based on the available datasets"""

        if curr_save_path is not None:
            save_path = curr_save_path
        else:
            save_path = self.save_path

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        save_json_lines = []

        if save_path is not None:
            save_file = open(save_path, 'a')
        
        uid2info = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            score = self.compute_score(
                data_source=data_source,
                tokenizer=self.tokenizer,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            if isinstance(score, tuple):
                score, reason, search_path = score
            else:
                reason = ''
                search_path = []
            reward_tensor[i, valid_response_length - 1] = score
            
            save_json_lines.append({
                'data_source': data_source,
                'sequences_str': sequences_str,
                'ground_truth': ground_truth,
                'score': score,
                'reason': reason,
                'search_path': search_path
            })

            if 'uid' in data_item.non_tensor_batch.keys():
                if data_item.non_tensor_batch['uid'] not in uid2info:
                    uid2info[data_item.non_tensor_batch['uid']] = []
                uid2info[data_item.non_tensor_batch['uid']].append([i, score, search_path, extract_search_content(sequences_str)])

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print('-' * 20)
                print(f"data_source: \n{data_source}")
                print(f"sequences_str: \n{sequences_str}")
                print(f"ground_truth: \n{ground_truth}")
                print(f"score: \n{score}")  
                print(f"reason: \n{reason}")
                print(f"search_path: \n{search_path}")
                print('-' * 20)

        # self-contrastive process-based reward based on search path
        stepwise_rewards, stepwise_stds = None, None
        if config is not None:
            stepwise_rewards = [[] for _ in range(len(data))]
            stepwise_stds = [1.0 for _ in range(len(data))]
            for uid, info in uid2info.items():
                stepwise_rewards4uid = self._stepwise_reward(info, config) # List[List[float]]
                # TODO: caculate stepwise_rewards for current uid

                for (i, _, _, _), stepwise_reward in zip(info, stepwise_rewards4uid):
                    save_json_lines[i]["stepwise_reward"] = stepwise_reward
                    stepwise_rewards[i] = stepwise_reward

        if save_path is not None:
            for save_json_line in save_json_lines:
                save_file.write(json.dumps(save_json_line, ensure_ascii=False) + '\n')
            save_file.close()

        if stepwise_rewards is not None:
            if not return_dict:
                return reward_tensor, stepwise_rewards
            else:
                return {"reward_tensor": reward_tensor, "stepwise_rewards": stepwise_rewards}
        else:
            if not return_dict:
                return reward_tensor
            else:
                return {"reward_tensor": reward_tensor}

    def _stepwise_reward(self, grouped_info, config):
        # For rollouts w/. wrong format, we do not employ stepwise reward
        # For rollouts which are absolutely right, we employ self-check stepwise reward to penalize the redundant query
        # For rollouts w/. right format but not perfect answer, we first find its best-match rollouts (w/. most similar path and better performance) and then calculate stepwise reward for them
        max_score = max([info[1] for info in grouped_info])

        stepwise_rewards = []
        for info in grouped_info:
            score, search_path, search_content = info[1], info[2], info[3]
            if score == 0.0:
                # TODO: whether to apply stepwise reward for rollouts w/. wrong format in the final step
                stepwise_rewards.append([])
            elif score == max_score:
                stepwise_rewards.append(self_check(search_content))
            else:
                if max_score < 1.0:
                    stepwise_rewards.append([])
                else:
                    stepwise_reward = [0.0 for _ in range(len(search_path))]
                    for info_ in grouped_info:
                        if score >= info_[1]:
                            continue
                        curr_stepwise_reward = cross_chain_of_search_check(info_[3], search_content)
                        stepwise_reward = curr_stepwise_reward if sum(curr_stepwise_reward) > sum(stepwise_reward) else stepwise_reward
                    stepwise_rewards.append(stepwise_reward)
        return stepwise_rewards
