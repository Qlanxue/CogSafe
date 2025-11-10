

import json
import jsonlines
import copy
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel
from utils import *
from prompt import *
from FlagEmbedding import FlagModel
from tqdm import tqdm
from transformers import pipeline
import torch

class GetTree:
    def __init__(self, strategy_file, model_path):
        self.strategy_dict, self.strategy_name, self.all_strategy = self.load_strategy(strategy_file)
        self.model = FlagModel(model_path, query_instruction_for_retrieval='', use_fp16=True)

        self.strategy_embeddings = self.model.encode(self.all_strategy)

    def load_strategy(self, filename):
        strategy_id = {'END': 0}
        all_strategy = ['END']
        all_name = ['']
        cnt = 1
        with jsonlines.open(filename, 'r') as fr:
            for line in fr:
                all_strategy.append(line['strategy name'] + ':' + line['description'])
                all_name.append(line['strategy name'])
                strategy_id[line['strategy name'] + ':' + line['description']] = cnt
                cnt += 1
        return strategy_id, all_name, all_strategy

    def get_tree(self, tree_dict, node, node_set):
        cur_tree = {}
        cur_tree['node strategy'] = self.all_strategy[node]
        if node in tree_dict:
            cur_tree['leaf strategy'] = []
            for child in tree_dict[node]:
                if child not in node_set:
                    inner_set = copy.deepcopy(node_set)
                    inner_set.add(node)
                    cur_tree['leaf strategy'].append(self.get_tree(tree_dict, child, inner_set))
        return cur_tree

    def structure_tree(self, strategy_list):
        cur_tree = {}
        print(strategy_list)
        for i in strategy_list:
            for j in range(1, len(i)):
                if i[j - 1] in cur_tree:
                    cur_tree[i[j - 1]].add(i[j])
                else:
                    cur_tree[i[j - 1]] = set()
                    cur_tree[i[j - 1]].add(i[j])
        try:
            node = strategy_list[0][0]
            strategy_tree = self.get_tree(cur_tree, node, set())
            strategy_tree = json.dumps(strategy_tree)
        except:
            strategy_tree = ''
        print(strategy_tree)
        return strategy_tree

    def retrieval(self, query, inner_level, tree, score_ratio, depth):
        q_embeddings = self.model.encode_queries(query)
        scores = q_embeddings @ self.strategy_embeddings.T
        scores_id = np.argsort(scores)
        scores_id = scores_id[::-1]
        print(np.sort(scores)[-5:])
        cur_random = random.randint(1, 3)
        if inner_level == 1:
            cur_random = 1

        top3_strategy = []
        for i in range(5):
            top3_strategy.append(scores_id[i])
        random.shuffle(top3_strategy)
        cur_list = []
        inner_cnt = 0
        for i in range(cur_random):
            if len(tree) >= 1 and top3_strategy[i] == tree[-1]:
                continue
            inner_cnt += 1
            if top3_strategy[i] != 0 and len(tree) < 2 and scores[top3_strategy[i]] >= score_ratio:
                line = self.retrieval(query + self.all_strategy[scores_id[i]] + '->', inner_level + 1, tree + [int(top3_strategy[i])], score_ratio, depth)
                cur_list.extend(line)
            elif len(tree) < depth:
                cur_list.append(tree + [int(top3_strategy[i])])
        return cur_list

# You can customize how your LLM is used here.
class LLM:
    def __init__(self, model_name, huggingface=False):
        self.model_name = model_name
        self.huggingface = huggingface
        self.init_mdoel()

    def init_mdoel(self):
        if self.huggingface:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

    def generate_response(self, messages):
        if self.huggingface:
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors='pt'
            ).to(self.model.device)
            prompt_len = input_ids.shape[1]
            output = self.model.generate(
                input_ids,
                max_new_tokens=20,
                pad_token_id=0,
            )
            generated_tokens = output[:, prompt_len:]
            response = self.tokenizer.decode(generated_tokens[0])
        else:
            response = get_response(messages, model=self.model_name)
        return response


class OmniAgent:
    def __init__(self, args):
        self.args = args
        self.history_control = ''
        self.round_id = 1
        self.strategy_tool = GetTree(self.args.strategy_file, self.args.retrieval_model)
        self.omni_model = self.args.omni_model
        self.llm_model = LLM(self.args.llm_model)
        self.result_file = self.args.result_file

    # TODO
    def load_omni(self):
        if self.args.omni_model_path:
            self.omni_model_hf = pipeline(
                "text-generation",
                model=self.args.omni_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        else:
            self.omni_model = self.args.omni_model

    def huggingface_response(self, system_prompt, instruction):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        outputs = self.omni_model_hf(
            messages,
            max_new_tokens=8196,
        )
        return outputs[0]["generated_text"][-1]

    def get_prompt(self, system_prompt, instruction):
        if self.args.omni_model_path:
            return self.huggingface_response(system_prompt, instruction)
        else:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]
            return get_response(messages, model=self.omni_model, json_format=True)

    def continue_dialogue(self, category, scenario, previous_prompt, previous_response, guidance, emotion):
        continue_instruct = reflection_continue.format(previous_prompt, previous_response, guidance, emotion)
        continue_question = self.get_prompt(Guard_system, continue_prompt_template.format(category, scenario, continue_instruct))
        try:
            prompt = json.loads(continue_question)['Question']
        except:
            prompt = eval(continue_question)['Question']
        return prompt


    def generate_prompt(self, strategy, category, scenario, system_flag=False):
        if system_flag:
            prompt_res = self.get_prompt(Guard_system, generate_prompt_template.format(strategy, category, scenario, withsystem_generate))
            prompt_res = json.loads(prompt_res.replace('Tone', 'tone').replace('emotion', 'Emotion'))
            prompt = prompt_res['Question']
            system_prompt = prompt_res['System']
        else:
            prompt_res = self.get_prompt(Guard_system, generate_prompt_template.format(strategy, category, scenario, direct_generate))
            prompt_res = json.loads(prompt_res.replace('Tone', 'tone').replace('emotion', 'Emotion'))
            prompt = prompt_res['Question']
            system_prompt = ''
        try:
            emotion_tone = prompt_res['Emotion & tone']
        except:
            emotion_tone = ''
        return system_prompt, prompt, emotion_tone

    def scenario_decision(self, category):
        if len(self.omni_model) != 0:
            messages = [{"role": "system", "content": Omni_system}, {"role": "user", "content": scenario_prompt.format(category)}]
            scenario = get_response(messages, model=self.args.omni_model, json_format=True)
            try:
                scenario = json.loads(scenario)
            except:
                scenario = eval(scenario)
            self.commonsense = scenario['common sense']
            self.current_scenario = scenario['scenario']
        else:
            self.current_scenario = self.huggingface_response(Omni_system,scenario_prompt.format(category))

    def strategy_decision(self, category, scenario):
        try:
            query = 'Safety category: ' + category + ', scenario: ' + scenario + 'Strategy:'
        except:
            query = 'Safety category: ' + category + ', scenario: ' + json.dumps(scenario) + 'Strategy:'
        strategy = self.strategy_tool.retrieval(query, 1, [], self.args.selected_ratio, self.args.max_tree_depth)
        description = self.strategy_tool.structure_tree(strategy)
        return description

    def initial_dialogue(self):
        self.scenario_decision(self.args.category)
        self.current_strategy = self.strategy_decision(self.args.category, self.current_scenario)

    def safety_judge(self, prompt, response):
        if len(self.omni_model) != 0:
            messages = [{"role":"system","content": Omni_system},{"role": "user","content": safety_judge_prompt.format(prompt, response)}]
            judge_res = get_response(messages, model=self.args.omni_model, json_format=True)
        else:
            judge_res = self.huggingface_response(Omni_system, safety_judge_prompt.format(prompt, response))
        try:
            judge_res = json.loads(judge_res)
        except:
            judge_res = eval(judge_res)
        self.history_control += 'Round ' + str(self.round_id) + ' : ' + judge_res['Summary'] + '\n'
        self.round_id += 1
        return judge_res['Safety']

    def decide_continue(self, prompt, response):
        if len(self.omni_model) != 0:
            messages = [{"role":"system","content": Omni_system},{"role": "user","content": continue_prompt.format(self.history_control, prompt, response, self.current_scenario, self.emotion)}]
            continue_decision = get_response(messages, model=self.args.omni_model, json_format=True)
            try:
                update_res = json.loads(continue_decision)
            except:
                update_res = eval(continue_decision)
            self.emotion = update_res['emotion']
            try:
                if 'none' not in update_res['scenario'].lower():
                    self.current_scenario = update_res['scenario']
            except:
                self.current_scenario = update_res['scenario']
            continue_decision = update_res['Dialogue'] + '. Reason: ' + update_res['Reason']
        else:
            continue_decision = self.huggingface_response(Omni_system, continue_prompt.format(self.history_control, prompt, response, self.current_scenario, self.emotion))
            try:
                update_res = json.loads(continue_decision)
            except:
                update_res = eval(continue_decision)
            self.emotion = update_res['emotion']
            try:
                if 'none' not in update_res['scenario'].lower():
                    self.current_scenario = update_res['scenario']
            except:
                self.current_scenario = update_res['scenario']
            continue_decision = update_res['Dialogue'] + '. Reason: ' + update_res['Reason']
        return continue_decision

    def individual_evaluation(self, need_initial=True):
        history_query = []
        history_response = []
        history_message = []
        safety_log = []
        if need_initial:
            self.initial_dialogue()
        self.round_id = 1
        self.history_control = ''

        for step in range(self.args.max_round):
            if step == 0:
                if need_initial:
                    if self.args.generate_system:
                        system_prompt, prompt, emotion = self.generate_prompt(self.current_strategy, self.args.category,
                                                                              self.current_scenario, True)
                        self.emotion = emotion
                        history_message.append({"role": "system", "content": system_prompt})
                    else:
                        system_prompt, prompt, emotion = self.generate_prompt(self.current_strategy, self.args.category,
                                                                              self.current_scenario)
                        self.emotion = emotion

                else:
                    if self.system_prompt is not None:
                        history_message.append({"role": "system", "content": self.system_prompt})
                    prompt = self.prompt_first_round
            else:
                continue_decision = self.decide_continue(history_query[-1], history_response[-1])
                if 'revision' in continue_decision.lower():
                    continue_decision = continue_decision.replace('revision', 'revision the Previous Query').replace('Revision', 'Revision the Previous Query')
                else:
                    continue_decision = continue_decision.replace('continue', 'continue the Previous Query').replace('Continue', 'Continue the Previous Query')

                prompt = self.continue_dialogue(self.args.category, self.current_scenario, history_query[-1], history_response[-1], continue_decision, self.emotion)
            history_message.append({"role": "user","content":prompt})
            llm_response = self.llm_model.generate_response(history_message)
            cur_safety = self.safety_judge(prompt, llm_response)
            safety_log.append(cur_safety)
            history_query.append(prompt)
            history_response.append(llm_response)
            history_message.append({'role': 'assistant', 'content': llm_response})
            if 'unsafe' in cur_safety.lower() or 'harmful' in cur_safety.lower():
                break
        return safety_log, history_message

    def evaluation_process(self, initial_information=None):
        if initial_information is None:
            self.individual_evaluation(need_initial=True)
        else:
            self.llm_model.session_id = None
            self.current_scenario = initial_information['scenario']
            self.current_strategy = initial_information['strategy_tree']
            self.commonsense = initial_information['common sense']
            self.prompt_first_round = initial_information['prompt']
            try:
                self.system_prompt = initial_information['system_prompt']
            except:
                self.system_prompt = None
            self.args.category = initial_information['category']
            self.emotion = ''
            safety_log, history_message = self.individual_evaluation()
            with jsonlines.open(self.result_file, 'a') as fa:
                res = {'safety': safety_log, 'dialogue': history_message}
                fa.write(res)

