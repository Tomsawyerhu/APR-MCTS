import argparse
import copy
import json
import math
import os
import random
import tqdm
import transformers

import framework
import prompt_mcts as prompt
from framework import Bug
from get_d4j_bug_list import get_defects4j_projects, get_defects4j_bugs_by_project
from llm import generate as generate_llm, generate_patches as generate_patches_llm
from gpt import generate as generate_gpt, generate_patches as generate_patches_gpt
from utils import run_bash, get_test_names, extract_method, make_failing_tests_short
import sys
from utils import *

accepted_policy_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "qwen-3b","qwen-7b", "yi-9b","llama-3b"]
model_paths = {
    "qwen-3b": "/root/autodl-tmp/Qwen2.5-Coder-3B-Instruct",
    "qwen-7b": "/mnt/data/hhc/Qwen2.5-Coder-7B-Instruct",
    "yi-9b": "/root/autodl-tmp/Yi-Coder-9B-Chat",
    "llama-3b": "/root/autodl-tmp/Llama-3.2-3B-Instruct"
}
token_statistics_file="./token_statistics.txt"

max_expansion = 3  # 最大扩展节点数量
max_rollout = 16  # 最大rollout次数
exploration_constant = 0.7  # 探索常数
mcts_inf = 1
policy_model = None
tokenizer = None
reward_model = None
existed_patches = {}
search_until_maxrollout = False
branch = 10
alpha = 0.8

output_file = ""
plausible_patch_dir = "./plausible"
if not os.path.exists(plausible_patch_dir):
    os.makedirs(plausible_patch_dir)


class Node:
    def __init__(self, bug: Bug):
        self.bug = bug
        self.num_visits = 0
        self.V = 0
        self.parent = None
        self.children = []
        self.motivation = None
        self.patches = []
        self.patch_diffs = []
        self.can_fix = False
        self.is_fully_expand = False

    def is_fully_expanded(self):
        return len(self.children) >= max_expansion or self.is_fully_expand


def get_best_child(node: Node):
    best_value = -10000
    best_nodes = []
    for child in node.children:
        node_value = child.V + exploration_constant * math.sqrt(
            2 * math.log(node.num_visits) / child.num_visits) if child.num_visits > 0 else child.V + mcts_inf
        if node_value > best_value:
            best_value = node_value
            best_nodes = [child]
        elif node_value == best_value:
            best_nodes.append(child)
    return random.choice(best_nodes)


def is_terminal(node: Node):
    return node.can_fix


def select_node(node: Node):
    while node.is_fully_expanded():
        node = get_best_child(node)
    if is_terminal(node):
        return True, node
    else:
        return False, node


def get_reward(bug: Bug, wrong_patch, reflection):
    global reward_model, tokenizer
    modes = list(bug.bug_type.split())
    mode = modes[0]
    if isinstance(policy_model, str):
        reward_prompt = prompt.construct_gpt_reward_prompt(bug=bug, wrong_patch=wrong_patch, reflection=reflection,
                                                           mode=mode, tokenizer=tokenizer)
        response = generate_gpt(reward_prompt, model_name=reward_model)
        score_list = response.split()
        score = 0
        for i in range(len(score_list)):
            score_list[i] = score_list[i].replace("**", "").replace(".", "").replace(",", "").strip()
            if score_list[i].isdigit():
                score = int(score_list[i])
                break

    else:
        reward_prompt = prompt.construct_llm_reward_prompt(bug=bug, wrong_patch=wrong_patch, reflection=reflection,
                                                           mode=mode, tokenizer=tokenizer)
        response = generate_llm(reward_model, reward_prompt)
        score_list = response.split()
        score = 0
        for i in range(len(score_list)):
            score_list[i] = score_list[i].replace("**", "").replace(".", "").replace(",", "").strip()
            if score_list[i].isdigit():
                score = int(score_list[i])
                break

    if score < 0:
        score = 0
    if score > 100:
        score = 100
    return score / 100


def extract_patch_from_response(response, mode=None):
    splitter_count = len(response.split("```java")) - 1
    if splitter_count > 2:
        patch = response[response.rfind("```java") + len("```java") + 1:]
        patch = patch[:patch.find("\n```")]
    elif splitter_count > 0:
        patch = response[response.find("```java") + len("```java") + 1:]
        patch = patch[:patch.find("\n```")]
    else:
        patch = response
    if mode == "SL":
        while len(patch) > 0 and patch.startswith("\n"):
            patch = patch[1:]
        while len(patch) > 0 and patch.endswith("\n"):
            patch = patch[:-1]
        if "\n" in patch:
            patch = patch[:patch.find("\n")]
    return patch


def extract_reflection_from_response(response):
    if "```java" not in response:
        return response[:100]
        # raise Exception("No patch found in response.")
    return response.split("```java")[0]


def expand(node: Node):
    global existed_patches, policy_model, tokenizer, reward_model
    if node.is_fully_expanded():
        return node

    bug = node.bug
    modes = list(bug.bug_type.split())
    mode = modes[0]
    print('-' * 40)
    print("Repair Prompt is:")
    if isinstance(policy_model, str):
        repair_prompt = prompt.construct_gpt_policy_prompt(bug=bug, mode=mode)
        print(repair_prompt[1]["content"] + "\n")
    else:
        repair_prompt = prompt.construct_llm_policy_prompt(bug=bug, mode=mode, tokenizer=tokenizer)
        print(repair_prompt + "\n")

    # we can expand several children at one time
    if isinstance(policy_model, str):
        responses = generate_patches_gpt(prompt=repair_prompt, num_samples=branch)
    else:
        responses = generate_patches_llm(policy_model, repair_prompt, num_samples=branch)
    existed = []
    for response in responses:
        if node.is_fully_expanded():
            # fully expanded, dicard rest
            break
        patch = extract_patch_from_response(response, mode=mode)
        if patch in existed:
            continue
        existed.append(patch)
        print('-' * 40)
        print("Repair Response is:")
        print(response + "\n")

        # check if patch exists
        # if patch in existed_patches.get((bug.project, bug.bug_id), []):
        #     return node

        # validate patch
        test_result, result_reason, patch_diff = framework.validate_patch(bug=bug, proposed_patch=patch, mode=mode)

        if test_result == "PASS":
            node.can_fix = True
            node.patches.append(patch)
            node.patch_diffs.append(patch_diff)
            if not search_until_maxrollout:
                return node

        print('-' * 40)
        print(f"Patch Validation Result:{result_reason}\n")

        next_bug_state = copy.deepcopy(bug)
        next_bug_state.test_line = ""
        next_bug_state.test_error_message = result_reason
        failing_tests = run_bash("get_failing_tests", bug.project, bug.bug_id).stdout
        test_names = get_test_names(failing_tests)
        test_code = bug.test_code
        test_methods = ""
        methods_found = []
        for test_name in test_names:
            t = extract_method(test_code, test_name)
            if t != "":
                methods_found.append(test_name)
                test_methods += t + "\n\n"
        failing_tests = make_failing_tests_short(failing_tests, methods_found)
        if test_result == "ERROR":
            next_bug_state.failing_tests = result_reason
        else:
            next_bug_state.failing_tests = failing_tests
        next_bug_state.extract_test_code = test_methods
        # 对于SL和SH类型bug, 下一个状态的bug的buggy_lines就是当前patch, 对于SF类型bug，直接更新下一个状态的bug的code为patch
        if bug.bug_type != "SF":
            next_bug_state.buggy_lines = patch
            next_bug_state.code = bug.masked_code.replace(">>> [ INFILL ] <<<", patch)
        else:
            next_bug_state.code = patch
        child = Node(next_bug_state)
        reflection = extract_reflection_from_response(response)
        child.motivation = reflection
        reward = get_reward(bug, patch, reflection)
        if test_result == "ERROR":
            reward = -1
        elif child.bug.failing_tests == node.bug.failing_tests:
            reward = 0
        child.V = reward
        print('-' * 40)
        print(f"Reward for this patch is:\n{child.V}\n")

        child.parent = node
        node.children.append(child)
    node.is_fully_expand = True
    return node


def back_propagate(node):
    while node is not None:
        node.num_visits += 1
        if node.is_fully_expanded():
            child_Vs = [child.V * child.num_visits for child in node.children]
            total_num_visits = sum([child.num_visits for child in node.children])
            if total_num_visits > 0:
                node.V = alpha * sum(child_Vs) / total_num_visits + (1 - alpha) * node.V
                print(f"Node V update to {node.V}\n")
        node = node.parent


def execute_round(root: Node, round_num):
    # execute a selection-expansion-simulation-backpropagation round
    print('-' * 40)
    print(f'Node Selecting, Round={round_num}\n')

    is_terminal_node, node = select_node(root)
    if is_terminal_node:
        print("Node is terminal, no need to expand\n")
        return False, node, root

    print("Node Selected\n")
    print(node.bug.code)

    print('-' * 40)
    print(f'Node Expanding, Round={round_num}\n')
    expand(node)

    print("Node Expanded\n")
    if node.can_fix:
        print("{} Plausible Patch Found\n".format(len(node.patches)))
        if search_until_maxrollout:
            with open(f"{plausible_patch_dir}/{node.bug.project}_{node.bug.bug_id}_plausible.jsonl", "a") as f:
                for patch in node.patches:
                    f.write(json.dumps({"patch": patch, "rollout": round_num + 1}) + "\n")
        else:
            return True, node, root

    # else:
    #     print("Motivation Found, motivation is {}\n".format(node.motivation))
    #     print("Current Patch is {}\n".format(node.bug.code))

    # skip simulating
    print('-' * 40)
    print(f'Skip Simulating, Round={round_num}\n')

    print('-' * 40)
    print(f'Backpropagating, Round={round_num}\n')
    back_propagate(node)
    return node.can_fix, node, root


def mcts_search(root: Node):
    print('-' * 40)
    print(f"Start MCTS Search for {root.bug.project}_{root.bug.bug_id}\n")
    first_plausible = None
    for i in range(max_rollout):
        flag, node, root = execute_round(root, i)
        if node.can_fix:
            if first_plausible is None:
                first_plausible = node
                with open(output_file, "a") as ff:
                    ff.write(json.dumps(
                        {"project": root.bug.project, "bug_id": root.bug.bug_id, "eval": "PASS",
                         "patch": node.patches[0],
                         "rollout": i + 1}) + "\n")
            if not search_until_maxrollout:
                return
    if first_plausible is None:
        with open(output_file, "a") as fff:
            fff.write(json.dumps({"project": root.bug.project, "bug_id": root.bug.bug_id, "eval": "FAIL", "patch": "",
                                  "rollout": max_rollout}) + "\n")


def check_bug_detail(bug_detail: Bug):
    if bug_detail.code is None or bug_detail.code == "":
        return False
    if not bug_detail.bug_type.startswith("SL") and not bug_detail.bug_type.startswith(
            "SH") and not bug_detail.bug_type.startswith(
        "SF"):
        return False
    if bug_detail.bug_type.startswith("SL") or bug_detail.bug_type.startswith("SH"):
        if not bug_detail.masked_code:
            return False
        if ">>> [ INFILL ] <<<" not in bug_detail.masked_code:
            return False
    return True


def mcts_repair():
    result_list = []
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f.readlines():
                json_line = json.loads(line)
                result_list.append((json_line["project"], str(json_line["bug_id"])))

    already_fixed = []
    # with open("./data/mcts_gpt_4o_mini_16_rollout.jsonl", 'r') as f:
    #     for line in f.readlines():
    #         json_line = json.loads(line)
    #         if json_line["eval"] == "PASS":
    #             already_fixed.append((json_line["project"], str(json_line["bug_id"])))

    projects = get_defects4j_projects()
    for proj in projects:
        bugs = get_defects4j_bugs_by_project(proj)

        for bug in tqdm.tqdm(bugs):
            # 记录一下patch名称
            write_line_to_txt(token_statistics_file,f"===================={proj}_{bug}====================")
            if (proj, str(bug)) in result_list:
                continue
            if (proj, str(bug)) in already_fixed:
                continue
            try:
                bug_detail = framework.get_bug_details(proj, bug)
            except:
                continue
            mode = bug_detail.bug_type.split()[0]
            if mode == "OT":
                continue
            if not check_bug_detail(bug_detail):
                with open(output_file, "a") as f:
                    f.write(json.dumps(
                        {"project": proj, "bug_id": bug, "eval": "FAIL", "patch": "", "rollout": max_rollout}) + "\n")
                    continue
            root = Node(bug_detail)
            mcts_search(root)


def parse_and_check_args():
    global policy_model, reward_model, max_rollout, max_expansion, exploration_constant, tokenizer, output_file
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--reward_model', type=str, default='')
    parser.add_argument('--max_rollout', type=int, default=16)
    parser.add_argument('--max_expansion', type=int, default=3)
    parser.add_argument('--exploration_constant', type=float, default=0.7)
    parser.add_argument('--logger', type=str, default="")
    parser.add_argument('--output_file', type=str, default="mcts_result.jsonl")
    args = parser.parse_args()
    _policy_model = args.policy_model
    # check policy model
    if _policy_model not in accepted_policy_models:
        raise ValueError("Policy model {} not accepted".format(_policy_model))
    # check model path
    if _policy_model in model_paths.keys():
        _policy_model_path = model_paths[_policy_model]
        if not os.path.exists(_policy_model_path):
            raise ValueError("Policy model path {} not exists".format(_policy_model_path))
    else:
        _policy_model_path = None

    _reward_model = args.reward_model
    if _reward_model is not None and _reward_model != "" and _reward_model != _policy_model:
        raise ValueError("Current only support self-evaluated mode, reward model must be equal to policy model")

    _max_rollout = args.max_rollout
    if _max_rollout not in [2, 4, 8, 16, 32]:
        raise ValueError("Max rollout must be one of [2,4,8,16,32]")

    _max_expansion = args.max_expansion
    if _max_expansion < 2:
        raise ValueError("Max expansion too small, must be at least 2")
    if _max_expansion > 3:
        raise ValueError("Max expansion too large, must be at most 3")

    _exploration_constant = args.exploration_constant
    if _exploration_constant < 0.5 or _exploration_constant >= 1.0:
        raise ValueError("Exploration constant must be in range [0.5, 1.0)")
    if branch < _max_expansion:
        raise ValueError("Branch must be larger than max expansion")
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be in range [0,1]")
    _logger = args.logger
    if _logger != "":
        # set logger file
        f = open(f'{os.getcwd()}/{_logger}', 'a')
        sys.stdout = f
        sys.stderr = f
    else:
        print("Using default std as logger")
    if not args.output_file.endswith(".jsonl"):
        raise ValueError("Output file must end with .jsonl")
    else:
        output_file = args.output_file

    # init policy model
    if _policy_model_path is None:
        # api
        policy_model = _policy_model
        print(f"Using API {_policy_model}")
    else:
        # local models
        from llm import make_model
        policy_model = make_model(_policy_model_path)
        print(f"Using local model {_policy_model}, model path {_policy_model_path}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(_policy_model_path, use_fast=False, )

    reward_model = policy_model
    max_rollout = _max_rollout
    max_expansion = _max_expansion
    exploration_constant = _exploration_constant

    print(
        f"MCTS Parameter:\n   max_rollout={max_rollout}\n   max_expansion={max_expansion}\n   exploration_constant={exploration_constant}\n    branch={branch}\n    alpha={alpha}\n")


if __name__ == '__main__':
    parse_and_check_args()
    mcts_repair()
    # f.close()
