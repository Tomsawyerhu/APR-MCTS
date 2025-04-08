import argparse
import copy
import json, jsonlines
import math
import os
import random
import tqdm
import transformers
from framework import Bug
import prompt_mcts as prompt
from condefects import *
from llm import generate as generate_llm, generate_patches as generate_patches_llm
from gpt import generate as generate_gpt, generate_patches as generate_patches_gpt
import sys

accepted_policy_models = ["gpt-4o-mini", "gpt-4o", "qwen-3b", "yi-9b", "llama-3b", "llama-8b"]
model_paths = {
    "qwen-3b": "/root/autodl-tmp/Qwen2.5-Coder-3B-Instruct",
    "yi-9b": "/root/autodl-tmp/Yi-Coder-9B-Chat",
    "llama-3b": "/root/autodl-tmp/Llama-3.2-3B-Instruct",
    "llama-8b": "/root/autodl-tmp/Llama-3.1-8B-Instruct",
}
condefects_meta = "./condefects_meta.jsonl"

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


def extract_patch_from_response(response, mode=None):
    splitter_count = len(response.split("```python")) - 1
    if splitter_count > 2:
        patch = response[response.rfind("```python") + len("```python") + 1:]
        patch = patch[:patch.find("\n```")]
    elif splitter_count > 0:
        patch = response[response.find("```python") + len("```python") + 1:]
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
        repair_prompt = prompt.construct_gpt_policy_prompt(bug=bug, mode=mode, language="python")
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

        # apply patch
        apply_patch(task_id=node.bug.project, program_id=node.bug.bug_id, patch=patch)
        print(f"=================== Apply patch =================== \n{patch}")

        # validate patch
        task_test_result = run_python_test(node.bug.project, test_list=node.bug.test_suite)

        # new bug
        next_bug_state = copy.deepcopy(bug)

        # update fail msg
        if task_test_result == "timeout":
            next_bug_state.test_output = "timeout"
            next_bug_state.failing_tests = "timeout happens"
            program_test_result = None
        else:
            program_test_result = task_test_result[node.bug.bug_id]
            if False not in program_test_result["is_test_passed"]:
                node.can_fix = True
                node.patches.append(patch)
                if not search_until_maxrollout:
                    return node
            next_bug_state.test_output = task_test_result[bug.bug_id]['test_results']
            is_passed_result = task_test_result[bug.bug_id]['is_test_passed']
            next_bug_state.failing_tests = format_test_failure_info(next_bug_state.test_input,
                                                                    next_bug_state.test_output,
                                                                    next_bug_state.expected_output,
                                                                    is_passed_result)

        # 对于SL和SH类型bug, 下一个状态的bug的buggy_lines就是当前patch, 对于SF类型bug，直接更新下一个状态的bug的code为patch
        next_bug_state.code = patch
        child = Node(next_bug_state)

        # reward是测试通过率
        if task_test_result == "timeout":
            reward = 0
        else:
            reward = program_test_result["is_test_passed"].count("True") / len(program_test_result["is_test_passed"])
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


def read_line(bug_code: str, line_num):
    all_lines = bug_code.split("\n")
    return all_lines[line_num - 1]


def mask_fill(code, line_num):
    all_lines = code.split("\n")
    all_lines[line_num - 1] = ">>> [ INFILL ] <<<"
    return '\n'.join(all_lines)


def format_test_failure_info(test_input, test_output, expected_output, is_passed_result, limit=100000):
    failure_info = ""
    for i, is_passed in enumerate(is_passed_result):
        if not is_passed:
            failure_info += (f"================== failed test {i} ==================\n"
                             f"input:\n"
                             f"{str(test_input[i])}\n"
                             f"expected output:\n"
                             f"{str(expected_output[i])}\n"
                             f"buggy output:\n"
                             f"{str(test_output[i])}\n")
    # truncate
    if len(failure_info) > limit:
        failure_info = failure_info[:limit] + "......"
    return failure_info


def mcts_repair():
    # read condefects meta
    with jsonlines.open(condefects_meta, 'r') as reader:
        for line in reader:
            buggy_code, correct_code, bug_location = read_python_program_code(line['task_id'], line['program_id'])
            if len(bug_location) > 1:
                #只修复单行缺陷,但是修复模式使用SF
                continue
            if line["time"] > 3:
                # 测试时间>3s的先不修复
                continue
            checkout_python_task(line['task_id'])
            bug_type = "SF"
            bug = Bug(test_framework="condefects",
                      project=line['task_id'],
                      bug_id=line['program_id'],
                      bug_type=bug_type,
                      code=buggy_code,
                      fixed_code=correct_code,
                      masked_code=mask_fill(buggy_code, bug_location[0]),
                      buggy_lines=read_line(buggy_code, bug_location[0]),
                      fixed_lines="",
                      test_code="",
                      extract_test_code="",
                      test_suite=line["test_list"],
                      test_name="",
                      test_line="",
                      failing_tests="",
                      test_error_message=""
                      )
            # condefects 新加的字段
            bug.test_input = [get_python_test_input(line['task_id'], x) for x in line["test_list"]]
            test_result = run_python_test(line['task_id'], test_list=line["test_list"])
            if test_result == "timeout":
                continue
            bug.test_output = test_result[line['program_id']]['test_results']
            bug.expected_output = test_result[line['program_id']]['correct_results']
            is_passed_result = line['test_result']
            bug.failing_tests = format_test_failure_info(bug.test_input, bug.test_output, bug.expected_output,
                                                         is_passed_result)
            root = Node(bug)
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
    parser.add_argument('--output_file', type=str, default="condefects_mcts_result.jsonl")
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
