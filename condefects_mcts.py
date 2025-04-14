import argparse
import copy
import json, jsonlines
import math
import os
import random
# import transformers
from framework import Bug
import prompt_mcts as prompt
from condefects import *
# from llm import generate as generate_llm, generate_patches as generate_patches_llm
from gpt import generate as generate_gpt, generate_patches as generate_patches_gpt
import sys

accepted_policy_models = ["gpt-4o-mini", "gpt-4o", "qwen-3b", "yi-9b", "llama-3b", "llama-8b"]
model_paths = {
    "qwen-3b": "/root/autodl-tmp/Qwen2.5-Coder-3B-Instruct",
    "yi-9b": "/root/autodl-tmp/Yi-Coder-9B-Chat",
    "llama-3b": "/root/autodl-tmp/Llama-3.2-3B-Instruct",
    "llama-8b": "/root/autodl-tmp/Llama-3.1-8B-Instruct",
}
condefects_meta = "./condefects_meta_old_with_date.jsonl"
specific_task_id = None  # 修复单个缺陷
specific_program_id = None  # 修复单个程序

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

mask_mode = False
output_file = ""

black_list = ["abc234_h", "abc319_b"]
white_list = ['abc223_a', 'abc224_a', 'abc225_a', 'abc226_a', 'abc226_b', 'abc229_a', 'abc229_b', 'abc230_a', 'abc230_b', 'abc230_e', 'abc232_b', 'abc232_d', 'abc233_a', 'abc233_b', 'abc233_c', 'abc233_e', 'abc234_a', 'abc234_b', 'abc234_c', 'abc234_e', 'abc235_b', 'abc235_c', 'abc236_a', 'abc236_b', 'abc237_a', 'abc238_b', 'abc238_c', 'abc239_a', 'abc239_g', 'abc240_a', 'abc240_d', 'abc241_a', 'abc241_b', 'abc241_e', 'abc242_a', 'abc242_b', 'abc243_b', 'abc244_b', 'abc244_c', 'abc244_d', 'abc245_a', 'abc245_b', 'abc246_a', 'abc247_a', 'abc247_b', 'abc247_c', 'abc247_d', 'abc247_e', 'abc248_a', 'abc248_c', 'abc249_a', 'abc249_b', 'abc249_f', 'abc250_b', 'abc251_a', 'abc251_c', 'abc251_d', 'abc252_a', 'abc252_b', 'abc252_c', 'abc253_a', 'abc253_b', 'abc253_d', 'abc254_a', 'abc254_c', 'abc255_a', 'abc255_b', 'abc256_b', 'abc257_a', 'abc257_b', 'abc257_c', 'abc257_e', 'abc258_a', 'abc258_b', 'abc259_a', 'abc259_b', 'abc260_a', 'abc260_e', 'abc261_a', 'abc262_a', 'abc262_c', 'abc262_d', 'abc263_a', 'abc263_b', 'abc263_c', 'abc264_a', 'abc264_b', 'abc265_a', 'abc265_c', 'abc266_b', 'abc266_c', 'abc266_e', 'abc267_a', 'abc267_b', 'abc268_b', 'abc268_c', 'abc269_a', 'abc269_b', 'abc269_c', 'abc269_e', 'abc270_a', 'abc270_b', 'abc271_a', 'abc271_c', 'abc271_d', 'abc271_e', 'abc272_c', 'abc273_b', 'abc274_a', 'abc275_a', 'abc275_b', 'abc275_c', 'abc275_d', 'abc276_a', 'abc276_b', 'abc276_d', 'abc277_a', 'abc277_b', 'abc278_a', 'abc278_b', 'abc278_d', 'abc279_a', 'abc279_b', 'abc279_d', 'abc279_e', 'abc280_b', 'abc280_c', 'abc280_e', 'abc281_a', 'abc281_b', 'abc282_a', 'abc282_c', 'abc282_e', 'abc282_f', 'abc283_a', 'abc283_c', 'abc283_d', 'abc284_d', 'abc285_a', 'abc285_c', 'abc286_a', 'abc286_f', 'abc287_a', 'abc288_b', 'abc289_a', 'abc289_b', 'abc289_c', 'abc290_b', 'abc290_d', 'abc291_a', 'abc291_b', 'abc292_b', 'abc292_f', 'abc293_b', 'abc293_c', 'abc293_e', 'abc293_f', 'abc294_a', 'abc294_b', 'abc295_a', 'abc295_b', 'abc296_b', 'abc296_c', 'abc297_a', 'abc297_b', 'abc297_c', 'abc297_d', 'abc298_a', 'abc298_b', 'abc298_e', 'abc299_a', 'abc299_d', 'abc299_e', 'abc300_a', 'abc300_b', 'abc300_c', 'abc300_e', 'abc301_a', 'abc301_b', 'abc301_d', 'abc302_a', 'abc302_b', 'abc303_a', 'abc303_b', 'abc304_a', 'abc304_b', 'abc304_f', 'abc305_a', 'abc305_b', 'abc306_b', 'abc306_e', 'abc307_a', 'abc307_b', 'abc307_c', 'abc307_d', 'abc307_e', 'abc308_a', 'abc308_b', 'abc309_a', 'abc309_b', 'abc310_a', 'abc310_b', 'abc310_c', 'abc311_a', 'abc311_b', 'abc312_a', 'abc312_b', 'abc313_a', 'abc313_b', 'abc313_c', 'abc313_d', 'abc313_e', 'abc313_f', 'abc314_a', 'abc314_b', 'abc314_e', 'abc315_a', 'abc315_b', 'abc315_h', 'abc318_a', 'abc318_c', 'abc319_a', 'abc319_b', 'agc058_a', 'agc060_b', 'agc061_a', 'agc063_a', 'agc064_a', 'arc128_b', 'arc129_a', 'arc129_c', 'arc131_a', 'arc131_c', 'arc133_a', 'arc133_d', 'arc134_a', 'arc135_a', 'arc136_a', 'arc136_c', 'arc137_a', 'arc137_b', 'arc138_b', 'arc140_e', 'arc141_b', 'arc142_a', 'arc142_b', 'arc142_c', 'arc143_a', 'arc143_c', 'arc144_a', 'arc145_a', 'arc145_b', 'arc146_a', 'arc146_c', 'arc147_b', 'arc149_a', 'arc150_e', 'arc152_a', 'arc152_d', 'arc153_a', 'arc155_a', 'arc156_a', 'arc156_c', 'arc156_d', 'arc157_a', 'arc159_a', 'arc159_b', 'arc161_a', 'arc162_b', 'arc162_c', 'arc162_e', 'arc163_a', 'arc163_c', 'arc164_c', 'arc165_a', 'abc224_b', 'abc224_c', 'abc224_d', 'abc225_d', 'abc229_e', 'abc229_f', 'abc232_e', 'abc235_e', 'abc236_e', 'abc240_e', 'abc241_d', 'abc241_g', 'abc242_c', 'abc242_d', 'abc242_e', 'abc244_e', 'abc245_f', 'abc248_d', 'abc248_e', 'abc249_d', 'abc250_d', 'abc250_e', 'abc251_e', 'abc252_e', 'abc253_f', 'abc254_e', 'abc255_e', 'abc258_c', 'abc259_e', 'abc260_f', 'abc261_e', 'abc262_e', 'abc263_e', 'abc264_e', 'abc266_d', 'abc266_g', 'abc267_f', 'abc268_e', 'abc274_d', 'abc277_e', 'abc278_e', 'abc281_f', 'abc283_e', 'abc286_d', 'abc287_c', 'abc289_e', 'abc291_e', 'abc292_c', 'abc292_d', 'abc297_e', 'abc298_c', 'abc298_d', 'abc300_d', 'abc302_f', 'abc304_d', 'abc305_e', 'abc306_g', 'abc308_c', 'abc309_d', 'abc311_e', 'abc312_f', 'abc313_g', 'abc314_d', 'abc315_c', 'abc319_c', 'abc319_f', 'agc055_a', 'agc063_c', 'abc225_e', 'abc238_g', 'abc241_c', 'abc261_d', 'abc264_c', 'abc267_e', 'abc274_e', 'abc275_f', 'abc284_f', 'abc300_g', 'abc301_e', 'abc304_c', 'abc319_g', 'abc223_d', 'abc223_f', 'abc225_c', 'abc226_c', 'abc229_d', 'abc230_d', 'abc232_c', 'abc235_d', 'abc237_c', 'abc238_a', 'abc238_d', 'abc239_c', 'abc243_c', 'abc243_d', 'abc246_d', 'abc246_e', 'abc249_c', 'abc250_c', 'abc251_b', 'abc252_d', 'abc253_e', 'abc255_c', 'abc256_d', 'abc257_d', 'abc257_f', 'abc257_g', 'abc258_d', 'abc258_e', 'abc259_c', 'abc261_b', 'abc263_d', 'abc265_b', 'abc265_d', 'abc267_c', 'abc267_d', 'abc268_d', 'abc270_d', 'abc272_d', 'abc272_e', 'abc273_d', 'abc277_d', 'abc278_c', 'abc280_d', 'abc281_d', 'abc286_e', 'abc287_d', 'abc288_d', 'abc289_d', 'abc290_c', 'abc291_c', 'abc291_d', 'abc291_f', 'abc296_d', 'abc299_c', 'abc300_f', 'abc302_c', 'abc303_c', 'abc303_d', 'abc309_c', 'abc309_e', 'abc312_c', 'abc315_f', 'abc318_e', 'abc319_d', 'agc056_c', 'agc057_a', 'agc057_b', 'agc058_b', 'agc059_a', 'agc059_b', 'agc060_a', 'agc062_a', 'agc062_c', 'agc063_b', 'arc128_c', 'arc128_d', 'arc128_e', 'arc129_b', 'arc129_d', 'arc130_b', 'arc130_f', 'arc131_b', 'arc131_d', 'arc131_e', 'arc132_a', 'arc132_b', 'arc132_c', 'arc132_d', 'arc133_b', 'arc134_b', 'arc134_d', 'arc135_b', 'arc135_c', 'arc135_d', 'arc136_b', 'arc136_d', 'arc137_c', 'arc137_d', 'arc138_a', 'arc138_c', 'arc138_d', 'arc139_a', 'arc139_b', 'arc139_c', 'arc139_d', 'arc140_a', 'arc140_b', 'arc140_c', 'arc140_d', 'arc141_a', 'arc141_c', 'arc141_e', 'arc143_d', 'arc144_b', 'arc145_c', 'arc145_d', 'arc145_e', 'arc146_b', 'arc146_d', 'arc147_c', 'arc147_e', 'arc148_a', 'arc148_b', 'arc148_c', 'arc148_d', 'arc148_e', 'arc149_c', 'arc149_d', 'arc150_a', 'arc150_b', 'arc151_a', 'arc151_b', 'arc151_c', 'arc151_e', 'arc152_b', 'arc152_c', 'arc153_b', 'arc153_c', 'arc154_b', 'arc154_c', 'arc154_d', 'arc155_b', 'arc155_c', 'arc156_b', 'arc157_b', 'arc157_d', 'arc157_e', 'arc158_a', 'arc158_b', 'arc158_c', 'arc158_d', 'arc159_c', 'arc159_d', 'arc160_a', 'arc160_b', 'arc160_c', 'arc161_b', 'arc161_c', 'arc161_d', 'arc163_b', 'arc164_a', 'arc164_d', 'arc164_e', 'arc165_b', 'arc165_c', 'arc165_d']

genetic_config = {
    "pool_size": 10,  # 保留10个
    "generation": 5,  # 每轮新生成5个
    "iter": 2,  # 2轮
}
patch_pool = {}

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
        # responses = generate_patches_llm(policy_model, repair_prompt, num_samples=branch)
        pass
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
        # for SF bugs, directly apply patch
        # for SL and SH bugs, replace placeholder and apply patch
        if node.bug.bug_type == "SF":
            apply_patch(task_id=node.bug.project, program_id=node.bug.bug_id, patch=patch)
        else:
            apply_patch(task_id=node.bug.project, program_id=node.bug.bug_id,
                        patch=node.bug.masked_code.replace(">>> [ INFILL ] <<<", patch))
        print(f"=================== Apply patch =================== \n{patch}")

        # validate patch
        task_test_result = run_python_test(node.bug.project, node.bug.bug_id, test_list=node.bug.test_suite)

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
        if bug.bug_type != "SF":
            next_bug_state.buggy_lines = patch
            next_bug_state.code = bug.masked_code.replace(">>> [ INFILL ] <<<", patch)
        else:
            next_bug_state.code = patch

        child = Node(next_bug_state)

        # reward是测试通过率
        if task_test_result == "timeout":
            reward = 0
        else:
            reward = program_test_result["is_test_passed"].count(True) / len(program_test_result["is_test_passed"])
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


def format_test_failure_info(test_input, test_output, expected_output, is_passed_result, limit=100000,
                             input_output_limit=1000):
    """

    :param test_input:
    :param test_output:
    :param expected_output:
    :param is_passed_result:
    :param limit: 总长度限制
    :param input_output_limit: 单个测试用例输入输出长度限制
    :return:
    """
    failure_info = ""
    for i, is_passed in enumerate(is_passed_result):
        if not is_passed and len(test_input[i]) <= input_output_limit and len(test_output[i]) <= input_output_limit:
            failure_info += (f"================== failed test {i} ==================\n"
                             f"input:\n"
                             f"{str(test_input[i])}\n"
                             f"expected output:\n"
                             f"'{str(expected_output[i])}'\n"
                             f"buggy output:\n"
                             f"'{str(test_output[i])}'\n")
    # truncate
    if len(failure_info) > limit:
        failure_info = failure_info[:limit] + "......"
    return failure_info


# def genetic_algorithm():


def mcts_repair():
    existed_ids = []
    if os.path.exists(output_file):
        with jsonlines.open(output_file, 'r') as f:
            for line in f:
                existed_ids.append((line["project"], line['bug_id']))
    # read condefects meta
    with jsonlines.open(condefects_meta, 'r') as reader:
        for line in reader:
            buggy_code, correct_code, bug_location = read_python_program_code(line['task_id'], line['program_id'])
            if (line['task_id'], line['program_id']) in existed_ids:
                continue
            # 白名单模式
            if line['task_id'] not in white_list:
                continue
            # # 黑名单模式
            # if line['task_id'] in black_list:
            #     continue
            # 单个缺陷修复模式
            if specific_task_id is not None and line['task_id'] != specific_task_id:
                continue
            # 单个程序模式
            if specific_program_id is not None and str(line['program_id'])!=specific_program_id:
                continue

            #是否使用masked mode
            if mask_mode:
                masked_code, buggy_lines, fix_lines = get_mask_code(correct_code, buggy_code)
                if "\n" in fix_lines:
                    # multi fix lines
                    bug_type = "SH"
                else:
                    # single fix lines
                    bug_type = "SL"
            else:
                masked_code = ""
                buggy_lines = ""
                fix_lines = ""
                bug_type = "SF"
            # if len(bug_location) > 1:
            #     #只修复单行缺陷,但是修复模式使用SF
            #     continue
#            if line["time"] > 10:
#                # 测试时间>10s的先不修复
#                continue
            checkout_python_task(line['task_id'])

            bug = Bug(test_framework="condefects",
                      project=line['task_id'],
                      bug_id=line['program_id'],
                      bug_type=bug_type,
                      code=buggy_code,
                      fixed_code=correct_code,
                      masked_code=masked_code,
                      buggy_lines=buggy_lines,
                      fixed_lines=fix_lines,
                      test_code="",
                      extract_test_code="",
                      test_suite=line["test_list"],
                      test_name="",
                      test_line="",
                      failing_tests="",
                      test_error_message=""
                      )
            # masked_code=mask_fill(buggy_code, bug_location[0])
            # buggy_lines = read_line(buggy_code, bug_location[0])
            # condefects 新加的字段
            try:
                bug.test_input = [get_python_test_input(line['task_id'], x) for x in line["test_list"]]
                test_result = run_python_test(line['task_id'], line['program_id'], test_list=line["test_list"])
                if test_result == "timeout":
                    continue
                bug.test_output = test_result[line['program_id']]['test_results']
                bug.expected_output = test_result[line['program_id']]['correct_results']
                print(test_result)
                is_passed_result = line['test_result']
                bug.failing_tests = format_test_failure_info(bug.test_input, bug.test_output, bug.expected_output,
                                                             is_passed_result)

                root = Node(bug)
                mcts_search(root)
            except Exception as e:
#                raise (e)
                continue


def parse_and_check_args():
    global policy_model, reward_model, max_rollout, max_expansion, exploration_constant, tokenizer, output_file, specific_task_id, mask_mode
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--reward_model', type=str, default='')
    parser.add_argument('--max_rollout', type=int, default=16)
    parser.add_argument('--max_expansion', type=int, default=3)
    parser.add_argument('--exploration_constant', type=float, default=0.7)
    parser.add_argument('--logger', type=str, default="")
    parser.add_argument('--output_file', type=str, default="condefects_mcts_result.jsonl")
    parser.add_argument('--task_id', type=str, default=None)
    parser.add_argument('--program_id', type=int, default=None)
    parser.add_argument('--mask_mode', type=str, default="unmasked")
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
    # if branch < _max_expansion:
    #     raise ValueError("Branch must be larger than max expansion")
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

    specific_task_id = args.task_id
    if args.mask_mode == "masked":
        mask_mode = True
        print("use mask mode")
    elif args.mask_mode == "unmasked":
        mask_mode = False
        print("use unmask mode")
    else:
        print("unknown mask mode,set to unmasked")
        mask_mode = False

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
        # tokenizer = transformers.AutoTokenizer.from_pretrained(_policy_model_path, use_fast=False, )

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
