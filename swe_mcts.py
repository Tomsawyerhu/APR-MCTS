import argparse
import json
import math
import os.path
import random
import subprocess

import jsonlines

from swe_repair import AGENTLESS_REFINE_PROMPT, generate_with_retries, AGENTLESS_PROMPT, evaluate_patch

max_expansion = 3  # 最大扩展节点数量
max_rollout = 16  # 最大rollout次数
exploration_constant = 0.7  # 探索常数
mcts_inf = 1
policy_model = ''
reward_model = ''
existed_patches = {}
branch = 10
alpha = 0.8
temperature = 1.0
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key = "sk-e346078d76f546c2ab04f0f008126a91"
instance_id = None
output_file = ""
input_file = ""
correct_patch_file = ""
swe_root = ""
log_file = ""


class Node:
    def __init__(self,
                 instance_id,
                 problem_statement,
                 buggy_files: list,
                 buggy_file_contents: list,
                 buggy_code: dict,
                 patch_diff: str,
                 test_report: str,
                 repair_model: str,
                 evaluate_model: str,
                 score=0,
                 can_fix=False
                 ):
        self.instance_id = instance_id
        self.problem_statement = problem_statement
        self.buggy_files = buggy_files
        self.buggy_file_contents = buggy_file_contents
        self.buggy_code = buggy_code
        self.num_visits = 0
        self.patch_diff = patch_diff
        self.test_report = test_report
        self.repair_model = repair_model
        self.evaluate_model = evaluate_model
        self.V = score
        self.parent = None
        self.children = []
        self.can_fix = can_fix
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


def expand(node: Node):
    global existed_patches, policy_model, reward_model
    if node.is_fully_expanded():
        return node

    problem_statement = node.problem_statement
    instance_id = node.instance_id
    buggy_locs = node.buggy_code
    found_files = node.buggy_files
    file_contents = node.buggy_file_contents

    if node.parent is None and len(node.children)==0:
        # 根节点,执行初始化
        prompt = AGENTLESS_PROMPT.format(problem_statement=problem_statement,
                                         retrieval=json.dumps(buggy_locs, indent=4),
                                         )
        generate_times = branch

    else:
        prompt = AGENTLESS_REFINE_PROMPT.format(problem_statement=problem_statement,
                                                retrieval=json.dumps(buggy_locs, indent=4),
                                                partial_patch=node.patch_diff,
                                                test_report=node.test_report
                                                )
        generate_times = 1

    child_node = None

    for i in range(generate_times):
        if node.is_fully_expanded():
            break

        with open(log_file, 'a') as log_writer:
            log_writer.write('=' * 80)
            log_writer.write('\n')
            log_writer.write(f'PROMPT ({instance_id})')
            log_writer.write('\n')
            log_writer.write('=' * 80)
            log_writer.write('\n')
            log_writer.write(prompt)
            log_writer.write('\n')
        patch = generate_with_retries(instance_id,
                                      prompt,
                                      output_file,
                                      file_contents=file_contents,
                                      found_files=found_files,
                                      model_name=policy_model,
                                      base_url=base_url,
                                      api_key=api_key,
                                      temperature=((i+1)/generate_times)*temperature,
                                      log_file=log_file)

        existed_patches = [x.patch_diff for x in node.children]

        if not patch:
            continue

        if patch in existed_patches:
            continue

        subprocess.run(f'cd {swe_root} && rm -rf logs', check=True, shell=True)

        subprocess.run(f'rm -rf ./patch.json', check=True, shell=True)

        with open('./patch.jsonl', 'w') as f:
            f.write(json.dumps({
                'instance_id': instance_id,
                'model_name_or_path': policy_model,
                'model_patch': patch
            }))
            f.write('\n')

        # evaluate the patch here
        score, is_passed, test_report = evaluate_patch(
            node.instance_id,
            problem_statement,
            patch,
            './patch.jsonl',
            node.repair_model,
            node.evaluate_model,
            base_url,
            api_key,
            'swe_mcts',
            swe_root,
            log_file=log_file
        )

        with open(log_file, 'a') as log_writer:
            log_writer.write('=' * 80)
            log_writer.write('\n')
            log_writer.write(f'Test Report')
            log_writer.write('\n')
            log_writer.write('=' * 80)
            log_writer.write('\n')
            log_writer.write(test_report)
            log_writer.write('\n')

        subprocess.run(f'rm -rf ./patch.jsonl', check=True, shell=True)

        subprocess.run(f'cd {swe_root} && rm -rf logs', check=True, shell=True)

        child_node = Node(instance_id=node.instance_id,
                          problem_statement=node.problem_statement,
                          buggy_files=node.buggy_files,
                          buggy_file_contents=node.buggy_files,
                          buggy_code=node.buggy_code,
                          patch_diff=patch,
                          test_report=test_report,
                          score=score,
                          can_fix=is_passed,
                          repair_model=node.repair_model,
                          evaluate_model=node.evaluate_model
                          )
        child_node.parent = node
        node.children.append(child_node)

        if child_node.can_fix:
            return child_node

    return child_node


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
    print(node.patch_diff)

    print('-' * 40)
    print(f'Node Expanding, Round={round_num}\n')
    one_child_node = expand(node)
    if one_child_node is None:
        return False,None,root

    print("Node Expanded\n")
    if one_child_node.can_fix:
        return True, one_child_node, root

    # skip simulating
    print('-' * 40)
    print(f'Skip Simulating, Round={round_num}\n')

    print('-' * 40)
    print(f'Backpropagating, Round={round_num}\n')
    back_propagate(node)
    return one_child_node.can_fix, one_child_node, root


def mcts_search(root: Node):
    for i in range(max_rollout):
        flag, node, root = execute_round(root, i)
        if flag:
            with jsonlines.open(correct_patch_file, 'a') as f:
                f.write({
                    'instance_id': node.instance_id,
                    'patch': node.patch_diff
                })
            break


def mcts_repair():
    existed_instance_ids = set()
    if os.path.exists(output_file):
        with jsonlines.open(output_file, 'r') as f:
            for line in f:
                existed_instance_ids.add(line['instance_id'])
    with jsonlines.open(input_file, 'r') as reader:
        for line in reader:
            if instance_id is not None and instance_id != '' and instance_id != line['instance_id']:
                continue
            if line['instance_id'] in existed_instance_ids:
                continue
            buggy_locs = line['buggy_code']
            found_files = list(line['buggy_files'].keys())
            file_contents = list(line['buggy_files'].values())
            bug_node = Node(
                instance_id=line['instance_id'],
                problem_statement=line['problem_statement'],
                buggy_files=found_files,
                buggy_file_contents=file_contents,
                buggy_code=buggy_locs,
                patch_diff='',
                test_report='',
                repair_model=policy_model,
                evaluate_model=reward_model
            )

            mcts_search(bug_node)


def parse_and_check_args():
    global input_file, policy_model, reward_model, max_rollout, max_expansion, exploration_constant, output_file, instance_id, swe_root, correct_patch_file, log_file
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_model', type=str, default='qwen3-coder-plus')
    parser.add_argument('--reward_model', type=str, default='qwen3-coder-plus')
    parser.add_argument('--max_rollout', type=int, default=16, choices=[2, 4, 8, 16, 32])
    parser.add_argument('--max_expansion', type=int, default=3, choices=[2, 3, 4, 5])
    parser.add_argument('--exploration_constant', type=float, default=0.7)
    parser.add_argument('--input_file', type=str, default="./swe_lite_loc_with_context.jsonl")
    parser.add_argument('--output_file', type=str, default="./swe_lite_patch.jsonl")
    parser.add_argument('--correct_patch_file', type=str, default="./swe_lite_correct_patch.jsonl")
    parser.add_argument('--instance_id', type=str, default=None)
    parser.add_argument('--swe_root', type=str, default='.')
    parser.add_argument('--log_file', type=str, default='./log.txt')
    args = parser.parse_args()
    policy_model = args.policy_model
    reward_model = args.reward_model
    max_rollout = args.max_rollout
    max_expansion = args.max_expansion
    exploration_constant = args.exploration_constant
    output_file = args.output_file
    input_file = args.input_file
    correct_patch_file = args.correct_patch_file
    instance_id = args.instance_id
    swe_root = args.swe_root
    log_file = args.log_file

    print(
        f"MCTS Parameter:\n   max_rollout={max_rollout}\n   max_expansion={max_expansion}\n   exploration_constant={exploration_constant}\n    branch={branch}\n    alpha={alpha}\n")


if __name__ == '__main__':
    parse_and_check_args()
    mcts_repair()
