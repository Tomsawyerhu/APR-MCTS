import argparse
import copy
import dataclasses
import json
import os

import jsonlines
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import BeamSearchParams, SamplingParams

from get_d4j_bug_list import get_defects4j_projects, get_defects4j_bugs_by_project
from gpt import generate as generate_gpt
import framework
from framework import Bug
import prompt_mcts as prompt
from llm import make_model
from utils import utc_now_str, run_bash, get_test_names, extract_method, make_failing_tests_short
import tqdm
# python beam_search.py --policy_model /mnt/data/hhc/Qwen2.5-Coder-7B-Instruct --reward_model qwen-max-2025-01-25

class Config:
    beam_width = 5
    max_tokens = 1024
    max_iter = 4
    pool_size = 4
    generate_once = 2
    init_pool_temperature = 0.9
    after_init_temperature = 0.4
    policy_model = ''
    reward_model = ''
    plausible_save_path = ''
    output_path = ''


@dataclasses.dataclass
class Patch:
    bug: Bug
    patch_code: str
    patch_diff: str
    reflection: str
    hash_id: int
    score: int
    is_plausible: bool
    gmt_created: str


class PatchPool:
    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.patch_pool = []
        self.plausible_patches = []
        self.all_patches = []

    def add_patch(self, patch: Patch):
        self.all_patches.append(patch)
        # 可信补丁单独存放
        if patch.is_plausible:
            self.plausible_patches.append(patch)
            return

        # 1) 先放入候选池
        self.patch_pool.append(patch)

        # 2) 如果超过容量，按 score 升序、时间早→晚 排序并丢弃最差的一个
        if len(self.patch_pool) > self.pool_size:
            # ISO-8601 或类似格式的时间字符串在字典序上就是先后顺序
            self.patch_pool.sort(key=lambda p: (p.score, p.gmt_created))
            # 移除列表第 0 个（最低分且最早的）
            self.patch_pool.pop(0)

    def is_patch_exist(self, p: Patch):
        for patch in self.all_patches:
            if patch.patch_code.strip() == p.patch_code:
                return True
        return False

    def is_patch_error_exist(self, p: Patch):
        for patch in self.all_patches:
            if patch.bug.test_error_message.strip() == p.bug.test_error_message.strip():
                return True
        return False


def extract_reflection_from_response(response):
    if "```java" not in response:
        return response[:100]
        # raise Exception("No patch found in response.")
    return response.split("```java")[0]


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


def get_reward(bug: Bug, patch: Patch, patch_pool: PatchPool):
    modes = list(bug.bug_type.split())
    mode = modes[0]

    # 如果出现了编译错误
    if 'cannot find symbol' in patch.bug.test_error_message:
        return -100
    if '\';\' expected' in patch.bug.test_error_message:
        return -100
    if '\')\' expected' in patch.bug.test_error_message:
        return -100
    if 'illegal' in patch.bug.test_error_message:
        return -100

    # 如果出现了一样的patch
    if patch_pool.is_patch_exist(patch):
        return 0
    # 如果出现了一样的error
    if patch_pool.is_patch_error_exist(patch):
        return 0


    reward_prompt = prompt.construct_gpt_reward_prompt(bug=bug, wrong_patch=patch.patch_code,
                                                       reflection=patch.reflection,
                                                       mode=mode, tokenizer=None)
    response = generate_gpt(reward_prompt, Config.reward_model)
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

    return score


def execute_round(bug: Bug, patch_pool: PatchPool, model: LLM, tokenizer: AutoTokenizer, round_num: int):
    if round_num > Config.max_iter:
        raise Exception('exceed max iter')

    modes = list(bug.bug_type.split())
    mode = modes[0]
    generated_patches = []

    # 如果patch pool为空，采样pool size个补丁初始化补丁池
    if len(patch_pool.all_patches) == 0:
        repair_prompt = prompt.construct_llm_policy_prompt(bug=bug, mode=mode, tokenizer=tokenizer)
        print(repair_prompt + "\n")

        sampling_params = SamplingParams(
            temperature=Config.init_pool_temperature,
            max_tokens=Config.max_tokens,
            frequency_penalty=1.0,
            presence_penalty=1.0,
            n=Config.pool_size*2,
        )

        responses = model.generate(
            repair_prompt,
            sampling_params
        )

        responses = [x.text for x in responses[0].outputs]
        patch_code_list = [extract_patch_from_response(x, mode) for x in responses]
        reflection_list = [extract_reflection_from_response(x) for x in responses]

        for patch_code, reflection in zip(patch_code_list, reflection_list):
            generated_patches.append(Patch(
                bug=bug,
                patch_code=patch_code,
                patch_diff='',
                reflection=reflection,
                hash_id=hash(patch_code),
                score=-1,
                is_plausible=False,
                gmt_created=utc_now_str()
            ))
    else:
        for patch in patch_pool.patch_pool:
            repair_prompt = prompt.construct_llm_policy_prompt(bug=patch.bug, mode=mode, tokenizer=tokenizer)
            print(repair_prompt + "\n")

            params = BeamSearchParams(beam_width=Config.beam_width,
                                      max_tokens=Config.max_tokens
                                      )

            responses = model.beam_search([{'prompt':repair_prompt}], params)
            responses = [responses[0].sequences[0].text]

            # sampling_params = SamplingParams(
            #     temperature=Config.after_init_temperature,
            #     max_tokens=Config.max_tokens,
            #     frequency_penalty=1.0,
            #     presence_penalty=1.0,
            #     # top_p=1.0,
            #     # top_k=10,
            #     n=Config.generate_once,
            # )
            #
            # responses = model.generate(
            #     repair_prompt,
            #     sampling_params
            # )
            #
            # responses = [x.text for x in responses[0].outputs]
            for response in responses:
                print('-' * 40)
                print("Repair Response is:")
                print(response + "\n")

                new_patch = extract_patch_from_response(response, mode=mode)
                generated_patches.append(Patch(
                    bug=patch.bug,
                    patch_code=new_patch,
                    patch_diff='',
                    reflection=extract_reflection_from_response(response),
                    hash_id=hash(new_patch),
                    score=-1,
                    is_plausible=False,
                    gmt_created=utc_now_str()
                ))

    for patch in generated_patches:
        # validate patch
        test_result, result_reason, patch_diff = framework.validate_patch(bug=patch.bug,
                                                                          proposed_patch=patch.patch_code,
                                                                          mode=mode)
        print('-' * 40)
        print(f"Patch Validation Result:{result_reason}\n")

        patch.patch_diff = patch_diff
        if test_result == "PASS":
            patch.is_plausible = True
            patch.score = 100
        else:
            old_bug = patch.bug
            next_bug_state = copy.deepcopy(patch.bug)
            next_bug_state.test_line = ""
            next_bug_state.test_error_message = result_reason
            failing_tests = run_bash("get_failing_tests", patch.bug.project, patch.bug.bug_id).stdout
            test_names = get_test_names(failing_tests)
            test_code = patch.bug.test_code
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
            if patch.bug.bug_type != "SF":
                next_bug_state.buggy_lines = patch.patch_code
                next_bug_state.code = patch.bug.masked_code.replace(">>> [ INFILL ] <<<", patch.patch_code)
            else:
                next_bug_state.code = patch.patch_code

            patch.bug = next_bug_state
            patch.is_plausible = False
            patch.score = get_reward(old_bug, patch, patch_pool)

        patch_pool.add_patch(patch)


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


def beam_search_repair():
    result_list = []
    if os.path.exists(Config.output_path):
        with open(Config.output_path, "r") as f:
            for line in f.readlines():
                json_line = json.loads(line)
                result_list.append((json_line["project"], str(json_line["bug_id"])))

    projects = get_defects4j_projects()

    # initialize policy model and tokenizer
    policy_model = make_model(Config.policy_model)
    tokenizer = AutoTokenizer.from_pretrained(Config.policy_model, use_fast=False)

    for proj in projects:
        bugs = get_defects4j_bugs_by_project(proj)

        for bug in tqdm.tqdm(bugs):
            if (proj, str(bug)) in result_list:
                continue
            try:
                bug_detail = framework.get_bug_details(proj, bug)
            except Exception as e:
                print(e)
                continue
            mode = bug_detail.bug_type.split()[0]
            if mode not in ["SL","SH"]:
                continue
            if not check_bug_detail(bug_detail):
                with jsonlines.open(Config.output_path, "a") as f:
                    f.write(
                        {"project": proj,
                         "bug_id": bug,
                         "eval": "FAIL",
                         "patch": "",
                         "patch_diff": "",
                         "score": 0,
                         "gmt_created": utc_now_str(),
                         "reflection": ''
                         }
                    )
                continue

            patch_pool = PatchPool(Config.pool_size)
            for i in range(Config.max_iter):
                execute_round(bug_detail, patch_pool, policy_model, tokenizer, i + 1)

            for patch in patch_pool.all_patches:
                with jsonlines.open(Config.output_path, "a") as f:
                    f.write(
                        {"project": proj,
                         "bug_id": bug,
                         "eval": "FAIL" if not patch.is_plausible else "PASS",
                         "patch": patch.patch_code,
                         "patch_diff": patch.patch_diff,
                         "score": patch.score,
                         "gmt_created": patch.gmt_created,
                         "reflection": patch.reflection
                         }
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Beam-search based APR"
    )

    parser.add_argument("--policy_model", required=True,
                        help="模型本地路径")
    parser.add_argument("--reward_model", required=True,
                        help="用于奖励评估的 GPT")
    parser.add_argument("--output_path", default="results.jsonl",
                        help="补丁/评估结果输出的 jsonl 文件")
    parser.add_argument("--plausible_save_path", default="",
                        help="可信补丁保存目录 (可选)")

    parser.add_argument("--beam_width", type=int, default=Config.beam_width,
                        help="Beam search 宽度")
    parser.add_argument("--max_tokens", type=int, default=Config.max_tokens,
                        help="LLM 生成的最大 token 数")
    parser.add_argument("--max_iter", type=int, default=Config.max_iter,
                        help="每个 bug 的最大迭代轮数")
    parser.add_argument("--pool_size", type=int, default=Config.pool_size,
                        help="PatchPool 容量")

    args = parser.parse_args()

    Config.policy_model = args.policy_model
    Config.reward_model = args.reward_model
    Config.output_path = args.output_path
    Config.plausible_save_path = args.plausible_save_path
    Config.beam_width = args.beam_width
    Config.max_tokens = args.max_tokens
    Config.max_iter = args.max_iter
    Config.pool_size = args.pool_size

    beam_search_repair()


if __name__ == '__main__':
    main()
