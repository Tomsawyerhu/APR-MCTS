import json
import os
import sys

import transformers
from tqdm import tqdm

import framework
from get_d4j_bug_list import get_defects4j_projects, get_defects4j_bugs_by_project
from llm import make_model, generate_patches as generate_patches_llm
from gpt import generate_patches as generate_patches_gpt

from prompt_tot import *
from prompt_cot import *

model = None

model_path = {
    "qwen-3b": "/mnt/data/hhc/Qwen2.5-Coder-3B-Instruct",
    "llama-3b": "/mnt/data/hhc/Llama-3.2-3B-Instruct",
    "gpt-4o":"gpt-4o",
    "gpt-4o-mini":"gpt-4o-mini"
}


def extract_patch_from_response(response, mode=None):
    if "```java" in response:
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


def llm_tot_repair(project, bug_id, model_name):
    global model, model_path
    if model is None:
        model = make_model(model_path.get(model_name, ""))
    plausible_patch_dir = f"./{model_name}_tot_plausible"
    if not os.path.exists(plausible_patch_dir):
        os.makedirs(plausible_patch_dir)
    result_file = f"./{model_name}.tot.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path.get(model_name, ""), use_fast=True,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = construct_llm_tot_prompt(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]

    first_plausible_recorded = False
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            record = {
                "project": b.project,
                "bug_id": b.bug_id,
                "eval": "PASS",
                "attempt": i + 1,
                "mode": mode,
                "patch": patch,
                "diff": patch_diff,
            }
            if not first_plausible_recorded:
                with open(result_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    first_plausible_recorded = True
            with open(plausible_patch_dir + f"/{b.project}-{b.bug_id}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

        elif result_reason == b.test_error_message:
            print(
                f"Proposed patch of {b.project}-{b.bug_id} ({mode}) failed with same error message as original bug")
        else:
            print(
                f"Proposed patch of {b.project}-{b.bug_id} ({mode}) failed with a different error message than original bug")

    record = {
        "project": b.project,
        "bug_id": b.bug_id,
        "eval": "FAIL",
        "attempt": num_samples,
        "mode": mode,
        "patch": "",
        "diff": "",
    }
    if not first_plausible_recorded:
        with open(result_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def llm_cot_repair(project, bug_id, model_name):
    global model, model_path
    if model is None:
        model = make_model(model_path.get(model_name, ""))
    plausible_patch_dir = f"./{model_name}_cot_plausible"
    if not os.path.exists(plausible_patch_dir):
        os.makedirs(plausible_patch_dir)
    result_file = f"./{model_name}.cot.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path.get(model_name, ""), use_fast=True,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = construct_llm_cot_prompt(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]

    first_plausible_recorded = False
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            record = {
                "project": b.project,
                "bug_id": b.bug_id,
                "eval": "PASS",
                "attempt": i + 1,
                "mode": mode,
                "patch": patch,
                "diff": patch_diff,
            }
            if not first_plausible_recorded:
                with open(result_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    first_plausible_recorded = True
            with open(plausible_patch_dir + f"/{b.project}-{b.bug_id}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

        elif result_reason == b.test_error_message:
            print(
                f"Proposed patch of {b.project}-{b.bug_id} ({mode}) failed with same error message as original bug")
        else:
            print(
                f"Proposed patch of {b.project}-{b.bug_id} ({mode}) failed with a different error message than original bug")

    record = {
        "project": b.project,
        "bug_id": b.bug_id,
        "eval": "FAIL",
        "attempt": num_samples,
        "mode": mode,
        "patch": "",
        "diff": "",
    }
    if not first_plausible_recorded:
        with open(result_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def gpt_tot_repair(project, bug_id, model_name):

    plausible_patch_dir = f"./{model_name}_tot_plausible"
    if not os.path.exists(plausible_patch_dir):
        os.makedirs(plausible_patch_dir)
    result_file = f"./{model_name}.tot.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    p = construct_gpt_tot_prompt(bug=b, mode=mode)
    print(p)
    num_samples = 20
    responses = generate_patches_gpt(p,model_name, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]

    first_plausible_recorded = False
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            record = {
                "project": b.project,
                "bug_id": b.bug_id,
                "eval": "PASS",
                "attempt": i + 1,
                "mode": mode,
                "patch": patch,
                "diff": patch_diff,
            }
            if not first_plausible_recorded:
                with open(result_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    first_plausible_recorded = True
            with open(plausible_patch_dir + f"/{b.project}-{b.bug_id}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

        elif result_reason == b.test_error_message:
            print(
                f"Proposed patch of {b.project}-{b.bug_id} ({mode}) failed with same error message as original bug")
        else:
            print(
                f"Proposed patch of {b.project}-{b.bug_id} ({mode}) failed with a different error message than original bug")

    record = {
        "project": b.project,
        "bug_id": b.bug_id,
        "eval": "FAIL",
        "attempt": num_samples,
        "mode": mode,
        "patch": "",
        "diff": "",
    }
    if not first_plausible_recorded:
        with open(result_file, "a") as f:
            f.write(json.dumps(record) + "\n")

def gpt_cot_repair(project, bug_id, model_name):
    plausible_patch_dir = f"./{model_name}_cot_plausible"
    if not os.path.exists(plausible_patch_dir):
        os.makedirs(plausible_patch_dir)
    result_file = f"./{model_name}.cot.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    p = construct_gpt_tot_prompt(bug=b, mode=mode)
    print(p)
    num_samples = 20
    responses = generate_patches_gpt(p,model_name, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]

    first_plausible_recorded = False
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            record = {
                "project": b.project,
                "bug_id": b.bug_id,
                "eval": "PASS",
                "attempt": i + 1,
                "mode": mode,
                "patch": patch,
                "diff": patch_diff,
            }
            if not first_plausible_recorded:
                with open(result_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    first_plausible_recorded = True
            with open(plausible_patch_dir + f"/{b.project}-{b.bug_id}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")

        elif result_reason == b.test_error_message:
            print(
                f"Proposed patch of {b.project}-{b.bug_id} ({mode}) failed with same error message as original bug")
        else:
            print(
                f"Proposed patch of {b.project}-{b.bug_id} ({mode}) failed with a different error message than original bug")

    record = {
        "project": b.project,
        "bug_id": b.bug_id,
        "eval": "FAIL",
        "attempt": num_samples,
        "mode": mode,
        "patch": "",
        "diff": "",
    }
    if not first_plausible_recorded:
        with open(result_file, "a") as f:
            f.write(json.dumps(record) + "\n")
def load_result(result_file="./result.jsonl"):
    if not os.path.exists(result_file):
        return []
    with open(result_file, 'r') as f:
        for line in f:
            yield json.loads(line)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cot_tot.py <model_name>")
        exit(1)
    model_name = sys.argv[1]
    if len(sys.argv) >= 3:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    else:
        print("use default gpu")

    if model_name not in model_path.keys():
        raise ValueError("Invalid model name")

    cot_result = load_result(f"./{model_name}.cot.jsonl")
    cot_verified_bugs = [(r['project'], str(r['bug_id'])) for r in cot_result]
    tot_result = load_result(f"./{model_name}.tot.jsonl")
    tot_verified_bugs = [(r['project'], str(r['bug_id'])) for r in tot_result]

    projects = get_defects4j_projects()
    for proj in projects:
        bugs = get_defects4j_bugs_by_project(proj)
        for bug in tqdm(bugs):
            # cot repair
            if (proj, str(bug)) in cot_verified_bugs:
                pass
            else:
                try:
                    if "gpt" in model_name:
                        gpt_cot_repair(proj, bug, model_name)
                    else:
                        llm_cot_repair(proj, bug, model_name)
                except Exception as e:
                    print(e)
            # tot repair
            if (proj, str(bug)) in tot_verified_bugs:
                pass
            else:
                try:
                    if "gpt" in model_name:
                        gpt_tot_repair(proj, bug, model_name)
                    else:
                        llm_tot_repair(proj, bug, model_name)
                except Exception as e:
                    print(e)

