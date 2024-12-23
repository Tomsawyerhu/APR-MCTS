import json
import os
import sys

import transformers
from tqdm import tqdm

import prompt, framework
from get_d4j_bug_list import get_defects4j_projects, get_defects4j_bugs_by_project
from gpt import generate_patches as gpt_generate_patches
from llm import generate_patches as generate_patches_llm, make_model

model = None


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


def load_result(result_file="./result.jsonl"):
    if not os.path.exists(result_file):
        return []
    with open(result_file, 'r') as f:
        for line in f:
            yield json.loads(line)


def gpt_repair(project, bug_id):
    result_file = "./result_gpt.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return
    p = prompt.construct_initial_prompt(bug=b, mode=mode)
    print(p)
    num_samples = 5
    responses = gpt_generate_patches(p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def qwen_repair(project, bug_id):
    global model
    model_path = "/root/autodl-tmp/Qwen2.5-Coder-7B-Instruct"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_qwen.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_qwen(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def deepseek_repair(project, bug_id):
    global model
    model_path = "/root/deepseek-coder-6.7b-instruct"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_dscoder.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_dscoder(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def yi_repair(project, bug_id):
    global model
    model_path = "/root/autodl-tmp/Yi-Coder-9B-Chat"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_yi.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_yi(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def llama_repair(project, bug_id):
    global model
    model_path = "/root/autodl-tmp/Llama-3.1-8B-Instruct"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_llama.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_llama(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def falcon_repair(project, bug_id):
    global model
    model_path = "/root/autodl-fs/falcon-7b-instruct"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_falcon.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_falcon(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def deci_repair(project, bug_id):
    global model
    model_path = "/root/autodl-fs/DeciLM-7B-instruct"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_deci.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_deci(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def phi_repair(project, bug_id):
    global model
    model_path = "/root/autodl-tmp/Phi-3.5-mini-instruct"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_phi.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_phi(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def calme_repair(project, bug_id):
    global model
    model_path = "/root/autodl-tmp/calme-3.1-instruct-3b"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_calme.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_calme(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def starcoder_repair(project, bug_id):
    global model
    model_path = "/root/autodl-tmp/starcoder2-3b-instruct"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_starcoder.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_starcoder(bug=b, mode=mode)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")

def stable_repair(project, bug_id):
    global model
    model_path = "/root/autodl-tmp/stable-code-instruct-3b"
    if model is None:
        model = make_model(model_path)
    result_file = "./result_stable.jsonl"
    b = framework.get_bug_details(project=project, bug_id=bug_id)
    modes = list(b.bug_type.split())
    mode = modes[0]
    if mode == "OT":
        return

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id

    p = prompt.construct_prompt_for_stable(bug=b, mode=mode, tokenizer=tokenizer)
    print(p)
    num_samples = 20
    responses = generate_patches_llm(model, p, num_samples=num_samples)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=b, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {b.project}-{b.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": b.project,
                    "bug_id": b.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
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
    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python repair.py <model_name>")
        exit(1)
    model_name = sys.argv[1]
    if len(sys.argv) >= 3:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    else:
        print("use default gpu")
    repair_method_mapping = {
        "gpt": gpt_repair,
        "qwen": qwen_repair,
        "dscoder": deepseek_repair,
        "yi": yi_repair,
        "llama": llama_repair,
        "falcon": falcon_repair,
        "deci": deci_repair,
        "phi": phi_repair,
        "calme": calme_repair,
        "starcoder": starcoder_repair,
        "stable":stable_repair
    }
    if model_name not in repair_method_mapping:
        raise ValueError("Invalid model name")
    repair_method = repair_method_mapping[model_name]

    result = load_result("./result_{}.jsonl".format(model_name))
    verified_bugs = [(r['project'], str(r['bug_id'])) for r in result]
    projects = get_defects4j_projects()
    for proj in projects:
        bugs = get_defects4j_bugs_by_project(proj)
        for bug in tqdm(bugs):
            if (proj, str(bug)) in verified_bugs:
                continue
            try:
                repair_method(proj, bug)
            except Exception as e:
                continue
