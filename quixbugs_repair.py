import json
import os
import subprocess
import difflib
import sys

import transformers

QUIXBUGS_PATH = "./QuixBugs"


def get_quixbugs_bug_list():
    bug_list = []
    for p in os.listdir(f"{QUIXBUGS_PATH}/correct_java_programs"):
        if p.endswith(".java"):
            bug_list.append(p.split(".")[0])
    # for p in os.listdir(f"{QUIXBUGS_PATH}/java_programs/extra"):
    #     if p.endswith(".java"):
    #         bug_list.append(p.split(".")[0])
    return bug_list


def get_quixbugs_bug_code(bug_name):
    with open(f"{QUIXBUGS_PATH}/java_programs/{bug_name}.java", "r") as f:
        return f.read()


def get_quixbugs_correct_code(bug_name):
    with open(f"{QUIXBUGS_PATH}/correct_java_programs/{bug_name}.java", "r") as f:
        return f.read()


def get_quixbugs_test_code(bug_name):
    with open(f"{QUIXBUGS_PATH}/java_testcases/junit/{bug_name}_TEST.java", "r") as f:
        return list(f.readlines())


class Block:
    def __init__(self, buggy_lines, correct_lines, label):
        self.buggy_lines = buggy_lines
        self.correct_lines = correct_lines
        self.label = label

    def __str__(self):
        return f"{self.label}:\n{self.buggy_lines}\n{self.correct_lines}"


class QuixBug:
    def __init__(self, bug_name):
        self.bug_name = bug_name
        self.bug_code = ""
        self.correct_code = ""
        self.diff = []
        self.test_result = None

    def get_bug_type(self):
        hunk_patch_num = 0
        line_patch_num = 0
        for t in self.diff:
            if isinstance(t, Block):
                if t.label == "hunk_patch":
                    hunk_patch_num += 1
                elif t.label == "line_patch":
                    line_patch_num += 1
        if line_patch_num == 1 and hunk_patch_num == 0:
            return "SL"
        elif hunk_patch_num == 1 and line_patch_num == 0:
            return "SH"
        else:
            return "SF"

    def get_mask_code(self):
        if self.get_bug_type() == "SF":
            raise Exception("SF bug not supported mask code")

        mask_code = ""
        for t in self.diff:
            if isinstance(t, Block):
                mask_code += "[INFILL]\n"
            else:
                mask_code += t
        return mask_code

    def get_buggy_lines(self):
        if self.get_bug_type() == "SF":
            raise Exception("SF bug have inconsecutive buggy lines")
        buggy_lines = ""
        for t in self.diff:
            if isinstance(t, Block):
                buggy_lines += '\n'.join(t.buggy_lines) + "\n"
        return buggy_lines


def get_bug_detail(bug_name):
    bug_detail = QuixBug(bug_name)
    bug_detail.bug_code = get_quixbugs_bug_code(bug_name)
    bug_detail.correct_code = get_quixbugs_correct_code(bug_name)

    bug_file = f"{QUIXBUGS_PATH}/java_programs/{bug_name}.java"
    correct_file = f"{QUIXBUGS_PATH}/correct_java_programs/{bug_name}.java"
    with open(bug_file, 'r') as f1:
        lines1 = f1.readlines()
    with open(correct_file, 'r') as f2:
        lines2 = f2.readlines()
    differ = difflib.Differ()
    diff = list(differ.compare(lines1, lines2))
    diff = [x for x in diff if not x.strip().startswith("?")]
    diff += [""]

    buggy_lines = []
    correct_lines = []

    for line in diff:
        if line.startswith("-"):
            if "package" in line or "import" in line:
                bug_detail.diff.append(line[1:].lstrip())
                continue
            buggy_lines.append(line[1:].lstrip())
        elif line.startswith("+"):
            if "package" in line or "import" in line:
                continue
            correct_lines.append(line[1:].lstrip())
        else:
            if len(buggy_lines) == 1:
                patch = Block(buggy_lines, correct_lines, "line_patch")
            elif len(buggy_lines) > 1:
                patch = Block(buggy_lines, correct_lines, "hunk_patch")
            else:
                patch = None
            if patch is not None:
                bug_detail.diff.append(patch)
            buggy_lines = []
            correct_lines = []
            bug_detail.diff.append(line[2:])
    bug_detail.diff = bug_detail.diff[:-1]
    return bug_detail


def run_test(bug_name):
    cmd = ["gradle", "test", "--tests", f"{bug_name}_TEST"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, cwd=QUIXBUGS_PATH, stderr=subprocess.PIPE,
                            universal_newlines=True).stdout
    test_code = get_quixbugs_test_code(bug_name)

    if "BUILD SUCCESSFUL" in result:
        return {
            "pass": True,
            "error": ""
        }
    else:
        lines = result.split("\n")
        errors = []
        for line in lines:
            if f"at {bug_name}_TEST.java" in line:
                location = int(line.strip().split(":")[-1])
                error_type = line.strip().split()[0]
                test_line = test_code[location - 1]
                errors.append({
                    "error_type": error_type,
                    "test_line": test_line,
                })
        return {
            "pass": False,
            "error": errors,
            "raw": result
        }


def construct_initial_message(bug: QuixBug, mode: str):
    if mode == "SL":
        prompt_header = f"""The following code contains a buggy line that has been removed.\n```java\n{bug.get_mask_code()}\n```
This was the original buggy line which was removed by the infill location:
```java\n{bug.get_buggy_lines()}\n```"""
        prompt_footer = "Please provide the correct line at the infill location."

    elif mode == "SH":
        prompt_header = f"""The following code contains a buggy hunk that has been removed.\n```java\n{bug.get_mask_code()}\n```
This was the original buggy hunk which was removed by the infill location:
```java\n{bug.get_buggy_lines()}\n```"""
        prompt_footer = "Please provide the correct hunk at the infill location."

    elif mode == "SF":
        prompt_header = f"""The following code contains a bug\n```java\n{bug.bug_code}\n```"""
        prompt_footer = "Please provide the correct function."

    initial_prompt_message = f"""{prompt_header}
The testcases fail with following errors:\n```text\n{bug.test_result['raw']}\n```
{prompt_footer}"""

    return initial_prompt_message


def construct_gpt_prompt(bug: QuixBug, mode: str):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)

    return [{"role": "system", "content": "You are an automated program repair tool."},
            {"role": "user", "content": initial_prompt_message}]


def construct_llm_prompt(bug: QuixBug, tokenizer, mode: str):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)
    dict = {
        "SL": "line",
        "SH": "hunk",
        "SF": "function"
    }
    assistant_response_header = "The correct {} is:\n```java".format(dict[mode])
    template = [
        {"role": "system", "content": "You are an automated program repair tool."},
        {"role": "user", "content": initial_prompt_message},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False,
                                         add_generation_prompt=True) + "\n" + assistant_response_header


def construct_prompt_without_template(bug: QuixBug, mode: str):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)
    dict = {
        "SL": "line",
        "SH": "hunk",
        "SF": "function"
    }
    assistant_response_header = "The correct {} is:\n```java".format(dict[mode])
    system_prompt = "You are an automated program repair tool."
    PROMPT = """
    ### Instruction
    {instruction}
    ### Response
    {response_header}
    """
    return PROMPT.format(instruction= initial_prompt_message,
                         response_header=assistant_response_header)


def extract_patch_from_response(response, mode=None):
    if "```java" in response:
        _patch = response[response.find("```java") + len("```java") + 1:]
        _patch = _patch[:_patch.find("\n```")]
    else:
        _patch = response
    if mode == "SL":
        while len(_patch) > 0 and _patch.startswith("\n"):
            _patch = _patch[1:]
        while len(_patch) > 0 and _patch.endswith("\n"):
            _patch = _patch[:-1]
        if "\n" in _patch:
            _patch = _patch[:_patch.find("\n")]
    return _patch


def validate_patch(bug: QuixBug, proposed_patch: str):
    bug_type = bug.get_bug_type()
    if bug_type == "SL":
        new_code = bug.get_mask_code().replace("[INFILL]", proposed_patch)
    elif bug_type == "SH":
        new_code = bug.get_mask_code().replace("[INFILL]", proposed_patch)
    else:
        new_code = proposed_patch
    with open(f"{QUIXBUGS_PATH}/java_programs/{bug.bug_name}.java", "w") as f:
        f.write(new_code)
    result = run_test(bug.bug_name)
    with open(f"{QUIXBUGS_PATH}/java_programs/{bug.bug_name}.java", "w") as f:
        f.write(bug.bug_code)
    return result, new_code


def llm_repair(model_name):
    from llm import generate_patches, make_model
    model_paths = {
        "dscoder": "/root/deepseek-coder-6.7b-instruct",
        "llama-3b": "/root/autodl-tmp/Llama-3.2-3B-Instruct",
        "qwen-3b": "/root/autodl-tmp/Qwen2.5-Coder-3B-Instruct",
        "stable-3b": "/root/autodl-tmp/stable-code-instruct-3b",
        "llama-8b": "/root/autodl-tmp/Llama-3.1-8B-Instruct",
        "yi-9b": "/root/autodl-tmp/Yi-Coder-9B-Chat",
        "qwen-7b": "/root/autodl-tmp/Qwen2.5-Coder-7B-Instruct",
        "deci-7b": "/root/autodl-tmp/DeciLM-7B-instruct",
        "falcon-7b": "/root/autodl-tmp/falcon-7b-instruct",
        "phi-mini": "/root/autodl-tmp/Phi-3.5-mini-instruct",
        "calme-3b": "/root/autodl-tmp/calme-3.1-instruct-3b",
        "starcoder2-3b":"/root/autodl-tmp/starcoder2-3b-instruct"
    }
    if model_name not in model_paths.keys():
        raise Exception("Model not supported")
    model_path = model_paths.get(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                                           trust_remote_code=True,
                                                           model_max_length=2048)
    tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
    tokenizer.bos_token_id = tokenizer.eos_token_id
    model = make_model(model_path=model_path)
    for bug_name in get_quixbugs_bug_list():
        bug_detail = get_bug_detail(bug_name)
        bug_detail.test_result = run_test(bug_name)
        mode = bug_detail.get_bug_type()

        if tokenizer.chat_template is None:
            prompt = construct_prompt_without_template(bug_detail, mode)
        else:
            prompt = construct_llm_prompt(bug_detail, tokenizer, mode)
        print(prompt)

        num_samples = 20
        responses = generate_patches(model, prompt, num_samples=num_samples)
        patches = [extract_patch_from_response(response, mode) for response in responses]
        is_fixed = False
        for i, patch in enumerate(patches):
            print(patch)
            result, new_code = validate_patch(bug_detail, patch)
            print(result)
            if result['pass']:
                with open(f"quixbugs_{model_name}.jsonl", "a") as f:
                    record = {
                        "bug": bug_detail.bug_name,
                        "eval": "PASS",
                        "attempt": i + 1,
                        "mode": mode,
                        "patch": new_code,
                    }
                    f.write(json.dumps(record) + "\n")
                    is_fixed = True
                    break
        if is_fixed:
            continue
        with open(f"quixbugs_{model_name}.jsonl", "a") as f:
            record = {
                "bug": bug_detail.bug_name,
                "eval": "FAIL",
                "attempt": num_samples,
                "mode": mode,
                "patch": "",
            }
            f.write(json.dumps(record) + "\n")


def gpt_repair():
    from gpt import generate_patches

    for bug_name in get_quixbugs_bug_list():
        bug_detail = get_bug_detail(bug_name)
        bug_detail.test_result = run_test(bug_name)
        mode = bug_detail.get_bug_type()

        prompt = construct_gpt_prompt(bug_detail, mode)
        print(prompt)

        num_samples = 20
        responses = generate_patches(prompt, num_samples=num_samples)
        patches = [extract_patch_from_response(response, mode) for response in responses]

        is_fixed = False
        for i, patch in enumerate(patches):
            print(patch)
            result, new_code = validate_patch(bug_detail, patch)
            print(result)
            if result['pass']:
                with open("quixbugs_gpt.jsonl", "a") as f:
                    record = {
                        "bug": bug_detail.bug_name,
                        "eval": "PASS",
                        "attempt": i + 1,
                        "mode": mode,
                        "patch": new_code,
                    }
                    f.write(json.dumps(record) + "\n")
                    is_fixed = True
                    break
        if is_fixed:
            continue
        with open("quixbugs_gpt.jsonl", "a") as f:
            record = {
                "bug": bug_detail.bug_name,
                "eval": "FAIL",
                "attempt": num_samples,
                "mode": mode,
                "patch": "",
            }
            f.write(json.dumps(record) + "\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        raise Exception("Model name not specified")
    if len(sys.argv) >= 3:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    else:
        print("use default gpu")
    llm_repair(model_name)
