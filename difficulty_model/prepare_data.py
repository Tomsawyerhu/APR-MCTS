# llm生成50个答案，基于self-consistency，50个答案与标准答案之间的语义是否相同 判断修复缺陷的难度
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def construct_prompt_for_repair_single_chunk(mask_code, buggy_lines, tokenizer):
    template = [
        {"role": "system", "content": "You are an automated program repair tool."},
        {"role": "user",
         "content": f"The following code contains a buggy hunk that has been removed.\n```java\n{mask_code}\n```\nThis was the original buggy hunk which was removed by the infill location:\n```java\n{buggy_lines}\n```\nPlease provide the correct hunk at the infill location."},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False,
                                         add_generation_prompt=True) + "\nThe correct chunk should be:\n```java"


def construct_prompt_for_repair_single_function(buggy_code, tokenizer):
    template = [
        {"role": "system", "content": "You are an automated program repair tool."},
        {"role": "user",
         "content": f"The following code contains a bug\n```java\n{buggy_code}\n```\nPlease provide the correct function."},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False,
                                         add_generation_prompt=True) + "\nThe correct function should be:\n```java"


def construct_prompt_for_gt_check_single_chunk(mask_code, gt_fixed_lines, answer, tokenizer):
    template = [
        {"role": "system", "content": "You are an automated program repair tool."},
        {"role": "user",
         "content": f"The following code contains a buggy hunk that has been removed.\n```java\n{mask_code}\n```\nThis was the groundtruth fixed hunk which was removed by the infill location:\n```java\n{gt_fixed_lines}\n```\nDo you think the following fixed lines can fix the bug?\n```java\n{answer}\n```\nPlease answer Yes or No."},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False,
                                         add_generation_prompt=True) + "\nThe answer is:\n```text"


def construct_prompt_for_gt_check_single_function(gt_function, answer, tokenizer):
    template = [
        {"role": "system", "content": "You are an automated program repair tool."},
        {"role": "user",
         "content": f"The following code is a groundtruth function after fix.\n```java\n{gt_function}\n```\nDo you think the following function is as correct as the groundtruth?\n```java\n{answer}\n```\nPlease answer Yes or No."},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False,
                                         add_generation_prompt=True) + "\nThe answer is:\n```text"


def make_model(model_path=""):
    kwargs = {
        "tensor_parallel_size": 1,  # int(os.getenv("VLLM_N_GPUS", "1"))
        "dtype": "float16",
        "trust_remote_code": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.98
    }
    model = LLM(model_path, max_model_len=2048, **kwargs)
    return model


def generate_patches(llm, prompt, num_samples=1):
    vllm_outputs = llm.generate(
        prompt,
        SamplingParams(
            temperature=0.9,
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            n=num_samples,
        ),
        use_tqdm=False,
    )

    output_texts = [x.text for x in vllm_outputs[0].outputs]
    return output_texts


def generate(llm, prompt):
    vllm_outputs = llm.generate(
        prompt,
        SamplingParams(
            temperature=0,
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        ),
        use_tqdm=False,
    )

    text = vllm_outputs[0].outputs[0].text
    return text


def extract_patch_from_response(response):
    if "```java" in response:
        patch = response[response.find("```java") + len("```java") + 1:]
        patch = patch[:patch.find("\n```")]
    else:
        patch = response
    if "```" in patch:
        patch = patch[:patch.find("```")]
    return patch


def read_jsonl(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list


k = 50
data = read_jsonl("./data/train-00000-of-00004-179f0635c54dfdf9.jsonl")
repair_model = make_model("/mnt/workspace/Qwen2.5-Coder-1.5B-Instruct")
repair_tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/Qwen2.5-Coder-1.5B-Instruct",
                                                 trust_remote_code=True)
check_model = repair_model
check_tokenizer = repair_tokenizer
for line in data:
    if line["is_single_chunk"]:
        prompt = construct_prompt_for_repair_single_chunk(line["mask_code"], line["buggy_lines"], repair_tokenizer)
    else:
        prompt = construct_prompt_for_repair_single_function(line["buggy_function"], repair_tokenizer)
    # 过滤掉prompt过长的
    if len(repair_tokenizer.encode(prompt)) > 2048:
        continue
    print(prompt)
    responses = generate_patches(repair_model, prompt, num_samples=k)
    patches = [extract_patch_from_response(response) for response in responses]
    print(patches)
    patch_check_result = []
    for patch in patches:
        if line["is_single_chunk"]:
            check_prompt = construct_prompt_for_gt_check_single_chunk(line["mask_code"], line["fixed_lines"], patch,
                                                                      check_tokenizer)
            if len(check_tokenizer.encode(check_prompt)) > 2048:
                continue
        else:
            check_prompt = construct_prompt_for_gt_check_single_function(line["fixed_function"], patch, check_tokenizer)
        yes_or_no = generate(repair_model, check_prompt)
        print(yes_or_no)
        if "yes" in yes_or_no.lower():
            patch_check_result.append(1)
        elif "no" in yes_or_no.lower():
            patch_check_result.append(0)
        else:
            # raise ValueError(f"Invalid check result,{yes_or_no}")
            patch_check_result.append(0)
    if len(patch_check_result) != k:
        continue
    saving={
        **line,
        "difficulty": 1 - sum(patch_check_result) / k
    }
    with open("./data/check_result.jsonl", "a") as f:
        f.write(json.dumps(saving) + "\n")
