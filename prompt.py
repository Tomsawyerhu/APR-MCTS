import transformers

from framework import Bug


def construct_initial_message(bug: Bug, mode: str):
    if mode == "SL":
        prompt_header = f"""The following code contains a buggy line that has been removed.\n```java\n{bug.masked_code}\n```
This was the original buggy line which was removed by the infill location:
```java\n{bug.buggy_lines}\n```"""
        prompt_footer = "Please provide the correct line at the infill location."

    elif mode == "SH":
        prompt_header = f"""The following code contains a buggy hunk that has been removed.\n```java\n{bug.masked_code}\n```
This was the original buggy hunk which was removed by the infill location:
```java\n{bug.buggy_lines}\n```"""
        prompt_footer = "Please provide the correct hunk at the infill location."

    elif mode == "SF":
        prompt_header = f"""The following code contains a bug\n```java\n{bug.code}\n```"""
        prompt_footer = "Please provide the correct function."

    initial_prompt_message = f"""{prompt_header}
The code fails on this test:\n```\n{bug.test_name}\n```
on this test line:\n```java\n{bug.test_line}\n```
with the following test error:\n```\n{bug.test_error_message}\n```
{prompt_footer}"""

    return initial_prompt_message


def construct_initial_prompt(bug: Bug, mode: str):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)

    return [{"role": "system", "content": "You are an automated program repair tool."},
            {"role": "user", "content": initial_prompt_message}]


def construct_prompt_for_qwen(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header


def construct_prompt_for_dscoder(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header


def construct_prompt_for_yi(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header

def construct_prompt_for_llama(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header

def construct_prompt_for_falcon(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header


def construct_prompt_for_deci(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header

def construct_prompt_for_phi(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header

def construct_prompt_for_calme(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header

def construct_prompt_for_stable(bug: Bug, mode: str, tokenizer):
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
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)+"\n"+assistant_response_header

def construct_prompt_for_starcoder(bug: Bug, mode: str):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)
    dict = {
        "SL": "line",
        "SH": "hunk",
        "SF": "function"
    }
    assistant_response_header = "The correct {} is:\n```java".format(dict[mode])
    PROMPT = """
    ### Instruction
    {instruction}
    ### Response
    {response_header}
    """
    return PROMPT.format(instruction= initial_prompt_message,
                         response_header=assistant_response_header)