from framework import Bug


def construct_initial_message(bug: Bug, mode: str):
    cot_prompt = "Before you give the final answer, let's think step by step. You need to explain where bug happens and how your answer can avoid it. "

    if mode == "SL":
        prompt_header = f"""The following code contains a buggy line that has been removed.\n```java\n{bug.masked_code}\n```
This was the original buggy line which was removed by the infill location:
```java\n{bug.buggy_lines}\n```\nTest cases look like:```java\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"""
        prompt_footer = f"Please provide the correct line at the infill location, your answer should begin with ```java"

    elif mode == "SH":
        prompt_header = f"""The following code contains a buggy hunk that has been removed.\n```java\n{bug.masked_code}\n```
This was the original buggy hunk which was removed by the infill location:
```java\n{bug.buggy_lines}\n```\nTest cases look like:```java\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"""
        prompt_footer = f"Please provide the correct hunk at the infill location, your answer should begin with ```java"

    elif mode == "SF":
        prompt_header = f"""The following code contains a bug\n```java\n{bug.code}\n```\nTest cases look like:```java\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"""
        prompt_footer = "Please provide the correct function, starting with ```java"

    initial_prompt_message = f"""{prompt_header}
{cot_prompt}\n
{prompt_footer}"""

    return initial_prompt_message


def construct_gpt_cot_prompt(bug: Bug, mode: str):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)

    return [{"role": "system",
             "content": "You are an automated program repair tool. Please do not use language features beyond Java 1.4, such as foreach and generics <>."},
            {"role": "user", "content": initial_prompt_message}]


def construct_llm_cot_prompt(bug: Bug, mode: str, tokenizer):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)
    template = [
        {"role": "system",
         "content": "You are an automated program repair tool. Please do not use language features beyond Java 1.4, such as foreach and generics <>."},
        {"role": "user", "content": initial_prompt_message},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
