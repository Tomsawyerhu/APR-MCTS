from framework import Bug

def construct_initial_message(bug: Bug, mode: str):
    tot_prompt = """
    Identify and behave as three different experts that are cooperating to repair the same bug .
All experts will write down the step and their thinking about the step, then share it with the group.
Then, all experts will go on to the next step, etc.
At each step all experts will score their peers response between 1 and 5, 1 meaning the patch is highly unlikely to be correct, and 5 meaning th patch is highly likely to be correct.
If any expert is judged to be wrong at any point then they leave.
After all experts have provided their analysis, you then analyze all 3 analyses and provide either the consensus patch or your best guess patch.
"""

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
{tot_prompt}\n
{prompt_footer}"""

    return initial_prompt_message


def construct_gpt_tot_prompt(bug: Bug, mode: str):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)

    return [{"role": "system",
             "content": "You are an automated program repair tool. Please do not use language features beyond Java 1.4, such as foreach and generics <>."},
            {"role": "user", "content": initial_prompt_message}]


def construct_llm_tot_prompt(bug: Bug, mode: str, tokenizer):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode)
    template = [
        {"role": "system",
         "content": "You are an automated program repair tool. Please do not use language features beyond Java 1.4, such as foreach and generics <>."},
        {"role": "user", "content": initial_prompt_message},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
