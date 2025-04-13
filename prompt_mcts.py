from framework import Bug


def construct_initial_message(bug: Bug, mode: str, language="java"):
    cot_prompt = "Before you give the final answer, let's think step by step. You need to explain where bug happens and how your answer can avoid it. "

    if mode == "SL":
        prompt_header = f"""The following code contains a buggy line that has been removed.\n```{language}\n{bug.masked_code}\n```
This was the original buggy line which was removed by the infill location:
```{language}\n{bug.buggy_lines}\n```\nTest cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"""
        prompt_footer = f"After giving reflection, please provide the correct line at the infill location, only single line is allowed. Pay attention to indentation. your answer must be different from ```{language}\n{bug.buggy_lines}\n``` , your answer should begin with ```{language}"

    elif mode == "SH":
        prompt_header = f"""The following code contains a buggy hunk that has been removed.\n```{language}\n{bug.masked_code}\n```
This was the original buggy hunk which was removed by the infill location:
```{language}\n{bug.buggy_lines}\n```\nTest cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"""
        prompt_footer = f"After giving reflection, please provide the correct hunk at the infill location, only single hunk is allowed. Pay attention to indentation. Your answer must be different from ```{language}\n{bug.buggy_lines}\n``` , your answer should begin with ```{language}"


    elif mode == "SF":
        prompt_header = f"The following code contains a bug\n```{language}\n{bug.code}\n```\nTest cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"
        prompt_footer = f"After giving reflection, please provide the correct function, starting with ```{language}"

    initial_prompt_message = f"""{prompt_header}
{cot_prompt}\n
{prompt_footer}"""

    return initial_prompt_message


def construct_gpt_policy_prompt(bug: Bug, mode: str,language="java"):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode,language=language)

    return [{"role": "system",
             "content": "You are an automated program repair tool. Please do not use language features beyond {language} 1.4, such as foreach and generics <>."},
            {"role": "user", "content": initial_prompt_message}]


def construct_llm_policy_prompt(bug: Bug, mode: str, tokenizer,language="java"):
    initial_prompt_message = construct_initial_message(bug=bug, mode=mode,language=language)
    template = [
        {"role": "system",
         "content": "You are an automated program repair tool. Please do not use language features beyond {language} 1.4, such as foreach and generics <>."},
        {"role": "user", "content": initial_prompt_message},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)


def construct_llm_reward_prompt(bug: Bug, wrong_patch: str, reflection: str, mode: str, tokenizer,language="java"):
    prompt_footer = "Please give a score between 0 and 100, the score stands for quality of the patch, 0 means the patch is of very poor quality, 100 means the patch is correct."
    if mode == "SL":
        prompt_header = f"""The following code contains a buggy line that has been removed.\n```{language}\n{bug.masked_code}\n```
    This was the original buggy line which was removed by the infill location:
    ```{language}\n{bug.buggy_lines}\n```\nTest cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```
    This was the patch line used to fix the bug:
    ```{language}\n{wrong_patch}\n```
    This was the reflection on bug and patch:
    ```text\n{reflection}\n```
    """


    elif mode == "SH":
        prompt_header = f"""The following code contains a buggy hunk that has been removed.\n```{language}\n{bug.masked_code}\n```
    This was the original buggy hunk which was removed by the infill location:
    ```{language}\n{bug.buggy_lines}\n```\nTest cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```
    This was the patch hunk used to fix the bug:
    ```{language}\n{wrong_patch}\n```
    This was the reflection on bug and patch:
    ```text\n{reflection}\n```
    """


    elif mode == "SF":
        prompt_header = f"""The following code contains a bug\n```{language}\n{bug.code}\n```\nTest cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```
    This was the patch used to fix the bug:
    ```{language}\n{wrong_patch}\n```
    This was the reflection on bug and patch:
    ```text\n{reflection}\n```
    """

    initial_prompt_message = f"""
    {prompt_header}
    {prompt_footer}
    """

    template = [
        {"role": "system", "content": "You are an automated program repair tool."},
        {"role": "user", "content": initial_prompt_message},
    ]
    return tokenizer.apply_chat_template(template, tokenize=False,
                                         add_generation_prompt=True) + "\n" + "The score is: ```text"


def construct_gpt_reward_prompt(bug: Bug, wrong_patch: str, reflection: str, mode: str, tokenizer,language="java"):
    prompt_footer = "Please give a score between 0 and 100, the score stands for quality of the patch, 0 means the patch is of very poor quality, 100 means the patch is correct. The answer should be an integer."
    if mode == "SL":
        prompt_header = f"""The following code contains a buggy line that has been removed.\n```{language}\n{bug.masked_code}\n```
    This was the original buggy line which was removed by the infill location:
    ```{language}\n{bug.buggy_lines}\n```\nTest cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests[:1024]}\n```
    This was the patch line used to fix the bug:
    ```{language}\n{wrong_patch}\n```
    This was the reflection on bug and patch:
    ```text\n{reflection}\n```
    """


    elif mode == "SH":
        prompt_header = f"""The following code contains a buggy hunk that has been removed.\n```{language}\n{bug.masked_code}\n```
    This was the original buggy hunk which was removed by the infill location:
    ```{language}\n{bug.buggy_lines}\n```\nTest cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests[:1024]}\n```
    This was the patch hunk used to fix the bug:
    ```{language}\n{wrong_patch}\n```
    This was the reflection on bug and patch:
    ```text\n{reflection}\n```
    """


    elif mode == "SF":
        prompt_header = f"""The following code contains a bug\n```{language}\n{bug.code}\n```
        Test cases look like:```{language}\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests[:1024]}\n```
    This was the patch used to fix the bug:
    ```{language}\n{wrong_patch}\n```
    This was the reflection on bug and patch:
    ```text\n{reflection}\n```
    """

    initial_prompt_message = f"""
    {prompt_header}
    {prompt_footer}
    """

    template = [
        {"role": "system", "content": "You are an automated program repair tool."},
        {"role": "user", "content": initial_prompt_message},
    ]
    return template
