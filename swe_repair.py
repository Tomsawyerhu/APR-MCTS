import json
import re
import time
import uuid
from difflib import get_close_matches
from typing import Optional
import os
import platform
import subprocess
import openai
import tree_sitter_javascript
from tree_sitter import Language, Parser

AGENTLESS_PROMPT = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to fix the issue.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

AGENTLESS_REFINE_PROMPT = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs:
--- BEGIN FILE ---
{retrieval}
--- END FILE ---

Below is a partial patch fails to fix the issue:
--- BEGIN PATCH ---
{partial_patch}
--- END PATCH ---

Its test report looks like:
--- BEGIN REPORT ---
{test_report}
--- END REPORT ---

Please first localize the bug based on the issue statement, and then generate *SEARCH/REPLACE* edits to refine the partial patch.

Every *SEARCH/REPLACE* edit must use this format:
1. The file path
2. The start of search block: <<<<<<< SEARCH
3. A contiguous chunk of lines to search for in the existing source code
4. The dividing line: =======
5. The lines to replace into the source code
6. The end of the replace block: >>>>>>> REPLACE

Here is an example:

```python
### mathweb/flask/app.py
<<<<<<< SEARCH
from flask import Flask
=======
import math
from flask import Flask
>>>>>>> REPLACE
```

Please note that the *SEARCH/REPLACE* edit REQUIRES PROPER INDENTATION. If you would like to add the line '        print(x)', you must fully write that out, with all those spaces before the code!
Wrap the *SEARCH/REPLACE* edit in blocks ```python...```.
"""

PATCH_EVALUATE_PROMPT = """
We are currently solving the following issue within our repository. Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below is a partial patch fails to fix the issue:
--- BEGIN PATCH ---
{partial_patch}
--- END PATCH ---

Its test report looks like:
--- BEGIN REPORT ---
{test_report}
--- END REPORT ---

Please give a score between 0 and 100 to evaluate the patch. e.g., I will give **90** to this patch, because ...
"""


def request_chatgpt_engine(config, base_url=None, api_key=None, max_retries=40, timeout=100):
    ret = None
    retries = 0

    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    while ret is None and retries < max_retries:
        try:
            print("Creating API request")
            ret = client.chat.completions.create(**config)

            if ret is None or not hasattr(ret, "choices"):
                print(f"Invalid response received: {ret}")
                raise Exception("Invalid API response")

        except openai.OpenAIError as e:
            if isinstance(e, openai.BadRequestError):
                print("Request invalid")
                print(str(e))
                raise Exception("Invalid API Request")
            elif isinstance(e, openai.RateLimitError):
                print("Rate limit exceeded. Waiting...")
                print(str(e))
                time.sleep(5)
            elif isinstance(e, openai.APIConnectionError):
                print("API connection error. Waiting...")
                print(str(e))
                time.sleep(5)
            elif isinstance(
                    e, openai.APITimeoutError
            ):  # Add specific handling for timeout
                print(f"Request timed out after {timeout} seconds. Retrying...")
                print(str(e))
                time.sleep(1)
            else:
                print("Unknown error. Waiting...")
                print(str(e))
                time.sleep(1)

        retries += 1
        if retries >= max_retries:
            print(f"Max retries ({max_retries}) exceeded")
            ret = None

    print(f"API response {ret}")
    return ret


def generate_with_retries(
        instance_id,
        prompt,
        output_file,
        file_contents: list,
        found_files: list,
        temperature=0.0,
        max_retries=5,
        model_name='',
        base_url='',
        api_key='',
        num_samples=1,
        max_completion_tokens=16000,
        log_file=None
):
    for attempt in range(max_retries):
        try:
            config = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "n": num_samples,
                "temperature": temperature,
                "max_tokens": max_completion_tokens,
            }
            completion = request_chatgpt_engine(config, base_url=base_url, api_key=api_key)
            if completion is None:
                with open(log_file, 'a') as log_writer:
                    log_writer.write('=' * 80)
                    log_writer.write('\n')
                    log_writer.write('RESPONSE')
                    log_writer.write('\n')
                    log_writer.write('=' * 80)
                    log_writer.write('\n')
                    log_writer.write("Failed to get response from API")
                    log_writer.write('\n')
                raise Exception("Failed to get response from API")
            with open(log_file, 'a') as log_writer:
                log_writer.write('=' * 80)
                log_writer.write('\n')
                log_writer.write('RESPONSE')
                log_writer.write('\n')
                log_writer.write('=' * 80)
                log_writer.write('\n')
                log_writer.write(completion.choices[0].message.content)
                log_writer.write('\n')

            git_diff = create_diff_from_response(
                completion.choices[0].message.content, file_contents, found_files
            )

            with open(log_file, 'a') as log_writer:
                log_writer.write('=' * 80)
                log_writer.write('\n')
                log_writer.write('PATCH')
                log_writer.write('\n')
                log_writer.write('=' * 80)
                log_writer.write('\n')
                log_writer.write(git_diff)
                log_writer.write('\n\n\n\n')

            if git_diff:
                # Write both the generation output and the diff
                with open(output_file, "a", encoding="utf-8") as f:
                    output = {
                        "instance_id": instance_id,
                        "model_name_or_path": model_name,
                        "model_patch": git_diff,
                        "temperature": temperature
                    }
                    f.write(json.dumps(output) + "\n")

                return git_diff

            temperature = min(1.0, temperature + 0.1)
        except Exception as e:
            print(f"Error in generation attempt {attempt + 1}: {e}")
            temperature = min(1.0, temperature + 0.1)

    return False


def extract_code_blocks(text):
    pattern = r"```(\w+)\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match[1] for match in matches if '<<< SEARCH' in match[1]]


def parse_edit_command(edit_string):
    """
    Parse an edit command string to extract filename and code blocks.

    Args:
        edit_string (str): The edit command string in the specified format

    Returns:
        tuple: (filename, search_code, replace_code)
    """
    try:
        lines = edit_string.rstrip().split("\n")

        if not lines:
            print("Error: Empty input")
            return "", "", ""

        filename = lines[0].replace("#", "").strip()

        search_start = -1
        replace_marker = -1
        replace_end = -1

        for i, line in enumerate(lines):
            if line.rstrip().endswith("SEARCH") and line.strip().startswith("<"):
                search_start = i
            # Look for "=======" with any number of '=' characters
            elif (
                    line.strip() and all(c == "=" for c in line.strip()) and ("==" in line)
            ):
                replace_marker = i
            # Look for ">+ REPLACE" with any number of '>' characters
            elif line.rstrip().endswith("REPLACE") and line.strip().startswith(">"):
                replace_end = i

        print(search_start, replace_marker, replace_end)

        if search_start == -1 or replace_marker == -1 or replace_end == -1:
            print("Error: Missing markers")
            return "", "", ""

        if not (search_start < replace_marker < replace_end):
            print("Error: Markers are in incorrect order")
            return "", "", ""

        search_code = "\n".join(lines[search_start + 1: replace_marker]).rstrip()
        replace_code = "\n".join(lines[replace_marker + 1: replace_end]).rstrip()

        return filename, search_code, replace_code

    except Exception as e:
        print(f"Error parsing edit command: {str(e)}")
        return "", "", ""


def apply_edit_commands(
        parsed_edit_commands, contents, files, match_fuzzy, test_each=False
):
    assert len(files) == len(contents), (
        f"Input lists to apply_edit_commands must have same length. "
        f"They have lengths: {len(files)} and {len(contents)}"
    )

    new_contents = []
    original_file_contents = ""
    for idx, file_name in enumerate(files):
        new_content = contents[idx]
        for file, original, replacement in parsed_edit_commands:
            if file_name == file:
                original_file_contents = contents[idx]
                # First try exact match
                if "\n" + original in new_content:
                    if test_each:
                        temp_new_content = new_content.replace(original, replacement)
                        print("checking an individual")

                        git_diff = fake_git_repo(
                            "playground", ["test_file.py"], [""], [temp_new_content]
                        )
                        if git_diff != "":
                            print("individual worked!")
                            new_content = temp_new_content

                    else:
                        new_content = new_content.replace(original, replacement)
                        print("Found exact match")
                elif match_fuzzy:
                    chunked = new_content.splitlines()
                    chunked_edit = original.splitlines()

                    if chunked_edit[0].strip() == "":
                        chunked_edit = chunked_edit[1:]
                    if chunked_edit[-1].strip == "":
                        chunked_edit = chunked_edit[:-1]

                    matching_line_numbers = []
                    for line in chunked_edit:
                        print(line)
                        matches = get_close_matches(
                            line.strip(),
                            [chunk.strip() for chunk in chunked],
                            n=1,
                            cutoff=0.8,
                        )
                        print(matches)
                        if matches:
                            line_numbers = [
                                i
                                for i, text in enumerate(chunked)
                                if text.strip() in matches
                            ]
                            matching_line_numbers.extend(line_numbers)

                    empty_lines = []
                    for i, line_text in enumerate(chunked):
                        if line_text.strip() == "":
                            empty_lines.append(i)

                    # Add in all the empty lines too
                    for each_line in chunked_edit:
                        if each_line.strip() == "":
                            matching_line_numbers.extend(empty_lines)
                            break

                    matched_line_numbers = find_consecutive_subset(
                        matching_line_numbers, len(chunked_edit), empty_lines
                    )
                    if matched_line_numbers:
                        replaced_indent = len(chunked[matched_line_numbers[0]]) - len(
                            chunked[matched_line_numbers[0]].lstrip()
                        )
                        replacement_indent = len(chunked_edit[0]) - len(
                            chunked_edit[0].lstrip()
                        )

                        new_replacement_text = replacement.splitlines()

                        fixed_replacement = []

                        if replacement_indent < replaced_indent:
                            for new_line in new_replacement_text:
                                fixed_replacement.append(
                                    " " * (replaced_indent - replacement_indent)
                                    + new_line
                                )
                        elif replacement_indent > replaced_indent:
                            for new_line in new_replacement_text:
                                fixed_replacement.append(
                                    new_line[(replacement_indent - replaced_indent):]
                                )
                        else:
                            fixed_replacement = new_replacement_text

                        fixed_replacement_str = "\n".join(fixed_replacement)
                        fixed_search_str = "\n".join(
                            [
                                chunked_line
                                for idx, chunked_line in enumerate(
                                new_content.splitlines()
                            )
                                if idx in matched_line_numbers
                            ]
                        )

                        new_content = new_content.replace(
                            fixed_search_str, fixed_replacement_str
                        )

        new_contents.append(new_content)

    return new_contents, original_file_contents


def fake_git_repo(repo_playground, file_paths, old_contents, new_contents) -> str:
    """create a fake git repo to obtain git diff format for multiple files"""
    assert (
            len(file_paths) == len(old_contents) == len(new_contents)
    ), f"Input lists must have same length. They have lengths: {len(file_paths)}, {len(old_contents)}, and {len(new_contents)}"

    repo_playground = os.path.join(repo_playground, f"{uuid.uuid4()}")

    assert not os.path.exists(repo_playground), f"{repo_playground} already exists"

    os.makedirs(repo_playground)
    subprocess.run(f"cd {repo_playground} && git init", shell=True)

    changed_files = []
    for file_path, old_content, new_content in zip(
            file_paths, old_contents, new_contents
    ):
        if old_content != new_content:
            # create directory if needed
            subprocess.run(
                f"mkdir -p {repo_playground}/{os.path.dirname(file_path)}", shell=True
            )
            # write old content
            with open(f"{repo_playground}/{file_path}", "w") as f:
                f.write(old_content)
            changed_files.append(file_path)

    if not changed_files:
        print("No changes were made")
        # No changes to commit, clean up and return empty string
        subprocess.run(f"rm -rf {repo_playground}", shell=True)
        return ""

    # add files to git
    changed_files_str = " ".join(changed_files)
    subprocess.run(
        f"cd {repo_playground} && git add {changed_files_str} && git commit -m 'initial commit'",
        shell=True,
    )

    # edit files with new content
    for file_path, old_content, new_content in zip(
            file_paths, old_contents, new_contents
    ):
        if old_content != new_content:
            with open(f"{repo_playground}/{file_path}", "w") as f:
                f.write(new_content)
            if not check_syntax(f"{repo_playground}/{file_path}")[0]:
                print("failed syntax check")
                with open(f"{repo_playground}/{file_path}", "w") as f:
                    f.write(old_content)

    # get git diff for changed files
    o = subprocess.run(
        f"cd {repo_playground} && git diff {changed_files_str}",
        shell=True,
        capture_output=True,
    )
    s = o.stdout.decode("utf-8")

    # remove playground
    subprocess.run(f"rm -rf {repo_playground}", shell=True)

    return s


def create_diff_from_response(
        response: str, old_contents: list, files: list
) -> Optional[str]:
    extracted_blocks = extract_code_blocks(response)
    try:
        edits = [parse_edit_command(edit_command) for edit_command in extracted_blocks]
        new_contents, _ = apply_edit_commands(edits, old_contents, files, False, False)
        git_diff = fake_git_repo("playground", files, old_contents, new_contents)
        return git_diff if git_diff.strip() else None
    except:
        return None


def check_syntax(filepath):
    """
    Check the syntax of a code file.

    Args:
        filepath (str): Path to the code file

    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    file_extension = os.path.splitext(filepath)[1].lower()

    try:
        if file_extension in [".py", ".pyw"]:
            return check_python_syntax(filepath)

        elif file_extension in [".js", ".jsx"]:
            return check_javascript_syntax(filepath)

        elif file_extension in [".sh", ".bash"]:
            return check_shell_syntax(filepath)

        else:
            return True, f"Unsupported file type: {file_extension}"

    except Exception as e:
        return False, f"Error during syntax check: {str(e)}"


def check_python_syntax(filepath, timeout_seconds=30):
    """Check Python syntax using py_compile with timeout"""
    try:
        command = f'python -m py_compile "{filepath}"'
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout_seconds
        )

        if result.returncode == 0:
            return True, "Syntax is valid"
        else:
            return False, f"Syntax error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, f"Process timed out after {timeout_seconds} seconds"
    except Exception as e:
        return False, f"Error checking Python syntax: {str(e)}"


def check_javascript_syntax(filepath):
    """Check JavaScript/JSX syntax using tree-sitter"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        try:
            # Initialize tree-sitter parser with JavaScript language
            JS_LANGUAGE = Language(tree_sitter_javascript.language())
            parser = Parser(JS_LANGUAGE)

            # Parse the code
            tree = parser.parse(bytes(code, "utf8"))

            # If there are syntax errors, tree-sitter will include ERROR nodes
            has_errors = any(node.type == "ERROR" for node in tree.root_node.children)

            if has_errors:
                return False, "Syntax error detected in the code"
            return True, "Syntax is valid"

        except Exception as e:
            return False, f"Syntax error: {str(e)}"

    except ImportError:
        return (
            False,
            "tree-sitter-javascript is not installed. Install it using: pip install tree-sitter-javascript",
        )
    except Exception as e:
        return False, f"Error checking JavaScript/JSX syntax: {str(e)}"


def check_shell_syntax(filepath):
    """Check shell script syntax using bash -n"""
    try:
        if platform.system() == "Windows":
            shell_cmd = "bash"
        else:
            shell_cmd = "bash"

        command = f'{shell_cmd} -n "{filepath}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            return True, "Syntax is valid"
        else:
            return False, f"Syntax error: {result.stderr}"
    except Exception as e:
        return False, f"Error checking shell script syntax: {str(e)}"


def find_consecutive_subset(numbers, x, no_start):
    numbers = sorted(list(set(numbers)))

    for i in range(len(numbers) - x + 1):
        sequence = numbers[i: i + x]

        if sequence[0] in no_start or sequence[-1] in no_start:
            continue

        if sequence == list(range(sequence[0], sequence[0] + x)):
            return sequence

    return None


evaluate_single_patch_cmd = """
python -m swebench.harness.run_evaluation \
    --predictions_path {prediction_path} \
    --max_workers 1 \
    --instance_ids {instance_id} \
    --run_id {run_id}
"""


def evaluate_patch(instance_id,
                   problem_statement,
                   patch_diff,
                   patch_prediction_file,
                   repair_model,
                   evaluate_model,
                   base_url,
                   api_key,
                   run_id,
                   swe_root,
                   log_file):
    cmd = evaluate_single_patch_cmd.format(instance_id=instance_id, prediction_path=patch_prediction_file,
                                           run_id=run_id)
    cmd = f"cd {swe_root} && {cmd}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'evaluate patch for {instance_id}', result)

    test_output_path = f'{swe_root}/logs/run_evaluation/{run_id}/{repair_model}/{instance_id}/test_output.txt'
    if not os.path.exists(test_output_path):
        return 0, False, "no test log"
    test_output_content = open(test_output_path, 'r').read()
    test_output_content = \
        test_output_content.split("Start Test Output")[-1].split("End Test Output")[0]

    test_report_path = f'{swe_root}/logs/run_evaluation/{run_id}/{repair_model}/{instance_id}/report.json'
    test_report_json = json.load(open(test_report_path, 'r'))
    is_resolved = test_report_json[instance_id]['resolved']
    FAIL_TO_PASS_SUCCESS = test_report_json[instance_id]['tests_status']['FAIL_TO_PASS']['success']
    FAIL_TO_PASS_FAIL = test_report_json[instance_id]['tests_status']['FAIL_TO_PASS']['failure']
    PASS_TO_PASS_SUCCESS = test_report_json[instance_id]['tests_status']['PASS_TO_PASS']['success']
    PASS_TO_PASS_FAIL = test_report_json[instance_id]['tests_status']['PASS_TO_PASS']['failure']
    # FAIL_TO_FAIL_SUCCESS = test_report_json[instance_id]['tests_status']['FAIL_TO_FAIL']['success']
    # FAIL_TO_FAIL_FAIL = test_report_json[instance_id]['tests_status']['FAIL_TO_FAIL']['failure']
    # PASS_TO_FAIL_SUCCESS = test_report_json[instance_id]['tests_status']['PASS_TO_FAIL']['success']
    # PASS_TO_FAIL_FAIL = test_report_json[instance_id]['tests_status']['PASS_TO_FAIL']['failure']

    if len(FAIL_TO_PASS_SUCCESS) + len(FAIL_TO_PASS_FAIL) == 0:
        FAIL_TO_PASS_SCORE = 1
    else:
        FAIL_TO_PASS_SCORE = len(FAIL_TO_PASS_SUCCESS) / (len(FAIL_TO_PASS_SUCCESS) + len(FAIL_TO_PASS_FAIL))

    if len(PASS_TO_PASS_SUCCESS) + len(PASS_TO_PASS_FAIL) == 0:
        PASS_TO_PASS_SCORE = 1
    else:
        PASS_TO_PASS_SCORE = len(PASS_TO_PASS_SUCCESS) / (len(PASS_TO_PASS_SUCCESS) + len(PASS_TO_PASS_FAIL))

    if FAIL_TO_PASS_SCORE == 1 and PASS_TO_PASS_SCORE == 1:
        # 成功修复
        assert is_resolved == True, "resolved not equal to tests_status"
        return 100, True, test_output_content

    evaluate_prompt = PATCH_EVALUATE_PROMPT.format(
        problem_statement=problem_statement,
        partial_patch=patch_diff,
        test_report=test_output_content
    )
    evaluate_config = {
        "model": evaluate_model,
        "messages": [{"role": "user", "content": evaluate_prompt}],
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 16000,
    }
    completion = request_chatgpt_engine(evaluate_config, base_url=base_url, api_key=api_key)
    if completion is None:
        raise Exception("Failed to get response from API")

    content = completion.choices[0].message.content

    score_list = content.split()
    score = 0
    for i in range(len(score_list)):
        score_list[i] = score_list[i].replace("**", "").replace(".", "").replace(",", "").strip()
        if score_list[i].isdigit():
            score = int(score_list[i])
            break
    if score <= 0:
        score = 0
    if score >= 100:
        score = 100
    if 1 >= score > 0:
        score *= 10
    assert is_resolved == False, "resolved not equal to tests_status"

    final_score = score * 0.5 + (FAIL_TO_PASS_SCORE + PASS_TO_PASS_SCORE) * 25
    with open(log_file, 'a') as log_writer:
        log_writer.write('=' * 80)
        log_writer.write('\n')
        log_writer.write('SCORE')
        log_writer.write('\n')
        log_writer.write('=' * 80)
        log_writer.write('\n')
        log_writer.write(f"Score = {final_score}")
        log_writer.write('\n\n\n\n')
    return final_score, False, test_output_content


if __name__ == '__main__':
    output_file = './swe_lite_patch.jsonl'
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = "sk-e346078d76f546c2ab04f0f008126a91"
    with open('swe_lite_loc_with_context.jsonl', 'r') as f:
        line1 = f.readlines()[0]
    sample = json.loads(line1)

    problem_statement = sample['problem_statement']
    instance_id = sample['instance_id']
    buggy_locs = sample['buggy_code']
    found_files = list(sample['buggy_files'].keys())
    file_contents = list(sample['buggy_files'].values())

    prompt = AGENTLESS_PROMPT.format(problem_statement=problem_statement, retrieval=json.dumps(buggy_locs, indent=4))
    generate_with_retries(instance_id,
                          prompt,
                          output_file,
                          file_contents=file_contents,
                          found_files=found_files,
                          model_name='qwen3-coder-plus',
                          base_url=base_url,
                          api_key=api_key)
