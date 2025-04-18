import os
import re
import json

import framework
from framework import get_bug_details, Bug
from gpt import generate, generate_patches
from neo4j_client import *
from utils import run_bash

client = Neo4jClient()


def get_full_qualified_name_from_path(path: str):
    path_split = path.split(os.path.sep)
    if "org" in path_split:
        idx = path_split.index("org")
        qualified_name_split = path_split[idx:]
        qualified_name_split[-1] = qualified_name_split[-1].replace(".java", "")
        return ".".join(qualified_name_split)
    return None


def get_simple_name_from_path(path: str):
    path_split = path.split(os.path.sep)
    if path_split[-1].endswith(".java"):
        return path_split[-1].replace(".java", "")
    return None


def get_method_name_from_code(code: str):
    # 定义正则表达式
    pattern = r'\w+\s+(\w+)\('

    # 使用正则表达式匹配函数名
    match = re.search(pattern, code)
    if match:
        return match.group(1)  # 返回捕获组中的函数名
    else:
        raise ValueError("未找到函数定义！")


def is_util_class(name: str):
    keywords = ['util', 'tool', 'helper', 'manager', 'factory', 'builder', 'converter']
    for keyword in keywords:
        if keyword in name.lower():
            return True
    return False


def get_field_info(field_node):
    return {"comment": field_node['comment'], "content": field_node['content']}


def remove_java_comments(code: str) -> str:
    """
    去除 Java 代码中的注释（单行注释、多行注释和文档注释）。

    :param code: 输入的 Java 源代码字符串
    :return: 去除注释后的代码字符串
    """
    # 正则表达式匹配多行注释（/* ... */）
    multiline_pattern = r'/\*.*?\*/'

    # 正则表达式匹配单行注释（// ...）
    singleline_pattern = r'(?<![:\'"/])//.*?$'

    # 正则表达式匹配文档注释（/** ... */ 或 /* ... */ 中的内容）
    doc_comment_pattern = r'/\*\*.*?\*/'

    # 去除多行注释和文档注释
    code = re.sub(multiline_pattern, '', code, flags=re.DOTALL)
    code = re.sub(doc_comment_pattern, '', code, flags=re.DOTALL)

    # 去除单行注释
    code = re.sub(singleline_pattern, '', code, flags=re.MULTILINE)

    # 去除多余的空白行
    code = '\n'.join(line.rstrip() for line in code.splitlines() if line.strip())

    return code


def format_context(context_dict, include_external=False):
    context_str = ""
    # 类内上下文只取类变量和类方法列表
    internal_context = context_dict['internal_context']
    context_str += "Fields of the buggy class:\n"
    for i, field_info in enumerate(internal_context['field']):
        context_str += f"Field{i + 1}\n - Definition: {remove_java_comments(field_info['content'])}\n - Comment: {field_info['comment']}\n"
    context_str += "==================================\n"
    context_str += "Methods of the buggy class:\n"
    for i, method_info in enumerate(internal_context['method']):
        context_str += f"Method{i + 1}\n - Signature: {method_info['signature']}\n - Summary: {method_info['summary']}\n"
    context_str += "==================================\n"

    external_context = context_dict['external_context']
    # 类间上下文
    if include_external:
        context_str += "Classes relevant to the buggy method:\n"
        for i, relevant_class in enumerate(external_context['relevant_class']):
            context_str += f"Class{i + 1}: {relevant_class['name']}"
            context_str += f"Class{i + 1} has following methods:\n"
            for j, method_info in enumerate(relevant_class['method']):
                context_str += f"Method{j + 1}\n - Signature: {method_info['signature']}\n - Summary: {method_info['summary']}\n"

        context_str += "==================================\n"
        context_str += "Utility Classes which may contain useful methods:\n"
        for i, relevant_class in enumerate(external_context['util_class']):
            context_str += f"Class{i + 1}: {relevant_class['name']}"
            context_str += f"Class{i + 1} has following methods:\n"
            for j, method_info in enumerate(relevant_class['method']):
                context_str += f"Method{j + 1}\n - Signature: {method_info['signature']}\n - Summary: {method_info['summary']}\n"
        context_str += "==================================\n"

        context_str += "The buggy method also has external function calls, listed as follows:\n"
        for i, external_method in enumerate(external_context['external_invoke_method']):
            context_str += "\n"
            context_str += external_method
            context_str += "\n"
        context_str += "==================================\n"
    return context_str


method_summary_prompt = ("Generate a concise and professional summary for the given function. The summary should "
                         "include the following details:\n"
                         "Purpose: A brief description of what the function does.\n"
                         "Parameters: List and explain the inputs (if any) the function accepts, including their types "
                         "and roles.\n"
                         "Return Value: Describe the output of the function, including its type and significance.\n"
                         "Key Logic: Highlight the main operations or logic implemented within the function.")

method_brief_summary_prompt = ("Generate a one-sentence summary (less than 20 words) for the given function, "
                               "describing its purpose.")


def construct_initial_message(bug: Bug, mode: str, context: str):
    cot_prompt = ("Before you give the final answer, let's think step by step. You need to explain where bug happens "
                  "and how your answer can avoid it.")
    context_str = f"Here are contexts which may help you fix the bug.\n{context}"

    if mode == "SL":
        prompt_header = f"""The following code contains a buggy line that has been removed.\n```java\n{bug.masked_code}\n```
This was the original buggy line which was removed by the infill location:
```java\n{bug.buggy_lines}\n```\nTest cases look like:```java\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"""
        prompt_footer = f"Please provide the correct line at the infill location, only single line is allowed. Your answer should begin with ```java"

    elif mode == "SH":
        prompt_header = f"""The following code contains a buggy hunk that has been removed.\n```java\n{bug.masked_code}\n```
This was the original buggy hunk which was removed by the infill location:
```java\n{bug.buggy_lines}\n```\nTest cases look like:```java\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"""
        prompt_footer = f"Please provide the correct hunk at the infill location, only single hunk is allowed. Your answer should begin with ```java"


    elif mode == "SF":
        prompt_header = f"The following code contains a bug\n```java\n{bug.code}\n```\nTest cases look like:```java\n{bug.extract_test_code}```\nThe code fails with the following test error:\n```\n{bug.failing_tests}\n```"
        prompt_footer = f"Please provide the correct function, starting with ```java"

    initial_prompt_message = f"""{prompt_header}
{cot_prompt}\n
{context_str}\n
{prompt_footer}"""

    return initial_prompt_message


def get_method_brief_info(method_node):
    """
    简短摘要，包括method签名和method的简要介绍(少于20字)
    :param method_node:
    :return:
    """
    method_content = method_node['content']
    prompt = [{
        'role': 'user',
        'content': method_brief_summary_prompt + f"\nThe method is:\n```java\n{method_content}\n```"
    }]
    method_brief_summary = generate(prompt)
    print(method_brief_summary)
    return {
        "signature": method_node['signature'],
        "summary": method_brief_summary
    }


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


def construct_context_and_save(proj="Chart", bid=5, output_dir="./context"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    bug = get_bug_details(proj, bid)
    path = run_bash("get_source_code_file_path", proj, bid).stdout  # 缺陷文件路径
    bug_start_line = int(run_bash("get_first_change_line_count_number", proj, bid).stdout.split(",")[0])  # 缺陷开始行
    bug_end_line = int(run_bash("get_last_change_line_count_number", proj, bid).stdout.split(",")[0])  # 缺陷结束行
    class_full_qualified_name = get_full_qualified_name_from_path(path)
    # simple_name = get_simple_name_from_path(path)

    method_name = get_method_name_from_code(bug.code)
    method_full_qualified_name = class_full_qualified_name + "." + method_name

    # 1. 找到method节点
    script = METHOD_QUERY(method_full_qualified_name, method_name)
    results = client.execute_query(script)
    buggy_method_node = None
    if len(results) == 0:
        # 没找到函数节点
        raise Exception("buggy method not found in KG")
    elif len(results) == 1:
        buggy_method_node = results[0]['n']
    else:
        # 同名函数，根据行数匹配
        for result in results:
            if result['n']["start_line"] < bug_start_line and result['n']["end_line"] > bug_end_line:
                buggy_method_node = result['n']
                break
        raise Exception("buggy method not found in KG")

    internal_context = {}
    # 2. 类内上下文 类变量，类函数列表
    class_query = ClAZZ_QUERY(class_full_qualified_name)
    class_node = client.execute_query(class_query)[0]['n']
    print("缺陷类")
    print(class_node)
    field_list_query = FIELD_LIST_QUERY(class_full_qualified_name)
    field_nodes = [x['n'] for x in client.execute_query(field_list_query)]
    print("类变量")

    field_info = [get_field_info(x) for x in field_nodes]
    internal_context['field'] = field_info
    print(field_info)
    method_list_query = METHOD_LIST_QUERY(class_full_qualified_name)
    method_nodes = [x['n'] for x in client.execute_query(method_list_query)]
    print("方法列表")
    method_info = [get_method_brief_info(x) for x in method_nodes]
    internal_context['method'] = method_info
    print(method_info)

    # 3. 跨类上下文
    external_context = {
        'relevant_class': [],
        'util_class': [],
        'external_invoke_method': []
    }
    # 3.1 方法局部变量对应的类，入参对应的类
    params_type = buggy_method_node['params']
    param_multiclass_query = ClAZZ_QUERY(params_type)
    param_class_nodes = [x['n'] for x in client.execute_query(param_multiclass_query)]

    local_variable_query = LOCAL_VARIABLE_QUERY(buggy_method_node['signature'])
    local_variable_nodes = [x['n'] for x in client.execute_query(local_variable_query)]
    local_variable_classes = [x['data_type'] for x in local_variable_nodes]
    local_variable_multiclass_query = ClAZZ_QUERY(local_variable_classes)
    local_variable_class_nodes = [x['n'] for x in client.execute_query(local_variable_multiclass_query)]

    class_nodes = param_class_nodes + local_variable_class_nodes
    # Step 1: Filter out the current class
    class_nodes = [x for x in class_nodes if x['full_qualified_name'] != class_node['full_qualified_name']]
    # Step 2: Deduplicate by `full_qualified_name`
    seen = set()
    class_nodes = [x for x in class_nodes if
                   x['full_qualified_name'] not in seen and not seen.add(x['full_qualified_name'])]

    for class_node in class_nodes:
        data = {}
        class_node_name = class_node["full_qualified_name"]
        data['name'] = class_node_name
        method_list_query = METHOD_LIST_QUERY(class_node_name)
        method_nodes = [x['n'] for x in client.execute_query(method_list_query)]
        method_info = [get_method_brief_info(x) for x in method_nodes]
        data['method'] = method_info
        external_context['relevant_class'] = external_context['relevant_class'] + [data]

    # 3.2 跨类函数调用
    method_invoke_query = METHOD2METHOD_INVOKE_QUERY(buggy_method_node['signature'])
    invoke_method_nodes = [x['n'] for x in client.execute_query(method_invoke_query)]
    invoke_method_nodes = [x for x in invoke_method_nodes if "test" not in x['name']]  # 过滤testmethod
    # 区分类内部调用/跨类调用
    internal_invoke_method_nodes = [x for x in invoke_method_nodes if
                                    x['full_qualified_name'].startswith(class_node['full_qualified_name'])]
    external_invoke_method_nodes = [x for x in invoke_method_nodes if
                                    not x['full_qualified_name'].startswith(class_node['full_qualified_name'])]
    print("类内调用")
    print(internal_invoke_method_nodes)  # 类内调用

    print("跨类调用")
    print(external_invoke_method_nodes)  # 跨类调用 提取跨类调用的方法体

    external_invoke_seen = set()
    external_invoke_methods = []
    for external_invoke_method_node in external_invoke_method_nodes:
        method_query = UNIQUE_METHOD_QUERY(external_invoke_method_node['full_qualified_name'],
                                           external_invoke_method_node['signature'])
        unique_method_node = client.execute_query(method_query)[0]['n']
        # 跳过构造函数
        if unique_method_node['type'].lower() == "constructor":
            continue
        if (
                unique_method_node['full_qualified_name'],
                external_invoke_method_node['signature']) in external_invoke_seen:
            continue
        external_invoke_seen.add((unique_method_node['full_qualified_name'], external_invoke_method_node['signature']))
        external_invoke_methods.append(unique_method_node['content'])
    external_context['external_invoke_method'] = external_invoke_methods

    # 3.3 父类
    parent_class_query = PARENT_CLASS_QUERY(class_full_qualified_name)
    result = client.execute_query(parent_class_query)
    if len(result) == 0:
        # 没有父类
        parent_class_node = None
    else:
        parent_class_node = result[0]['n']
    print("父类")
    print(parent_class_node)

    # 3.4 import的工具类
    import_query = IMPORT_QUERY(class_full_qualified_name)
    import_class_nodes = [x['n'] for x in client.execute_query(import_query)]
    print("Import的类")
    print([x['name'] for x in import_class_nodes])
    # 筛选工具类
    import_util_class_nodes = [x for x in import_class_nodes if is_util_class(x['name'])]
    print("Import的工具类")

    relevant_class_names = [x['name'] for x in external_context['relevant_class']]
    for import_util_class_node in import_util_class_nodes:
        # 如果在相关类中，不需要二次检索
        if import_util_class_node['full_qualified_name'] in relevant_class_names:
            continue
        data = {}
        import_util_class_name = import_util_class_node["full_qualified_name"]
        data['name'] = import_util_class_name
        method_list_query = METHOD_LIST_QUERY(import_util_class_name)
        method_nodes = [x['n'] for x in client.execute_query(method_list_query)]
        method_info = [get_method_brief_info(x) for x in method_nodes]
        data['method'] = method_info
        external_context['util_class'] = external_context['util_class'] + [data]

    # 将内部上下文和外部上下文合并为一个字典
    context = {
        "project": proj,
        "bugid": bid,
        "internal_context": internal_context,
        "external_context": external_context
    }

    # 写入 JSON 文件
    output_file = f"{output_dir}/context_{proj}_{bid}.json"  # 输出文件名
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=4)

    print(f"Context has been written to {output_file}")


def repair(proj="Chart", bid=5, context_dir="./context", model="gpt-4o-mini", result_file="./context_repair_result"
                                                                                          ".jsonl", num_patches=10):
    context_file = f"{context_dir}/context_{proj}_{bid}.json"  # 输出文件名
    if not os.path.exists(context_file):
        construct_context_and_save(proj, bid, context_dir)
    context_dict = dict(json.load(open(context_file, 'r')))
    context_str = format_context(context_dict, True)
    bug = get_bug_details(proj, bid)
    modes = list(bug.bug_type.split())
    mode = modes[0]
    prompt = construct_initial_message(bug, mode, context_str)
    prompt = [{
        "role": "user",
        "content": prompt
    }]
    print(prompt)
    responses = generate_patches(prompt, model, num_samples=num_patches)
    patches = [extract_patch_from_response(response, mode) for response in responses]
    for i, patch in enumerate(patches):
        test_result, result_reason, patch_diff = framework.validate_patch(bug=bug, proposed_patch=patch, mode=mode)
        if test_result == "PASS":
            print(f"Proposed patch of {bug.project}-{bug.bug_id} ({mode}) patch passed all tests")
            with open(result_file, "a") as f:
                record = {
                    "project": bug.project,
                    "bug_id": bug.bug_id,
                    "eval": "PASS",
                    "attempt": i + 1,
                    "mode": mode,
                    "patch": patch,
                    "diff": patch_diff,
                }
                f.write(json.dumps(record) + "\n")
            return
        elif result_reason == bug.test_error_message:
            record = {
                "project": bug.project,
                "bug_id": bug.bug_id,
                "eval": result_reason,
                "attempt": i + 1,
                "mode": mode,
                "patch": patch,
                "diff": "",
            }
            with open(result_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"Proposed patch of {bug.project}-{bug.bug_id} ({mode}) failed with same error message as original bug")
        else:
            record = {
                "project": bug.project,
                "bug_id": bug.bug_id,
                "eval": result_reason,
                "attempt": i + 1,
                "mode": mode,
                "patch": patch,
                "diff": "",
            }
            with open(result_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"Proposed patch of {bug.project}-{bug.bug_id} ({mode}) failed with a different error message than original bug")

    with open(result_file, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == '__main__':
    repair()
