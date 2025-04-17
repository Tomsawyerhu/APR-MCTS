import os
import re

from framework import get_bug_details
from neo4j_client import *
from utils import run_bash


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


client = Neo4jClient()

proj = "Chart"
bid = 5
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

# 2. 类内上下文 类变量，类函数列表
class_query = ClAZZ_QUERY(class_full_qualified_name)
class_node = client.execute_query(class_query)[0]['n']
print("缺陷类")
print(class_node)
field_list_query = FIELD_LIST_QUERY(class_full_qualified_name)
field_nodes = [x['n'] for x in client.execute_query(field_list_query)]
print("类变量")
print(field_nodes)
method_list_query = METHOD_LIST_QUERY(class_full_qualified_name)
method_nodes = [x['n'] for x in client.execute_query(method_list_query)]
print("方法列表")
print(method_nodes)

# 3. 跨类上下文
# 3.1 方法局部变量对应的类，入参对应的类
params_type = buggy_method_node['params']
multiclass_query = ClAZZ_QUERY(params_type)
class_nodes = [x['n'] for x in client.execute_query(multiclass_query)]
print("缺陷方法入参对应的类")
print(class_nodes)
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
print(external_invoke_method_nodes)  # 跨类调用
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
print(import_util_class_nodes)
