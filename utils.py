from subprocess import PIPE, run

import javalang

tmp_dir = "/mnt/data/hhc/APRMcts/tmp2"
shell_script_folder = "/mnt/data/hhc/APRMcts/scripts"
script_name = "defects4j"
java_home = "/usr/lib/jvm/java-8-openjdk-amd64"
d4j_path = "/mnt/data/hhc/defects4j"


def run_bash(function, project, bug_id, extra_arg1=None, extra_arg2=None):
    work_dir = f"{tmp_dir}/{project}-{bug_id}"
    command = ['bash', f'{shell_script_folder}/{script_name}.sh', function, f"{project}", f"{bug_id}", f"{work_dir}",
               f"{java_home}", f"{d4j_path}", f"{extra_arg1}", f"{extra_arg2}"]
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if len(result.stdout) > 0:
        if result.stdout[-1] == "\n":
            result.stdout = result.stdout[:-1]
    return result


def get_test_names(failing_tests):
    lines = failing_tests.split("\n")
    lines = [x for x in lines if x.startswith("---")]
    return [x.split(":")[-1].strip() for x in lines]


def extract_method(java_code, method_name):
    # 解析 Java 代码
    tree = javalang.parse.parse(java_code)
    java_code = java_code.split("\n")

    # 遍历所有的类型（类、接口等）
    for type_declaration in tree.types:
        if isinstance(type_declaration, javalang.tree.ClassDeclaration):
            # 遍历类中的所有方法
            for i, method in enumerate(type_declaration.methods):
                if method.name == method_name:
                    if i + 1 >= len(type_declaration.methods):
                        body = "\n".join(
                            java_code[method.position[0] - 1:])

                    else:
                        body = "\n".join(
                            java_code[method.position[0] - 1:type_declaration.methods[i + 1].position[0]])
                        body = body[:body.rfind("}") + 1]
                    if method.documentation is not None:
                        body = method.documentation + "\n" + body
                    return body.strip()
    return ""


def make_failing_tests_short(failing_tests, test_names):
    lines = failing_tests.split("\n")
    result = ""
    for i in range(len(lines)):
        if lines[i].startswith("---") and lines[i].split("::")[-1].strip() in test_names:
            result += "\n".join(lines[i:i + 20]) + "\n"
    return result

def write_line_to_txt(file_path, line_content):
    """
    向指定的 txt 文件写入一行内容。
    
    参数：
        file_path (str): 目标文件路径（如 "example.txt"）。
        line_content (str): 要写入的内容（字符串格式）。
    """
    try:
        # 使用追加模式 ('a') 打开文件，如果文件不存在会自动创建
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(line_content + '\n')  # 写入内容并换行
        print(f"成功写入: {line_content}")
    except Exception as e:
        print(f"写入文件时出错: {e}")
