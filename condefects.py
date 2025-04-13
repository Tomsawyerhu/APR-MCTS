import re
import signal
import subprocess
import json
import os
import time
import tqdm
import uuid
import jsonlines

condefects_dir = "/root/autodl-tmp/ConDefects-main"
condefects_code_dir = condefects_dir + "/Code"
condefects_test_dir = condefects_dir + "/Test"
condefects_coverage_dir = condefects_dir + "/Coverage"
condefects_tmp_dir = condefects_dir + "/Tmp"

if not os.path.exists(condefects_tmp_dir):
    os.makedirs(condefects_tmp_dir)


class TimeoutException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def terminate_process(process):
    """
    Terminate a subprocess and its child processes to ensure cleanup.
    """
    try:
        # Send SIGTERM to the process group (Unix/Linux only)
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass  # Process already terminated
    finally:
        # Ensure the process is cleaned up
        process.terminate()
        process.wait()


def run_command(command, cwd="", timeout=10):
    """
    Helper function to run a shell command and return its output.
    Ensures cleanup of child processes in case of timeout.
    """
    try:
        # Start the subprocess in a new process group
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            preexec_fn=os.setsid  # Create a new process group for the subprocess
        )

        # Wait for the process to complete with a timeout
        stdout, stderr = process.communicate(timeout=timeout)

        # Check if the process exited successfully
        if process.returncode != 0:
            print(f"Error running command: {command}")
            print(f"Error message: {stderr.strip()}")
            return None

        return stdout.strip()

    except subprocess.TimeoutExpired:
        # Terminate the process group to ensure cleanup
        terminate_process(process)
        raise TimeoutException(f"Command timed out after {timeout} seconds: {command}")

    except Exception as e:
        print(f"Unexpected error running command: {command}")
        print(f"Error message: {str(e)}")
        return None


def parse_tasks(output):
    """
    Parse the output of `--list-tasks` command to extract task names.
    """
    # Assuming the output is a list of tasks, one per line
    return [line.strip() for line in output.splitlines() if line.strip()]


def get_all_tasks():
    """
    Get all Python projects and their corresponding program IDs.
    """
    list_tasks_command = "python3 ConDefects.py info --list-tasks"
    tasks_output = run_command(list_tasks_command, cwd=condefects_dir)
    if not tasks_output:
        print("Failed to retrieve tasks.")
        return {}

    tasks = parse_tasks(tasks_output)
    return tasks


def judge_if_python_task(task_id=""):
    return os.path.exists(os.path.sep.join([condefects_code_dir, str(task_id), "Python"]))


def judge_if_java_task(task_id=""):
    return os.path.exists(os.path.sep.join([condefects_code_dir, str(task_id), "Java"]))


def get_all_python_tasks():
    all_tasks = get_all_tasks()
    return [x for x in all_tasks if judge_if_python_task(x)]


def get_all_java_tasks():
    all_tasks = get_all_tasks()
    return [x for x in all_tasks if judge_if_java_task(x)]


def get_python_programs(task_id=""):
    dir = os.path.sep.join([condefects_code_dir, task_id, "Python"])
    return os.listdir(dir)


def checkout_python_task(task_id=""):
    if check_if_checkout_python(task_id):
        rm_command = f"rm -rf {os.path.sep.join([condefects_dir, str(task_id)])}"
        run_command(rm_command, cwd=condefects_dir)
    checkout_command = f"python3 ConDefects.py checkout -w {condefects_dir} -l python -s {task_id}"
    run_command(checkout_command, cwd=condefects_dir)


def check_if_checkout_python(task_id=""):
    return os.path.exists(os.path.sep.join([condefects_dir, str(task_id), "Python"]))


def parse_programs(output):
    return [x.split()[0] for x in output.split("\n")[1:]]


def get_python_task_programs(task_id=""):
    command = f"python3 ConDefects.py info --programs --task {task_id} --language python"
    output = run_command(command, cwd=condefects_dir)
    return parse_programs(output)


def parse_testcases(output):
    return [x.split()[0] for x in output.split("\n")[1:]]


def get_python_task_testcases(task_id=""):
    command = f"python3 ConDefects.py info --test-cases --task {task_id} -l python"
    output = run_command(command, cwd=condefects_dir)
    return parse_testcases(output)


def get_python_task_coverage(task_id=""):
    if not check_if_checkout_python(task_id):
        checkout_python_task(task_id)
    checkout_path = os.path.sep.join([condefects_dir, str(task_id)])
    # command = f"python3 ConDefects.py coverage -w {checkout_path} -o {coverage_path}"
    command = f"python3 ConDefects.py coverage -w {condefects_dir} -s {task_id} -o {condefects_coverage_dir}"
    run_command(command, cwd=condefects_dir, timeout=100)


def parse_python_task_coverage(task_id=""):
    coverage_result = {}

    def read_cov_matrix(file):
        cov_matrix = []
        with open(file, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                cov_matrix.append([int(x) for x in json.loads(line.strip())])
        return cov_matrix

    def read_test_result(file):
        result = []
        with open(file, 'r') as f:
            for line in f:
                if line.strip() == "False":
                    result.append(False)
                if line.strip() == "True":
                    result.append(True)
        return result

    def read_test_list(file):
        test_list = []
        with open(file, 'r') as f:
            for line in f:
                if line.strip().endswith(".txt"):
                    test_list.append(line.strip())
        return test_list

    if not os.path.exists(os.path.sep.join([condefects_coverage_dir, str(task_id), "Python"])):
        get_python_task_coverage(task_id)
    programs = get_python_task_programs(task_id)

    for program in programs:
        cov_matrix_file = os.path.sep.join([condefects_coverage_dir, str(task_id), "Python", program, "covMatrix.txt"])
        test_result_file = os.path.sep.join([condefects_coverage_dir, str(task_id), "Python", program, "results.txt"])
        test_list_file = os.path.sep.join([condefects_coverage_dir, str(task_id), "Python", program, "testList.txt"])
        coverage_result[program] = {
            "cov_matrix": read_cov_matrix(cov_matrix_file),
            "test_result": read_test_result(test_result_file),
            "test_list": read_test_list(test_list_file)
        }
    return coverage_result


def parse_test_result(output_file: str):
    with open(output_file, 'r') as f:
        result_dict = json.load(f)
        # "is_passed": is_passed,
        # "test_output": test_output,
        # "groundtruth_output": groundtruth_output
        new_result = {}
        for k in result_dict.keys():
            new_result[k] = {
                "correct_result": result_dict[k]["groundtruth_output"],
                "test_result": result_dict[k]["test_output"],
                "is_test_passed": result_dict[k]["is_passed"]
            }
    f.close()

    return new_result


def run_python_test(task_id="", program_id="", test_list=[]):
    """

    :param task_id:
    :param test_list:
    :return: 测试用例列表,标准答案,测试结果,是否通过
    """
    if not isinstance(test_list, list) or len(test_list) == 0:
        raise Exception("test num must >=1")
    test_str = ' '.join(test_list)
    # 输出文件
    output_file = f"{condefects_tmp_dir}/{task_id}-{program_id}-{uuid.uuid4()}.json"
    command = f"python3 ConDefects.py runsingle -w {condefects_dir}  -s {task_id}  -t {test_str} --program-id {program_id} --result-path {output_file}"
    print(f"execute {command}")
    try:
        run_command(command, cwd=condefects_dir, timeout=100)
    except TimeoutException:
        return "timeout"
    test_result = parse_test_result(output_file)
    new_result = {}
    new_result[program_id] = {
        "correct_results": [test_result[test_name]["correct_result"] for test_name in test_list],
        "test_results": [test_result[test_name]["test_result"] for test_name in test_list],
        "is_test_passed": [test_result[test_name]["is_test_passed"] for test_name in test_list],
        "test_list": test_list
    }

    return new_result


def read_difficulty():
    difficulty = {}
    difficulty_file = os.path.sep.join([condefects_dir, "difficulty.txt"])
    with open(difficulty_file, 'r') as f:
        for line in f.readlines():
            items = line.strip().split()
            difficulty[items[0]] = items[1]
    return difficulty


def record_python_task_meta(output_file=""):
    existed_task_ids = set()
    existed_ids = []
    if os.path.exists(output_file):
        with jsonlines.open(output_file, 'r') as reader:
            for line in reader:
                existed_ids.append((line['task_id'], line['program_id']))
                existed_task_ids.add(line['task_id'])
    python_tasks = get_all_python_tasks()
    difficulty = read_difficulty()
    for python_task in tqdm.tqdm(python_tasks):
        if python_task in existed_task_ids:
            continue
        program_ids = get_python_programs(python_task)
        testlist = get_all_python_tests(python_task)

        for program_id in program_ids:
            start_time = time.time()
            try:
                result = run_python_test(task_id=python_task,program_id=program_id,test_list=testlist)
            except:
                continue
            end_time = time.time()

            if (python_task, program_id) in existed_ids:
                continue
            meta_info = {
                "task_id": python_task,
                "program_id": program_id,
                "test_list": testlist,
                "test_result": result[program_id]['is_test_passed'],
                "difficulty": difficulty[python_task],
                "time": end_time - start_time
            }
            with jsonlines.open(output_file, 'a') as writter:
                writter.write(meta_info)


def record_task_to_programs():
    tasks = get_all_python_tasks()
    programs = {}
    with open("./programs.txt", 'w') as f:
        for task in tasks:
            programs[task] = get_python_programs(task)
            f.write(f"{task}: {str(programs[task])}")
            f.write("\n")

    f.close()


def incorporate_date_meta(meta_file="", date_file="", output_file=""):
    # read task date
    task_date = {}
    with open(date_file, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            items = line.strip().split()
            task_date[items[0]] = items[1]

    with jsonlines.open(meta_file, 'r') as reader, jsonlines.open(output_file, 'w') as writer:
        for line in reader:
            line["date"] = task_date.get(line['task_id'], "")
            writer.write(line)


def read_python_program_code(task_id="", program_id=""):
    program_path = os.path.sep.join([condefects_code_dir, task_id, "Python", program_id])
    correct_version_path = os.path.sep.join([program_path, "correctVersion.py"])
    fault_location_path = os.path.sep.join([program_path, "faultLocation.txt"])
    faulty_version_path = os.path.sep.join([program_path, "faultyVersion.py"])
    if not os.path.exists(correct_version_path) or not os.path.exists(fault_location_path) or not os.path.exists(
            faulty_version_path):
        raise Exception(f"task {task_id} program {program_id} not complete")
    correct_code = str(open(correct_version_path, 'r').read())
    buggy_code = str(open(faulty_version_path, 'r').read())
    bug_location = [int(x) for x in open(fault_location_path, 'r').readlines() if len(x.strip()) > 0]
    return buggy_code, correct_code, bug_location


def get_all_python_tests(task_id=""):
    test_list = []
    dir1, dir2 = task_id.split("_")[0], task_id.split("_")[1].capitalize()
    in_path = os.path.sep.join([condefects_test_dir, dir1, dir2, "in"])
    out_path = os.path.sep.join([condefects_test_dir, dir1, dir2, "out"])
    if not os.path.exists(in_path) and os.path.exists(out_path):
        in_path = os.path.sep.join([condefects_test_dir, dir1, "Ex", "in"])
        out_path = os.path.sep.join([condefects_test_dir, dir1, "Ex", "out"])

    if not os.path.exists(in_path) and os.path.exists(out_path):
        return []
    else:
        for test_file in os.listdir(in_path):
            if os.path.exists(os.path.sep.join([out_path, test_file])):
                test_list.append(test_file)
    return test_list


def get_python_test_input(task_id="", test_name=""):
    dir1, dir2 = task_id.split("_")[0], task_id.split("_")[1].capitalize()
    path = os.path.sep.join([condefects_test_dir, dir1, dir2, "in", test_name])
    if not os.path.exists(path):
        path = os.path.sep.join([condefects_test_dir, dir1, "Ex", "in", test_name])
    return str(open(path, 'r').read())


def apply_patch(task_id="", program_id="", patch=""):
    program_path = os.path.sep.join([condefects_dir, task_id, "Python", program_id])
    faulty_version_path = os.path.sep.join([program_path, "faultyVersion.py"])
    with open(faulty_version_path, 'w') as f:
        f.write(patch)
    f.close()


def apply_ground_truth(task_id="", program_id=""):
    program_path = os.path.sep.join([condefects_dir, task_id, "Python", program_id])
    faulty_version_path = os.path.sep.join([program_path, "faultyVersion.py"])
    ground_truth_path = os.path.sep.join([program_path, "correctVersion.py"])
    correct_patch = open(ground_truth_path, 'r').read()
    with open(faulty_version_path, 'w') as f:
        f.write(correct_patch)
    f.close()


def longest_common_prefix(str1, str2):
    str1_lines = str1.split("\n")
    str2_lines = str2.split("\n")
    prefix = []
    for a, b in zip(str1_lines, str2_lines):
        if a == b:
            prefix.append(a)
        else:
            break
    return '\n'.join(prefix)


def longest_common_suffix(str1, str2):
    str1_lines = str1.split("\n")
    str2_lines = str2.split("\n")
    # 初始化索引和结果列表
    i = len(str1_lines) - 1
    j = len(str2_lines) - 1
    common_suffix = []

    # 从末尾开始逐字符比较
    while i >= 0 and j >= 0:
        if str1_lines[i] == str2_lines[j]:
            common_suffix.append(str1_lines[i])  # 或者 str2[j]，因为它们相等
            i -= 1
            j -= 1
        else:
            break

    # 将收集到的字符反转以得到正确的顺序
    return '\n'.join(reversed(common_suffix))


def get_mask_code(correct_code, buggy_code):
    prefix = longest_common_prefix(correct_code, buggy_code)
    suffix = longest_common_suffix(correct_code, buggy_code)

    return prefix + "\n>>> [ INFILL ] <<<\n" + suffix, buggy_code[len(prefix):len(buggy_code) - len(suffix)].lstrip(
        '\n').rstrip('\n'), correct_code[len(prefix):len(correct_code) - len(suffix)].lstrip('\n').rstrip('\n')


def collect_import():
    tasks = get_all_python_tasks()
    all_packages = set()  # 使用集合存储包名以去重

    for task_id in tasks:
        program_ids = get_python_programs(task_id)
        for program_id in program_ids:
            try:
                buggy_code, correct_code, bug_location = read_python_program_code(task_id, program_id)

                # 提取 correct_code 中的 import 语句
                import_statements = re.findall(r'(?:^import\s+|from\s+)(\w+)', correct_code, re.MULTILINE)

                # 将提取到的包名加入集合
                all_packages.update(import_statements)
            except:
                continue

    return all_packages


if __name__ == '__main__':
    # print(run_python_test("abc229_d", ['000.txt', '001.txt']))
    record_python_task_meta(output_file="./condefects_meta.jsonl")
    # incorporate_date_meta("./data/condefects_meta.jsonl", "./data/date.txt", "./data/condefects_meta_with_date.jsonl")

#    test_list = get_all_python_tests("abc255_g")
#    print(test_list)
#    program_id="35788058"
#    checkout_python_task("abc255_g")
#    # apply_ground_truth(task_id="abc255_g",program_id=program_id)
#    start_time=time.time()
#    print(run_python_test("abc255_g",program_id,test_list))
#    end_time=time.time()
#    print(f"consume {end_time-start_time} seconds")
# python /root/autodl-tmp/ConDefects-main/Code/abc255_g/Python/35788058/correctVersion.py < /root/autodl-tmp/ConDefects-main/Test/abc255/G/in/000.txt


# python3 ConDefects.py runsingle -w /root/autodl-tmp/ConDefects-main  -s abc255_g  -t 000.txt 001.txt 002.txt 003.txt 004.txt 005.txt 006.txt 007.txt 008.txt 009.txt 010.txt 011.txt 012.txt 013.txt 014.txt 015.txt 016.txt 017.txt 018.txt 019.txt 020.txt 021.txt 022.txt 023.txt 024.txt 025.txt 026.txt 027.txt 028.txt 029.txt 030.txt 031.txt 032.txt 033.txt 034.txt 035.txt 036.txt 037.txt 038.txt 039.txt 040.txt 041.txt 042.txt 043.txt 044.txt example0.txt example1.txt --program-id 35788058 --result-path abc255_g-35788058-ee59fb32-5c6a-497b-b5e9-29851e740a7a.json
