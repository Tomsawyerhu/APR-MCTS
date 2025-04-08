import subprocess
import json
import os
import time
import tqdm

import jsonlines

condefects_dir = "/mnt/data/hhc/ConDefects-main"
condefects_code_dir = condefects_dir + "/Code"
condefects_test_dir = condefects_dir + "/Test"
condefects_coverage_dir = condefects_dir + "/Coverage"


def run_command(command, cwd=""):
    """
    Helper function to run a shell command and return its output.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error message: {e.stderr}")
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
    run_command(command, cwd=condefects_dir)


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


def parse_test_result(outputs: str):
    program_marker = "--------------------------------------------------"
    test_result_marker = "=====================test result====================="
    correct_result_marker = "=====================correct result====================="
    results = {}
    for output in outputs.split(program_marker):
        if output.strip() == "":
            continue
        program_id = ""
        test_results, correct_results, is_test_passed = [], [], []
        test_result_pointer, correct_result_pointer = 0, 0
        i = output.find(test_result_marker, test_result_pointer)
        while i >= 0:
            j = output.find(test_result_marker, i + len(test_result_marker))
            test_results.append(output[i + len(test_result_marker):j].strip())
            i = output.find(test_result_marker, j + len(test_result_marker))

        p = output.find(correct_result_marker, correct_result_pointer)
        while p >= 0:
            q = output.find(correct_result_marker, p + len(correct_result_marker))
            correct_results.append(output[p + len(correct_result_marker):q].strip())
            p = output.find(correct_result_marker, q + len(correct_result_marker))
        for line in output.split("\n"):
            if "task" in line and "code:" in line:
                program_id = line.split("code:")[-1].strip()
            if "test:" in line and "result:" in line:
                if "False" in line:
                    is_test_passed.append(False)
                if "True" in line:
                    is_test_passed.append(True)
        results[program_id] = {
            "correct_results": correct_results,
            "test_results": test_results,
            "is_test_passed": is_test_passed
        }

    return results


def run_python_test(task_id="", test_list=[]):
    """

    :param task_id:
    :param test_list:
    :return: 测试用例列表,标准答案,测试结果,是否通过
    """
    if not isinstance(test_list, list) or len(test_list) == 0:
        raise Exception("test num must >=1")
    test_str = ' '.join(test_list)
    command = f"python3 ConDefects.py run -w {condefects_dir}  -s {task_id}  -t {test_str}"
    output = run_command(command, cwd=condefects_dir)
    test_result = parse_test_result(output)
    for k in test_result.keys():
        test_result[k]["test_list"] = test_list
    return test_result


def record_python_task_meta(output_file=""):
    existed_ids = []
    if os.path.exists(output_file):
        with jsonlines.open(output_file, 'r') as reader:
            for line in reader:
                existed_ids.append((line['task_id'], line['program_id']))
    python_tasks = get_all_python_tasks()
    for python_task in tqdm.tqdm(python_tasks):
        start_time = time.time()
        try:
            coverage_result = parse_python_task_coverage(python_task)
        except:
            continue
        end_time = time.time()
        for program in coverage_result.keys():
            if (python_task, program) in existed_ids:
                continue
            meta_info = {
                "task_id": python_task,
                "program_id": program,
                "test_list": coverage_result[program]['test_list'],
                "test_result": coverage_result[program]['test_result'],
                "time": end_time - start_time
            }
            with jsonlines.open(output_file, 'a') as writter:
                writter.write(meta_info)


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


def get_python_test_input(task_id="", test_name=""):
    dir1, dir2 = task_id.split("_")[0], task_id.split("_")[1].capitalize()
    path = os.path.sep.join([condefects_test_dir, dir1, dir2, "in", test_name])
    return str(open(path, 'r').read())


def apply_patch(task_id="", program_id="", patch=""):
    program_path = os.path.sep.join([condefects_dir, task_id, "Python", program_id])
    faulty_version_path = os.path.sep.join([program_path, "faultyVersion.py"])
    with open(faulty_version_path, 'w') as f:
        f.write(patch)
    f.close()


if __name__ == '__main__':
    print(run_python_test("abc229_d", ['000.txt', '001.txt']))
    # record_python_task_meta(output_file="./condefects_meta.jsonl")
