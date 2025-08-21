import hashlib
import json
import os
import re
from pathlib import Path
import logging

from utils import run_bash, get_test_names, extract_method, make_failing_tests_short, tmp_dir

bug_details_cache_folder = "./bug_cache"
# validate_patch_cache_folder = "./validate_patch_cache"
validate_patch_cache_folder = None


class Bug(object):
    def __init__(self,
                 test_framework,
                 project,
                 bug_id,
                 bug_type,
                 code,
                 masked_code,
                 fixed_code,
                 buggy_lines,
                 fixed_lines,
                 test_code,
                 extract_test_code,
                 failing_tests,
                 test_suite,
                 test_name,
                 test_line,
                 test_error_message,
                 test_input=None,
                 test_output=None,
                 expected_output=None):
        self.test_framework = test_framework
        self.project = project
        self.bug_id = bug_id
        self.bug_type = bug_type
        self.code = code
        self.masked_code = masked_code
        self.fixed_code = fixed_code
        self.buggy_lines = buggy_lines
        self.fixed_lines = fixed_lines
        self.test_code = test_code
        self.extract_test_code = extract_test_code
        self.failing_tests = failing_tests
        self.test_suite = test_suite
        self.test_name = test_name
        self.test_line = test_line
        self.test_error_message = test_error_message
        # condefects 新加的字段
        self.test_input = test_input
        self.test_output = test_output
        self.expected_output = expected_output


def get_test(project, bug_id, test_suit):
    test_file = f"{tmp_dir}/{project}-{bug_id}/src/test/java/{test_suit.replace('.', '/')}.java"
    test_file1 = f"{tmp_dir}/{project}-{bug_id}/src/tests/java/{test_suit.replace('.', '/')}.java"
    test_file2 = f"{tmp_dir}/{project}-{bug_id}/src/tests/{test_suit.replace('.', '/')}.java"
    test_file3 = f"{tmp_dir}/{project}-{bug_id}/src/test/{test_suit.replace('.', '/')}.java"
    test_file4 = f"{tmp_dir}/{project}-{bug_id}/tests/{test_suit.replace('.', '/')}.java"
    test_file5 = f"{tmp_dir}/{project}-{bug_id}/test/{test_suit.replace('.', '/')}.java"
    test_file6 = f"{tmp_dir}/{project}-{bug_id}/source/test/java/{test_suit.replace('.', '/')}.java"
    test_file7 = f"{tmp_dir}/{project}-{bug_id}/source/tests/java/{test_suit.replace('.', '/')}.java"
    test_file8 = f"{tmp_dir}/{project}-{bug_id}/source/tests/{test_suit.replace('.', '/')}.java"
    test_file9 = f"{tmp_dir}/{project}-{bug_id}/source/test/{test_suit.replace('.', '/')}.java"
    #gson is special
    test_file10 = f"{tmp_dir}/{project}-{bug_id}/gson/src/test/java/{test_suit.replace('.', '/')}.java"

    files = [test_file, test_file1, test_file2, test_file3, test_file4, test_file5, test_file6, test_file7, test_file8,
             test_file9, test_file10]
    for f in files:
        if os.path.exists(f):
            return str(open(f, "r").read())
    raise FileNotFoundError(f"No test file found for {project}-{bug_id}")


def get_bug_details(project, bug_id, max_test_to_show=10,max_test_length=500):
    if bug_details_cache_folder is not None:
        file_path = f"{bug_details_cache_folder}/{project}-{bug_id}.json"
        if Path(file_path).is_file():
            logging.debug(f"Retrieving bug details from cache (project={project}, bug_id={bug_id})")
            with open(file_path, "r") as f:
                bug_details = json.load(f)
            return Bug(**bug_details)

    logging.debug(f"Checking out bug (project={project}, bug_id={bug_id}))")
    run_bash("checkout_bug", project, bug_id)

    bug_type = run_bash("get_bug_type", project, bug_id).stdout
    test_suite, test_name, test_error, test_line, buggy_lines, fixed_lines, code, masked_code, fixed_code, failing_tests, test_code, extract_test_code = None, None, None, None, None, None, None, None, None, None, None, None
    if bug_type != "OT":
        logging.debug(f"Compiling and running tests")
        a = run_bash("compile_and_run_tests", project, bug_id)
        # print(a)
        logging.debug(f"Retreiving test suite")
        test_suite = run_bash("get_test_suite", project, bug_id).stdout
        logging.debug(f"Retreiving test name")
        test_name = run_bash("get_test_name", project, bug_id).stdout
        logging.debug(f"Retreiving test error message")
        test_error = run_bash("get_test_error", project, bug_id).stdout
        logging.debug(f"Retreiving test line")
        test_line = run_bash("get_test_line", project, bug_id).stdout
        logging.debug(f"Retreiving buggy lines")
        buggy_lines = run_bash("get_buggy_lines", project, bug_id).stdout
        logging.debug(f"Retreiving fixed lines")
        fixed_lines = run_bash("get_fixed_lines", project, bug_id).stdout
        logging.debug(f"Retreiving code")
        result = run_bash("get_code", project, bug_id)
        code = run_bash("get_code", project, bug_id).stdout
        logging.debug(f"Retreiving masked code")
        masked_code = run_bash("get_masked_code", project, bug_id).stdout
        logging.debug(f"Retreiving fixed code")
        fixed_code = run_bash("get_fixed_code", project, bug_id).stdout
        logging.debug(f"Retreiving failing tests")
        failing_tests = run_bash("get_failing_tests", project, bug_id).stdout
        logging.debug(f"Retreiving test code")
        test_code = get_test(project, bug_id, test_suite)
        test_names = get_test_names(failing_tests)
        test_methods = ""
        methods_found = []

        if len(test_names) > max_test_to_show:
            skip_tests = len(test_names) - max_test_to_show
            test_names = test_names[:max_test_to_show]
        else:
            skip_tests = 0
        for test_name in test_names:
            t = extract_method(test_code, test_name)
            if t != "":
                methods_found.append(test_name)
                if len(t)>max_test_length:
                    t=t[:max_test_length]+'\n+...'
                test_methods += t + "\n\n"
        extract_test_code = test_methods
        if skip_tests > 0:
            extract_test_code += f'{skip_tests} tests have been omitted.\n\n'
        failing_tests = make_failing_tests_short(failing_tests, methods_found)

    bug = Bug(test_suite=test_suite, test_name=test_name, test_line=test_line, test_error_message=test_error,
              buggy_lines=buggy_lines, fixed_lines=fixed_lines, code=code, masked_code=masked_code,
              fixed_code=fixed_code,
              test_framework="cigar", project=project, bug_id=bug_id, bug_type=bug_type, failing_tests=failing_tests,
              test_code=test_code, extract_test_code=extract_test_code)

    # if bug_details_cache_folder is not None:
    #     with open(f'{file_path}', 'w') as f:
    #         vars_object = vars(bug)
    #         f.write(json.dumps(vars_object, indent=4, sort_keys=True))

    return bug


def validate_patch(bug: Bug, proposed_patch: str, mode: str):
    assert mode in ["SL", "SH", "SF"]

    test_result = None
    result_reason = None
    patch_hash = hashlib.md5(str(proposed_patch).encode('utf-8')).hexdigest()

    # if validate_patch_cache_folder is not None:
    #     cache_file_path = f"{validate_patch_cache_folder}/{bug.project}_{bug.bug_id}_{mode}_{patch_hash}.json"
    #     if Path(cache_file_path).is_file():
    #         with open(cache_file_path, "r") as file:
    #             json_to_load = json.load(file)
    #             test_result = json_to_load['test_result']
    #             result_reason = json_to_load['result_reason']
    #             patch_diff = json_to_load['patch_diff']
    #         logging.info(f"Retrieved test result from cache: {patch_hash}")

    if test_result is None and result_reason is None:

        project = bug.project
        bug_id = bug.bug_id

        a = run_bash("checkout_bug", bug.project, bug.bug_id)

        result = run_bash("validate_patch", project, bug_id, proposed_patch, mode)
        patch_diff = run_bash("get_patch_git_diff", bug.project, bug.bug_id).stdout

        logging.debug(f"Retreiving test name")
        test_name = run_bash("get_test_name", project, bug_id).stdout

        if result.returncode != 0:
            if result.stderr.find("error: ") > 0:
                result_reason = result.stderr
                result_reason = result_reason[result_reason.find("error: "):]
                result_reason = result_reason[:result_reason.find("\n")]
            elif "BUILD FAILED" in result.stderr:
                stderr_lines = result.stderr.split("\n")
                build_failed_line_i = next((i for i, line in enumerate(stderr_lines) if "BUILD FAILED" in line),
                                           None)  # line number of line that contains "BUILD FAILED"
                result_reason = stderr_lines[build_failed_line_i + 1]
                result_reason = result_reason[result_reason.find(' '):]
            else:
                result_reason = "Test timed out after 600 seconds"

            test_result, result_reason = "ERROR", result_reason  # compilation error
        else:
            all_tests_passed = result.stdout.find("Failing tests: 0") != -1

            if all_tests_passed:
                test_result, result_reason = "PASS", "all tests passed"  # test pass
            else:
                test_result = "FAIL"  # test fail
                result_reason = run_bash("get_test_error", project, bug_id).stdout

        # if validate_patch_cache_folder is not None:
        #     with open(cache_file_path, "w") as file:
        #         json.dump({'patch': proposed_patch, 'test_result': test_result, 'result_reason': result_reason,
        #                    'patch_diff': patch_diff}, file, indent=4, sort_keys=True)

    logging.info(
        f"Test result for patch with the hash {patch_hash} is: {test_result}. The reason for this result is: {result_reason}")
    return test_result, result_reason, patch_diff


def remove_comments_and_blank(code):
    # 去除单行注释
    code = re.sub(r'//.*', '', code)
    # 去除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', '', code)
    return code


if __name__ == '__main__':

    manual_txt = "./data/overlap_yi-9b.txt"

    already_have = []
    if os.path.exists(manual_txt):
        with open(manual_txt, 'r') as t:
            for line in t:
                already_have.append(line.strip().split(',')[0])
    # plausible_dir = "./data/gp"
    with open("./data/mcts_yi_9b_16_rollout.jsonl", 'r') as f:
        for line in f:
            json_line = json.loads(line)
            if json_line['eval'] == 'PASS':
                if json_line["project"] + "_" + str(json_line["bug_id"]) in already_have:
                    continue
                bug = get_bug_details(json_line["project"], int(json_line["bug_id"]))
                # plausible_path = f"{plausible_dir}/{json_line['project']}-{json_line['bug_id']}.jsonl"
                find_em=False
                # if os.path.exists(plausible_path):
                #     with open(plausible_path, 'r') as t:
                #         for t_line in t:
                #             json_line = json.loads(line)
                #             if (bug.fixed_lines is not None and remove_comments_and_blank(
                #                     json_line["patch"]) == remove_comments_and_blank(bug.fixed_lines)) or (
                #                     bug.fixed_code is not None and remove_comments_and_blank(
                #                     json_line["patch"]) == remove_comments_and_blank(bug.fixed_code)):
                #                 with open(manual_txt, 'a') as tt:
                #                     tt.write(json_line["project"] + "_" + str(json_line["bug_id"]) + ",True\n")
                #                     find_em=True
                #                 break
                #         if not find_em:
                #             with open(manual_txt, 'a') as tt:
                #                 tt.write(json_line["project"] + "_" + str(json_line["bug_id"]) + ",\n")


                # else:
                with open(manual_txt, 'a') as t:
                    if (bug.fixed_lines is not None and remove_comments_and_blank(
                            json_line["patch"]) == remove_comments_and_blank(bug.fixed_lines)) or (
                            bug.fixed_code is not None and remove_comments_and_blank(
                        json_line["patch"]) == remove_comments_and_blank(bug.fixed_code)):
                        t.write(json_line["project"] + "_" + str(json_line["bug_id"]) + ",True\n")
                    else:
                        t.write(json_line["project"] + "_" + str(json_line["bug_id"]) + ",\n")

    # proj = "JacksonDatabind"
    # bid = 88
    # bug = get_bug_details(proj, bid)
    #
    # print(bug.fixed_code)
    # # write to diff1.txt
    # with open("./data/buggy_code.java", "w") as f:
    #     f.write(bug.code)
    # with open("./data/fixed_code.java", "w") as f:
    #     f.write(bug.fixed_code)
    # with open("./data/cot_tot/yi-9b.tot.jsonl", 'r') as f:
    #     for line in f:
    #         json_line = json.loads(line)
    #         if json_line['eval'] == 'PASS' and json_line['project'] == proj and str(json_line['bug_id']) == str(bid):
    #             with open("./data/plausible_patch.java", "w") as ff:
    #                 ff.write(json_line["patch"])
