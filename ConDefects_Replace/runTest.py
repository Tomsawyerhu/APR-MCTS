import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

COMLINE_COV_PY = "coverage run --rcfile=%s %s <%s > NUL 2>&1"
COMLINE_RUN_PY = "python3 %s <%s >%s "

COMLINE_COV_PY2 = "coverage2 run --rcfile=%s %s <%s > NUL 2>&1"
COMLINE_RUN_PY2 = "python2 %s <%s >%s "


def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    if including_parent:
        parent.kill()


def run_command_with_timeout(cmd, tempdir, timeout_sec):
    """在给定的超时时间内运行命令，如果命令超过设定的时间则杀死进程"""
    # 创建一个 Popen 对象
    process = subprocess.Popen(cmd, shell=True, cwd=tempdir)

    # 创建一个 Timer 对象
    timer = threading.Timer(timeout_sec, kill_proc_tree, args=[process.pid])

    try:
        timer.start()
        stdout, stderr = process.communicate()
    finally:
        timer.cancel()

    return process.returncode, stdout, stderr


def read_line(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().splitlines()
        return text
    except:
        print("read", file_name, "error")


def compare_res(user_res, correct_res):
    flag = True
    try:
        if not os.path.exists(user_res):
            return False
        user_res_text = read_line(user_res)
        correct_res_text = read_line(correct_res)
        if len(user_res_text) != len(correct_res_text):
            flag = False
        if flag:
            for i in range(len(correct_res_text)):
                if user_res_text[i] != correct_res_text[i]:
                    flag = False
    except:
        flag = False
    return flag


def getSrcCov_PY(prename, testLimit, srcPath, testDataPathDir, sourcelist, tempdir):
    covMatrix = []
    res = []
    files = os.listdir(testDataPathDir)
    testList = []
    python3 = True
    for source in sourcelist:
        if "print " in source:
            python3 = False
            break

    coverageFilePath = os.path.join(tempdir, prename) + "coverage"
    configtemp = "[run]\ndata_file = " + coverageFilePath
    configname = os.path.join(tempdir, "Configuration", "configuration" + prename)
    configPath = os.path.join(tempdir, "Configuration")
    if not os.path.exists(configPath):
        os.makedirs(configPath)

    with open(configname, "w") as f:
        f.write(configtemp)

    for i in files:
        # 0 list, size=len(sourcelist)
        singleCov = []
        for j in range(len(sourcelist)):
            singleCov.append(0)
        if testLimit != None and i not in testLimit:
            continue
        testList.append(i)
        input_file = os.path.join(testDataPathDir, i)
        output_file = os.path.join(tempdir, "result.out")
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(coverageFilePath):
            os.remove(coverageFilePath)
        if python3:
            cmd = COMLINE_COV_PY % (configname, srcPath, input_file)
        else:
            cmd = COMLINE_COV_PY2 % (configname, srcPath, input_file)
        try:
            os.system(cmd)
            with open(coverageFilePath, 'r') as f:
                text = f.read()

            # print(text)
            posr = text.rfind(']')
            posl = text.rfind('[')
            sub = text[posl + 1:posr].split(',')
            for item in sub:
                if item != '':
                    singleCov[int(item) - 1] = 1
            covMatrix.append(singleCov)
        except:
            covMatrix.append([])
        path_list = os.path.join(testDataPathDir, i).split(os.sep)
        path_list[-2] = "out"
        new_path = os.sep.join(path_list)
        if python3:
            cmd = COMLINE_RUN_PY % (srcPath, input_file, output_file)
        else:
            cmd = COMLINE_RUN_PY2 % (srcPath, input_file, output_file)

        os.system(cmd)
        res.append(compare_res(output_file, new_path))

    if os.path.exists(configname):
        os.remove(configname)
    return covMatrix, res, testList


# def getSrcCov_PY2(prename, testLimit, srcPath, testDataPathDir, sourcelist, tempdir):
#     covMatrix = []
#     res = []
#     files = os.listdir(testDataPathDir)
#     testList = []
#     python3 = True
#     for source in sourcelist:
#         if "print " in source:
#             python3 = False
#             break
#
#     coverageFilePath = os.path.join(tempdir, prename) + "coverage"
#     configtemp = "[run]\ndata_file = " + coverageFilePath
#     configname = os.path.join(tempdir, "Configuration", "configuration" + prename)
#     configPath = os.path.join(tempdir, "Configuration")
#     if not os.path.exists(configPath):
#         os.makedirs(configPath)
#
#     with open(configname, "w") as f:
#         f.write(configtemp)
#
#     for i in files:
#         # 0 list, size=len(sourcelist)
#         singleCov = []
#         for j in range(len(sourcelist)):
#             singleCov.append(0)
#         if testLimit != None and i not in testLimit:
#             continue
#         testList.append(i)
#         input_file = os.path.join(testDataPathDir, i)
#         output_file = os.path.join(tempdir, "result.out")
#         if os.path.exists(output_file):
#             os.remove(output_file)
#         if os.path.exists(coverageFilePath):
#             os.remove(coverageFilePath)
#         if python3:
#             cmd = COMLINE_COV_PY % (configname, srcPath, input_file)
#         else:
#             cmd = COMLINE_COV_PY2 % (configname, srcPath, input_file)
#
#         covMatrix.append([])
#         path_list = os.path.join(testDataPathDir, i).split(os.sep)
#         path_list[-2] = "out"
#         new_path = os.sep.join(path_list)
#         if python3:
#             cmd = COMLINE_RUN_PY % (srcPath, input_file, output_file)
#         else:
#             cmd = COMLINE_RUN_PY2 % (srcPath, input_file, output_file)
#         #        print("here")
#         os.system(cmd)
#
#         res.append(compare_res(output_file, new_path))
#
#     if os.path.exists(configname):
#         os.remove(configname)
#     return covMatrix, res, testList

import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_single_file(python3, testname, srcPath, input_file, output_file, new_path):
    """
    处理单个文件的逻辑，包括运行命令和比较结果。
    """
    if python3:
        cmd = COMLINE_RUN_PY % (srcPath, input_file, output_file)
    else:
        cmd = COMLINE_RUN_PY2 % (srcPath, input_file, output_file)

    # 执行命令
    os.system(cmd)

    # 比较结果
    result = compare_res(output_file, new_path)

    #测试输出
    test_output = open(output_file, 'r').read().rstrip("\n")

    #正确答案
    groundtruth_output = open(new_path, 'r').read().rstrip("\n")

    return testname, result, test_output, groundtruth_output


def getSrcCov_PY2(prename, testLimit, srcPath, testDataPathDir, sourcelist, tempdir):
    files = os.listdir(testDataPathDir)
    testResult = {}
    python3 = True

    # 检查是否为 Python 2
    for source in sourcelist:
        if "print " in source:
            python3 = False
            break

    # 配置 coverage 文件
    coverageFilePath = os.path.join(tempdir, prename) + "coverage"
    configtemp = "[run]\ndata_file = " + coverageFilePath
    configname = os.path.join(tempdir, "Configuration", "configuration" + prename)
    configPath = os.path.join(tempdir, "Configuration")
    if not os.path.exists(configPath):
        os.makedirs(configPath)

    with open(configname, "w") as f:
        f.write(configtemp)

    # 多线程处理文件
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_file = {}
        temp_files = []  # 用于存储所有临时文件路径，方便后续清理

        for i in files:
            if testLimit is not None and i not in testLimit:
                continue
            input_file = os.path.join(testDataPathDir, i)

            # 为每个子线程生成唯一的输出文件路径
            unique_output_file = os.path.join(tempdir, f"result_{i}.out")
            temp_files.append(unique_output_file)  # 记录临时文件路径

            # 清理旧文件
            if os.path.exists(unique_output_file):
                os.remove(unique_output_file)
            if os.path.exists(coverageFilePath):
                os.remove(coverageFilePath)

            # 构造输出路径
            path_list = os.path.join(testDataPathDir, i).split(os.sep)
            path_list[-2] = "out"
            new_path = os.sep.join(path_list)

            # 提交任务到线程池
            future = executor.submit(
                process_single_file,
                python3,
                i,
                srcPath,
                input_file,
                unique_output_file,  # 使用唯一输出文件
                new_path
            )
            future_to_file[future] = i

        # 收集结果
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                test_name , is_passed, test_output, groundtruth_output = future.result()
                testResult[test_name] = {
                    "is_passed": is_passed,
                    "test_output": test_output,
                    "groundtruth_output": groundtruth_output
                }
            except Exception as exc:
                print(f"File {file_name} generated an exception: {exc}")

    # 清理配置文件和所有临时文件
    if os.path.exists(configname):
        os.remove(configname)
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return testResult


def get_coverage(html, htmlLines):
    # Adjusted regular expression to also match "bfc"
    pattern = re.compile(r'<span class="([ a-z]*?)" id="L(\d+)"')
    # Find all matches
    matches = pattern.findall(html)
    # Maximum line number
    max_line = len(htmlLines) - 1
    # Create a list to represent coverage status for each line
    coverage_list = [0] * max_line
    for status, line_num in matches:
        if "fc" in status or "pc" in status:
            coverage_list[int(line_num) - 1] = 1
        elif "nc" in status:
            pass
        else:
            print("Error: Unknown status: " + status)
    return coverage_list


def getSrcCov_Java(prename, testLimit, testDataPathDir, tempdir, libdir):
    covMatrix = []
    res = []
    testList = []
    files = os.listdir(testDataPathDir)
    runFailFlag = False
    for i in files:
        if testLimit != None and i not in testLimit:
            continue
        testList.append(i)
        try:
            cmd = "javac " + os.path.join(tempdir, prename)
            os.system(cmd)
            reportPath = os.path.join(tempdir, "report")
            jacocoPath = os.path.join(tempdir, "jacoco.exec")
            if os.path.exists(reportPath):
                shutil.rmtree(reportPath)
            if os.path.exists(jacocoPath):
                os.remove(jacocoPath)
            cmd = "java -javaagent:" + os.path.join(libdir, "jacocoagent.jar") + "=destfile=" + jacocoPath \
                  + " Main < " + os.path.join(testDataPathDir, i) \
                  + " > " + os.path.join(tempdir, "Main.out")

            returncode, stdout, stderr = run_command_with_timeout(cmd, tempdir, timeout_sec=60)
            if returncode != 0:
                print("Error:", returncode)
                runFailFlag = True
                break
            cmd = "java -jar " + os.path.join(libdir, "jacococli.jar") \
                  + " report " \
                  + jacocoPath + " --classfiles . --sourcefiles . --html " \
                  + reportPath + " > NUL 2>&1"
            # os.system(cmd)
            process = subprocess.Popen(cmd, shell=True, cwd=tempdir)
            while not os.path.exists(os.path.join(reportPath, "default", "Main.java.html")):
                time.sleep(0.5)
            process.terminate()
            with open(os.path.join(reportPath, "default", "Main.java.html"), "r") as f:
                html = f.read()
            with open(os.path.join(reportPath, "default", "Main.java.html"), "r") as f:
                htmlLines = f.readlines()
            coverList = get_coverage(html, htmlLines)
            covMatrix.append(coverList)

            path_list = os.path.join(testDataPathDir, i).split(os.sep)
            path_list[-2] = "out"
            new_path = os.sep.join(path_list)

            res.append(compare_res(os.path.join(tempdir, "Main.out"), os.path.join(testDataPathDir, new_path)))
        except:
            covMatrix.append([])
            res.append(None)
    if runFailFlag:
        return None, None, None
    return covMatrix, res, testList


def runTest(destination_directory, output_directory, task, testLimit, cwd, coverage=False):
    testPath = os.path.join(cwd, "Test")
    if testLimit != None and task == None:
        print("Error: The -t parameter requires the -s parameter.")
        print("Use -h for help.")
        return

    DirList = os.listdir(destination_directory)
    for dir in DirList:
        contestName = dir.split('_')[0]
        if task != None and task != dir:
            continue
        targetContestPath = os.path.join(destination_directory, dir)
        codeList = []
        codeIDList = []
        if os.path.exists(os.path.join(targetContestPath, "Python")):
            codePythonList = os.listdir(os.path.join(targetContestPath, "Python"))
            for code in codePythonList:
                codeList.append(os.path.join(targetContestPath, "Python", code, "faultyVersion.py"))
                codeIDList.append(code)
        if os.path.exists(os.path.join(targetContestPath, "Java")):
            codeJavaList = os.listdir(os.path.join(targetContestPath, "Java"))
            for code in codeJavaList:
                codeList.append(os.path.join(targetContestPath, "Java", code, "faultyVersion.java"))
                codeIDList.append(code)
        targetTestPath = os.path.join(testPath, contestName, dir.split("_")[1].upper(), "in")
        if not os.path.exists(targetTestPath):
            targetTestPath = os.path.join(testPath, contestName, "Ex", "in")
        testList = os.listdir(targetTestPath)
        for codeIndex in range(len(codeList)):
            print("--------------------------------------------------")
            print("start task", dir, "code:", codeIDList[codeIndex])
            codeID = codeIDList[codeIndex]
            codePath = os.path.join(targetContestPath, codeList[codeIndex])

            tempPath = os.path.join(cwd, "Configuration", codeID)
            if os.path.exists(tempPath):
                shutil.rmtree(tempPath)
            os.makedirs(tempPath)
            languag = codePath.split(os.sep)[-3]
            if codePath.endswith("java"):
                preName = "Main.java"
                tempRunFilePath = os.path.join(tempPath, preName)
                shutil.copy(codePath, tempRunFilePath)
                covMatrix, res, testList = getSrcCov_Java(preName, testLimit, targetTestPath, tempPath,
                                                          os.path.join(cwd, "lib"))
                shutil.rmtree(tempPath)
            elif codePath.endswith("py"):
                covMatrix, res, testList = getSrcCov_PY(codeID, testLimit, codePath, targetTestPath,
                                                        read_line(codePath), tempPath)
                shutil.rmtree(tempPath)
            print("task", dir, "code:", codeID)
            if covMatrix == None or res == None:
                print("Error: Test failed.", codeID)
                continue
            for testIndex in range(len(testList)):
                print("test:", testList[testIndex], "result:", res[testIndex])
            if res == None:
                print("run fail", codeID)
                continue
            # calculate pass rate
            passNum = 0
            for item in res:
                if item == True:
                    passNum += 1
            print("pass rate:", passNum / len(res))
            if coverage != False:
                covMatrixPath = os.path.join(output_directory, dir, languag, codeID, "covMatrix.txt")
                resultPath = os.path.join(output_directory, dir, languag, codeID, "results.txt")
                testListPath = os.path.join(output_directory, dir, languag, codeID, "testList.txt")
                if not os.path.exists(os.path.join(output_directory, dir, languag, codeID)):
                    os.makedirs(os.path.join(output_directory, dir, languag, codeID))
                with open(covMatrixPath, "w") as f:
                    # convert list to string
                    for item in covMatrix:
                        f.write(str(item))
                        f.write("\n")
                with open(resultPath, "w") as f:
                    for item in res:
                        f.write(str(item))
                        f.write("\n")
                with open(testListPath, "w") as f:
                    for item in testList:
                        f.write(str(item))
                        f.write("\n")
                print("covMatrix saved in", covMatrixPath)


def runSingleTest(destination_directory, result_path, task, code, testLimit, cwd):
    testPath = os.path.join(cwd, "Test")
    dir = task
    contestName = dir.split('_')[0]

    targetContestPath = os.path.join(destination_directory, dir)
    faulty_path = os.path.join(targetContestPath, "Python", code, "faultyVersion.py")
    codeID = code

    targetTestPath = os.path.join(testPath, contestName, dir.split("_")[1].upper(), "in")
    if not os.path.exists(targetTestPath):
        targetTestPath = os.path.join(testPath, contestName, "Ex", "in")
    codePath = os.path.join(targetContestPath, faulty_path)
    tempPath = os.path.join(cwd, "Configuration", codeID)
    if os.path.exists(tempPath):
        shutil.rmtree(tempPath)
    os.makedirs(tempPath)
    test_result = getSrcCov_PY2(codeID, testLimit, codePath, targetTestPath,
                                read_line(codePath), tempPath)
    with open(result_path, 'w') as f:
        f.write(json.dumps(test_result))
    shutil.rmtree(tempPath)

    print(test_result)
