import os
import subprocess

import tqdm


def get_defects4j_projects():
    # 获取 bug 列表
    try:
        # 调用 defects4j 的命令行工具
        result = subprocess.run(['defects4j', 'pids'], capture_output=True, text=True, check=True)
        project_list = result.stdout.splitlines()

        return project_list
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return []


def get_defects4j_bugs_by_project(proj):
    # 获取 bug 列表
    try:
        # 调用 defects4j 的命令行工具
        result = subprocess.run(['defects4j', 'bids', '-p', proj], capture_output=True, text=True, check=True)
        bug_list = result.stdout.splitlines()
        return bug_list
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return []


def get_history_defects4j_project_and_bug(root_dir):
    projects_dir = f"{root_dir}/framework/projects"
    result = {}
    for proj in os.listdir(projects_dir):
        if proj in ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore',
                        'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']:
            patch_dir = f"{projects_dir}/{proj}/patches"
            bug_ids = set()
            for patch in os.listdir(patch_dir):
                bug_ids.add(patch.split('.')[0])
            result[proj] = list(bug_ids)
    return result


if __name__ == "__main__":
    projects = get_defects4j_projects()
    print(projects)
    # for proj in projects:
    #     bugs = get_defects4j_bugs_by_project(proj)
    #     for bug in tqdm.tqdm(bugs):
    #         clean_cmd=['rm', '-rf', '~/defects4j/defects4j_workspace']
    #         checkout_cmd = ['sh', './scripts/defects4j.sh', 'checkout_bug', proj, bug,
    #                         '~/defects4j/defects4j_workspace', "", "~/defects4j"]
    #         check_type_cmd = ['sh', './scripts/defects4j.sh', 'get_bug_type', proj, bug,
    #                           '~/defects4j/defects4j_workspace', "", "~/defects4j"]
    #         subprocess.run(clean_cmd)
    #         subprocess.run(checkout_cmd, capture_output=True, text=True, check=True)
    #         bug_type = subprocess.run(check_type_cmd, capture_output=True, text=True, check=True).stdout.strip()
    #         print(proj,bug,bug_type)
