# parse_patch.py
import os
import subprocess

import jsonlines
import pandas as pd
from typing import List, Tuple
from unidiff import PatchSet
from utils import run_command


def load_swe_lite(file=''):
    df = pd.read_parquet(file)
    data = []

    for _, row in df.iterrows():  # row 是 Series
        repo = row.get('repo')
        instance_id = row.get('instance_id')
        base_commit = row.get('base_commit')
        patch = row.get('patch')
        test_patch = row.get('test_patch')
        problem_statement = row.get('problem_statement')
        hints_text = row.get('hints_text')
        created_at = row.get('created_at')
        FAIL_TO_PASS = row.get('FAIL_TO_PASS')
        PASS_TO_PASS = row.get('PASS_TO_PASS')
        environment_setup_commit = row.get('environment_setup_commit')
        data.append({
            'repo': repo,
            'instance_id': instance_id,
            'base_commit': base_commit,
            'patch': patch,
            'test_patch': test_patch,
            'problem_statement': problem_statement,
            'hints_text': hints_text,
            'created_at': created_at,
            'FAIL_TO_PASS': FAIL_TO_PASS,
            'PASS_TO_PASS': PASS_TO_PASS,
            'environment_setup_commit': environment_setup_commit
        })
    return data


def parse_git_diff(diff_text: str):
    """
    解析 git diff / patch 字符串
    返回: [{file:str, added:list[int], deleted:list[int]}]
    """
    patch = PatchSet(diff_text.splitlines(keepends=True))
    result = []

    for f in patch:
        # 跳过被整文件删除的情况
        if f.is_removed_file and not f.is_added_file:
            continue

        added, deleted = [], []

        for hunk in f:
            old_ln = hunk.source_start  # 旧文件行号
            new_ln = hunk.target_start  # 新文件行号

            for line in hunk:
                if line.is_added:  # '+' 行
                    added.append(new_ln)
                    new_ln += 1
                elif line.is_removed:  # '-' 行
                    deleted.append(old_ln)
                    old_ln += 1
                else:  # ' ' 行 (context)
                    old_ln += 1
                    new_ln += 1

        result.append({"file": f.path, "added": added, "deleted": deleted})

    return result


def get_context_blocks(defect_lines: List[int], total_lines: int, context_span: int = 20) -> List[Tuple[int, int]]:
    """
    根据缺陷行号生成包含上下文的代码块，并合并重叠的区间。

    :param defect_lines: 缺陷行号列表
    :param total_lines: 文件总行数
    :param context_span: 每个缺陷行前后保留的行数
    :return: 合并后的上下文代码块列表（每个元素是 (start_line, end_line)）
    """
    if not defect_lines:
        return []

    # Step 1: 为每个缺陷行生成上下文区间
    intervals = []
    for line in defect_lines:
        start = max(1, line - context_span)
        end = min(total_lines, line + context_span)
        intervals.append((start, end))

    # Step 2: 排序区间
    intervals.sort()

    # Step 3: 合并重叠或相邻的区间
    merged_intervals = []
    for interval in intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last_start, last_end = merged_intervals[-1]
            current_start, current_end = interval

            # 如果当前区间与上一个有重叠或连续，就合并
            if current_start <= last_end + 1:
                new_start = last_start
                new_end = max(last_end, current_end)
                merged_intervals[-1] = (new_start, new_end)
            else:
                merged_intervals.append(interval)

    return merged_intervals


def wrap_relevant_lines_with_context(bug_repo, formated_line_fl: dict):
    result = {}
    for file_path in formated_line_fl:
        localized_lines = formated_line_fl[file_path]
        if len(localized_lines) == 0:
            continue
        abs_path = f'{bug_repo}/{file_path}'
        file_content = open(abs_path, 'r').read()
        total_lines = file_content.splitlines()
        total_line_num = len(total_lines)
        intervals = get_context_blocks(localized_lines, total_line_num, context_span=20)
        if len(intervals) == 0:
            continue
        result[file_path] = []
        for interval in intervals:
            result[file_path].append('\n'.join(total_lines[interval[0] - 1:interval[1]]))
    return result


def get_swe_gt_fl(
        input_file='./data/swe_lite.parquet',
        output_file='./swe_lite_loc.jsonl'
):
    swe_data = load_swe_lite(input_file)
    for item in swe_data:
        locations = parse_git_diff(item['patch'])
        item['buggy_locations'] = locations
    with jsonlines.open(output_file, 'w') as f:
        f.write_all(swe_data)


def get_current_branch(repo_dir="."):
    """
    返回 repo_dir Git 仓库的当前分支名。
    如果是 detached HEAD，返回实际的 commit SHA。
    """
    result = subprocess.check_output(
        ["git", "-C", repo_dir, "rev-parse", "--abbrev-ref", "HEAD"],
        text=True  # Py3.7+ 直接返回 str
    ).strip()
    return result


def get_swe_gt_fl_with_context(
        repo_root='./data/repo',
        input_file='./swe_lite_loc.jsonl',
        output_file='./swe_lite_loc_with_context.jsonl'):
    with jsonlines.open(input_file, 'r') as f:
        for line in f:
            if line['repo'] in ['astropy/astropy']:
                continue
            base_commit = line['base_commit']
            instance_id = line['instance_id']
            buggy_locations = line['buggy_locations']
            repo = line['repo']
            repo_name = repo.split('/')[-1]
            # clone repository to local
            if not os.path.exists(f'{repo_root}/{repo_name}'):
                clone_result = run_command(f'git clone https://github.com/{repo}', cwd=repo_root, timeout=600)
                if clone_result['returncode'] == 0:
                    print(f'successfully clone {repo} to local')
                else:
                    print(f'clone {repo} to local fail', clone_result)
                    return
            main_branch = 'main'
            if line['repo'] in ['sphinx-doc/sphinx','mwaskom/seaborn','sympy/sympy']:
                main_branch = 'master'
            checkout_master_result = run_command(f'git checkout {main_branch}', cwd=f'{repo_root}/{repo_name}')
            print(f'checkout to {main_branch}', checkout_master_result)
            rm_branch_result = run_command(f'git branch -D {instance_id}', cwd=f'{repo_root}/{repo_name}')
            print(f'delete branch {instance_id}', rm_branch_result)
            checkout_result = run_command(f'git checkout -b {instance_id}', cwd=f'{repo_root}/{repo_name}')
            print(f'checkout to {instance_id}', checkout_result)
            reset_result = run_command(f'git reset --hard {base_commit}', cwd=f'{repo_root}/{repo_name}')
            print(f'reset to {base_commit}', reset_result)

            buggy_lines = {}
            for location in buggy_locations:
                lines = list(set(location['deleted'] + location['added']))
                lines.sort()
                buggy_lines[location['file']] = lines

            wrap_result = wrap_relevant_lines_with_context(bug_repo=f'{repo_root}/{repo_name}',
                                                           formated_line_fl=buggy_lines)

            buggy_files = {}
            for file_name in wrap_result:
                if file_name not in buggy_files:
                    buggy_files[file_name] = open(f'{repo_root}/{repo_name}/{file_name}', 'r').read()

            checkout_master_result = run_command(f'git checkout {main_branch}', cwd=f'{repo_root}/{repo_name}')
            print(f'checkout to {main_branch}', checkout_master_result)

            line['buggy_code'] = wrap_result
            line['buggy_files'] = buggy_files
            with jsonlines.open(output_file, 'a') as f:
                f.write(line)


if __name__ == "__main__":
    get_swe_gt_fl_with_context()
