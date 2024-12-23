import pandas as pd
import json

# 读取Parquet文件
input_parquet_file = './data/train-00000-of-00004-179f0635c54dfdf9.parquet'
output_jsonl_file = './data/train-00000-of-00004-179f0635c54dfdf9.jsonl'

# 读取数据
df = pd.read_parquet(input_parquet_file)

# 过滤条件: is_single_function为True，并且buggy_function和fixed_function不同
filtered_df = df[(df['is_single_function'] == True) & (df['buggy_function'] != df['fixed_function'])]

# 处理记录
results = []

for index, row in filtered_df.iterrows():
    if row['is_single_chunk'] == True:
        # 比较相同的行（从头比较和从尾比较）
        buggy_lines = row['buggy_function'].splitlines()
        fixed_lines = row['fixed_function'].splitlines()

        # 计算从开头相同的行数
        common_head_lines = 0
        for i in range(min(len(buggy_lines), len(fixed_lines))):
            if buggy_lines[i] != fixed_lines[i]:
                break
            else:
                common_head_lines += 1

        # 计算从结尾相同的行数
        common_tail_lines = 0
        for i in range(min(len(buggy_lines), len(fixed_lines))):
            if buggy_lines[-i - 1] != fixed_lines[-i - 1]:
                break
            else:
                common_tail_lines += 1

        if common_head_lines + common_tail_lines < min(len(buggy_lines), len(fixed_lines)):
            diff_buggy_lines = buggy_lines[common_head_lines:-common_tail_lines]
            diff_fixed_lines = fixed_lines[common_head_lines:-common_tail_lines]
            mask_code_lines = buggy_lines[:common_head_lines]+[">>>[[INFILL]]<<<"]+buggy_lines[-common_tail_lines:]
        else:
            continue

        # 记录不同的行
        buggy_diff = [line for line in buggy_lines if line not in fixed_lines]
        fixed_diff = [line for line in fixed_lines if line not in buggy_lines]

        result = {
            'buggy_function': row['buggy_function'],
            'fixed_function': row['fixed_function'],
            'is_single_chunk': row['is_single_chunk'],
            'is_single_function': row['is_single_function'],
            'mask_code': '\n'.join(mask_code_lines),
            'buggy_lines': '\n'.join(diff_buggy_lines),
            'fixed_lines': '\n'.join(diff_fixed_lines)
        }

        results.append(result)

    else:
        # is_single_chunk为False的记录
        result = {
            'buggy_function': row['buggy_function'],
            'fixed_function': row['fixed_function'],
            'is_single_chunk': row['is_single_chunk'],
            'is_single_function': row['is_single_function'],
            'mask_code': None,
            'buggy_lines': None,
            'fixed_lines': None
        }

        results.append(result)

# 保存结果到JSONL文件
with open(output_jsonl_file, 'w') as jsonl_file:
    for result in results:
        jsonl_file.write(json.dumps(result) + '\n')

print(f"处理完成，结果已保存到 {output_jsonl_file}.")
