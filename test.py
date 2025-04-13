import jsonlines

from datetime import datetime


def is_date_in_range(date_str):
    """
    判断给定的日期字符串是否在 2021-10-01 到 2023-09-30 之间。

    参数:
        date_str (str): 日期字符串，格式为 "YYYY-MM-DD"。

    返回:
        bool: 如果日期在范围内，返回 True；否则返回 False。
    """
    # 定义日期范围
    start_date = datetime(2021, 10, 1)
    end_date = datetime(2023, 9, 30)

    try:
        # 将输入字符串转换为日期对象
        input_date = datetime.strptime(date_str, "%Y-%m-%d")

        # 判断日期是否在范围内
        return start_date <= input_date <= end_date
    except ValueError:
        # 如果日期格式不正确，抛出异常并返回 False
        print(f"Invalid date format: {date_str}. Please use 'YYYY-MM-DD'.")
        return False


def main1():
    print("2021-10-01~2023-09-30,","2023-09-30~")
    meta_info = {}
    date_info = {}
    meta_info_old=[]
    with jsonlines.open("./data/condefects_meta_with_date.jsonl", 'r') as reader:
        for line in reader:
            a = meta_info.get(line["task_id"], [])
            a.append(line["program_id"])
            meta_info[line["task_id"]] = a
            date_info[line["task_id"]] = line["date"]
            if is_date_in_range(line["date"]):
                meta_info_old.append(line)
    with jsonlines.open("./data/condefects_meta_old_with_date.jsonl", 'w') as writter:
        writter.write_all(meta_info_old)

    print("元信息",len([x for x in meta_info.keys() if is_date_in_range(date_info[x])]),len([x for x in meta_info.keys() if not is_date_in_range(date_info[x])]))
    pass_result, fail_result = [], []
    tasks = set()
    with jsonlines.open("./data/condefects_mcts_result.jsonl", 'r') as reader:
        for line in reader:
            tasks.add(line['project'])
            if line["eval"] == "PASS":
                pass_result.append((line['project'], line['bug_id']))
            else:
                fail_result.append((line['project'], line['bug_id']))
    print("运行的",len([x for x in tasks if is_date_in_range(date_info[x])]),len([x for x in tasks if not is_date_in_range(date_info[x])]))
    pass_num = 0
    pass_num_new = 0
    for p in meta_info.keys():
        # check date

        is_pass = True
        for q in meta_info[p]:
            if (p, q) not in pass_result:
                is_pass = False
                break
        if is_pass:
            if not is_date_in_range(date_info[p]):
                pass_num_new += 1
            else:
                pass_num += 1
    print("修复的",pass_num, pass_num_new)
    print([x for x in meta_info.keys() if len(meta_info[x])==1])

def main2():
    with jsonlines.open("./data/condefects_mcts_result.jsonl", 'r') as reader,jsonlines.open("./data/condefects_mcts_result2.jsonl", 'w') as writer:
        for line in reader:
            if line["eval"]=="PASS":
                writer.write(line)

main1()
