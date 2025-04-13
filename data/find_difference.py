import json

part = []
with open('./mcts_llama_8b_16_rollout.jsonl', 'r') as f:
    for line in f:
        json_line = json.loads(line)
        part.append(json_line["project"] + "_" + str(json_line["bug_id"]))

with open('./result_yi.jsonl', 'r') as f:
    for line in f:
        json_line = json.loads(line)
        if json_line["project"] + "_" + str(json_line["bug_id"]) in part:
            continue
        print(json_line["project"] + "_" + str(json_line["bug_id"]))
