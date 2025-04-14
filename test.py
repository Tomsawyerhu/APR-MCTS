import os

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


var = ['abc221_f', 'abc221_g', 'abc221_h', 'abc222_g', 'abc223_a', 'abc223_b', 'abc223_d', 'abc223_e', 'abc223_f',
       'abc223_g', 'abc223_h', 'abc224_a', 'abc224_b', 'abc224_c', 'abc224_d', 'abc224_f', 'abc224_g', 'abc224_h',
       'abc225_a', 'abc225_c', 'abc225_d', 'abc225_e', 'abc225_f', 'abc225_h', 'abc226_a', 'abc226_b', 'abc226_c',
       'abc226_d', 'abc226_e', 'abc226_g', 'abc229_a', 'abc229_b', 'abc229_d', 'abc229_e', 'abc229_f', 'abc230_a',
       'abc230_b', 'abc230_c', 'abc230_d', 'abc230_e', 'abc230_g', 'abc232_b', 'abc232_c', 'abc232_d', 'abc232_e',
       'abc232_f', 'abc232_g', 'abc232_h', 'abc233_a', 'abc233_b', 'abc233_c', 'abc233_e', 'abc233_f', 'abc233_g',
       'abc233_h', 'abc234_a', 'abc234_b', 'abc234_c', 'abc234_d', 'abc234_e', 'abc234_f', 'abc234_g', 'abc234_h',
       'abc235_b', 'abc235_c', 'abc235_d', 'abc235_e', 'abc235_g', 'abc235_h', 'abc236_a', 'abc236_b', 'abc236_d',
       'abc236_e', 'abc236_g', 'abc236_h', 'abc237_a', 'abc237_b', 'abc237_c', 'abc237_d', 'abc237_f', 'abc237_h',
       'abc238_a', 'abc238_b', 'abc238_c', 'abc238_d', 'abc238_e', 'abc238_f', 'abc238_g', 'abc239_a', 'abc239_c',
       'abc239_d', 'abc239_e', 'abc239_g', 'abc240_a', 'abc240_d', 'abc240_e', 'abc240_g', 'abc240_h', 'abc241_a',
       'abc241_b', 'abc241_c', 'abc241_d', 'abc241_e', 'abc241_f', 'abc241_g', 'abc242_a', 'abc242_b', 'abc242_c',
       'abc242_d', 'abc242_e', 'abc242_g', 'abc243_b', 'abc243_c', 'abc243_d', 'abc243_e', 'abc243_f', 'abc243_h',
       'abc244_b', 'abc244_c', 'abc244_d', 'abc244_e', 'abc244_g', 'abc245_a', 'abc245_b', 'abc245_d', 'abc245_e',
       'abc245_f', 'abc245_g', 'abc245_h', 'abc246_a', 'abc246_d', 'abc246_e', 'abc246_g', 'abc247_a', 'abc247_b',
       'abc247_c', 'abc247_d', 'abc247_e', 'abc247_f', 'abc247_g', 'abc247_h', 'abc248_a', 'abc248_c', 'abc248_d',
       'abc248_e', 'abc248_f', 'abc249_a', 'abc249_b', 'abc249_c', 'abc249_d', 'abc249_e', 'abc249_f', 'abc250_a',
       'abc250_b', 'abc250_c', 'abc250_d', 'abc250_e', 'abc250_g', 'abc250_h', 'abc251_a', 'abc251_b', 'abc251_c',
       'abc251_d', 'abc251_e', 'abc251_f', 'abc251_g', 'abc251_h', 'abc252_a', 'abc252_b', 'abc252_c', 'abc252_d',
       'abc252_e', 'abc253_a', 'abc253_b', 'abc253_d', 'abc253_e', 'abc253_f', 'abc253_g', 'abc254_a', 'abc254_c',
       'abc254_d', 'abc254_e', 'abc254_f', 'abc254_g', 'abc255_a', 'abc255_b', 'abc255_c', 'abc255_e', 'abc255_f',
       'abc255_g', 'abc256_b', 'abc256_d', 'abc256_f', 'abc256_g', 'abc257_a', 'abc257_b', 'abc257_c', 'abc257_d',
       'abc257_e', 'abc257_f', 'abc257_g', 'abc257_h', 'abc258_a', 'abc258_b', 'abc258_c', 'abc258_d', 'abc258_e',
       'abc258_f', 'abc258_h', 'abc259_a', 'abc259_b', 'abc259_c', 'abc259_d', 'abc259_e', 'abc259_g', 'abc260_a',
       'abc260_b', 'abc260_d', 'abc260_e', 'abc260_f', 'abc260_g', 'abc261_a', 'abc261_b', 'abc261_d', 'abc261_e',
       'abc261_g', 'abc261_h', 'abc262_a', 'abc262_c', 'abc262_d', 'abc262_e', 'abc262_f', 'abc262_g', 'abc262_h',
       'abc263_a', 'abc263_b', 'abc263_c', 'abc263_d', 'abc263_e', 'abc263_g', 'abc263_h', 'abc264_a', 'abc264_b',
       'abc264_c', 'abc264_e', 'abc264_g', 'abc265_a', 'abc265_b', 'abc265_c', 'abc265_d', 'abc265_f', 'abc265_g',
       'abc266_b', 'abc266_c', 'abc266_d', 'abc266_e', 'abc266_f', 'abc266_g', 'abc266_h', 'abc267_a', 'abc267_b',
       'abc267_c', 'abc267_d', 'abc267_e', 'abc267_f', 'abc267_g', 'abc267_h', 'abc268_b', 'abc268_c', 'abc268_d',
       'abc268_e', 'abc268_g', 'abc268_h', 'abc269_a', 'abc269_b', 'abc269_c', 'abc269_d', 'abc269_e', 'abc269_g',
       'abc270_a', 'abc270_b', 'abc270_c', 'abc270_d', 'abc270_g', 'abc271_a', 'abc271_c', 'abc271_d', 'abc271_e',
       'abc271_g', 'abc271_h', 'abc272_a', 'abc272_b', 'abc272_c', 'abc272_d', 'abc272_e', 'abc272_f', 'abc272_g',
       'abc273_b', 'abc273_c', 'abc273_d', 'abc273_g', 'abc274_a', 'abc274_c', 'abc274_d', 'abc274_e', 'abc274_f',
       'abc275_a', 'abc275_b', 'abc275_c', 'abc275_d', 'abc275_f', 'abc275_g', 'abc276_a', 'abc276_b', 'abc276_c',
       'abc276_d', 'abc276_h', 'abc277_a', 'abc277_b', 'abc277_d', 'abc277_e', 'abc277_h', 'abc278_a', 'abc278_b',
       'abc278_c', 'abc278_d', 'abc278_e', 'abc278_g', 'abc279_a', 'abc279_b', 'abc279_d', 'abc279_e', 'abc279_g',
       'abc280_b', 'abc280_c', 'abc280_d', 'abc280_e', 'abc281_a', 'abc281_b', 'abc281_d', 'abc281_f', 'abc281_h',
       'abc282_a', 'abc282_c', 'abc282_e', 'abc282_f', 'abc283_a', 'abc283_c', 'abc283_d', 'abc283_e', 'abc283_f',
       'abc283_g', 'abc283_h', 'abc284_c', 'abc284_d', 'abc284_e', 'abc284_f', 'abc284_g', 'abc285_a', 'abc285_c',
       'abc285_e', 'abc286_a', 'abc286_d', 'abc286_e', 'abc286_f', 'abc286_g', 'abc286_h', 'abc287_a', 'abc287_c',
       'abc287_d', 'abc287_e', 'abc287_f', 'abc287_g', 'abc287_h', 'abc288_b', 'abc288_d', 'abc288_e', 'abc288_f',
       'abc288_g', 'abc288_h', 'abc289_a', 'abc289_b', 'abc289_c', 'abc289_d', 'abc289_e', 'abc289_h', 'abc290_b',
       'abc290_c', 'abc290_d', 'abc290_g', 'abc291_a', 'abc291_b', 'abc291_c', 'abc291_d', 'abc291_e', 'abc291_f',
       'abc291_h', 'abc292_b', 'abc292_c', 'abc292_d', 'abc292_e', 'abc292_f', 'abc292_h', 'abc293_b', 'abc293_c',
       'abc293_e', 'abc293_f', 'abc294_a', 'abc294_b', 'abc294_d', 'abc294_e', 'abc294_f', 'abc294_g', 'abc295_a',
       'abc295_b', 'abc295_e', 'abc295_g', 'abc296_b', 'abc296_c', 'abc296_d', 'abc296_f', 'abc296_h', 'abc297_a',
       'abc297_b', 'abc297_c', 'abc297_d', 'abc297_e', 'abc297_f', 'abc297_g', 'abc298_a', 'abc298_b', 'abc298_c',
       'abc298_d', 'abc298_e', 'abc298_g', 'abc299_a', 'abc299_c', 'abc299_d', 'abc299_e', 'abc299_f', 'abc299_g',
       'abc299_h', 'abc300_a', 'abc300_b', 'abc300_c', 'abc300_d', 'abc300_e', 'abc300_f', 'abc300_g', 'abc301_a',
       'abc301_b', 'abc301_d', 'abc301_e', 'abc301_f', 'abc302_a', 'abc302_b', 'abc302_c', 'abc302_e', 'abc302_f',
       'abc302_g', 'abc303_a', 'abc303_b', 'abc303_c', 'abc303_d', 'abc303_e', 'abc303_f', 'abc303_g', 'abc304_a',
       'abc304_b', 'abc304_c', 'abc304_d', 'abc304_f', 'abc304_g', 'abc304_h', 'abc305_a', 'abc305_b', 'abc305_e',
       'abc306_b', 'abc306_e', 'abc306_g', 'abc307_a', 'abc307_b', 'abc307_c', 'abc307_d', 'abc307_e', 'abc307_f',
       'abc307_g', 'abc307_h', 'abc308_a', 'abc308_b', 'abc308_c', 'abc308_d', 'abc308_g', 'abc309_a', 'abc309_b',
       'abc309_c', 'abc309_d', 'abc309_e', 'abc309_f', 'abc309_g', 'abc310_a', 'abc310_b', 'abc310_c', 'abc310_d',
       'abc310_f', 'abc311_a', 'abc311_b', 'abc311_e', 'abc311_g', 'abc312_a', 'abc312_b', 'abc312_c', 'abc312_d',
       'abc312_f', 'abc313_a', 'abc313_b', 'abc313_c', 'abc313_d', 'abc313_e', 'abc313_f', 'abc313_g', 'abc314_a',
       'abc314_b', 'abc314_c', 'abc314_d', 'abc314_e', 'abc314_g', 'abc314_h', 'abc315_a', 'abc315_b', 'abc315_c',
       'abc315_f', 'abc315_h', 'abc317_a', 'abc317_c', 'abc317_d', 'abc317_e', 'abc317_f', 'abc318_a', 'abc318_b',
       'abc318_c', 'abc318_e', 'abc318_g', 'abc319_a', 'abc319_b', 'abc319_c', 'abc319_d', 'abc319_f', 'abc319_g',
       'abc320_a', 'abc320_b', 'abc320_c', 'abc320_d', 'abc320_f', 'abc320_g', 'abc321_a', 'abc321_b', 'abc321_d',
       'abc321_e', 'abc321_f', 'abc321_g', 'abc322_a', 'abc322_b', 'abc322_c', 'abc322_d', 'abc322_e', 'abc322_f',
       'abc323_a', 'abc323_b', 'abc323_c', 'abc323_f', 'abc324_a', 'abc324_b', 'abc324_d', 'abc324_e', 'abc324_g',
       'abc325_a', 'abc325_b', 'abc325_c', 'abc325_d', 'abc325_f', 'abc325_g', 'abc326_a', 'abc326_b', 'abc326_c',
       'abc326_d', 'abc326_f', 'abc326_g', 'abc327_a', 'abc327_b', 'abc327_c', 'abc327_e', 'abc327_g', 'abc328_a',
       'abc328_b', 'abc328_e', 'abc328_f', 'abc329_a', 'abc329_b', 'abc329_c', 'abc329_e', 'abc329_f', 'abc330_b',
       'abc330_c', 'abc330_f', 'abc331_a', 'abc331_b', 'abc331_c', 'abc331_d', 'abc331_e', 'abc332_a', 'abc332_b',
       'abc332_c', 'abc332_d', 'abc332_e', 'abc332_f', 'abc332_g', 'abc333_a', 'abc333_b', 'abc333_e', 'abc333_g',
       'abc334_a', 'abc334_b', 'abc334_c', 'abc334_e', 'abc334_f', 'abc334_g', 'abc335_a', 'abc335_b', 'abc335_c',
       'abc335_d', 'abc335_e', 'abc335_f', 'abc335_g', 'abc336_a', 'abc336_c', 'abc336_d', 'abc336_f', 'abc336_g',
       'abc337_a', 'abc337_b', 'abc337_c', 'abc337_d', 'abc337_e', 'abc337_f', 'abc338_a', 'abc338_b', 'abc338_c',
       'abc338_d', 'abc338_e', 'abc338_f', 'abc339_a', 'abc339_b', 'abc339_c', 'abc339_e', 'abc339_f', 'abc340_a',
       'abc340_b', 'abc340_c', 'abc340_d', 'abc340_f', 'abc340_g', 'abc341_a', 'abc341_b', 'abc341_d', 'abc341_e',
       'abc342_a', 'abc342_b', 'abc342_c', 'abc342_d', 'abc342_e', 'abc342_f', 'abc342_g', 'abc343_b', 'abc343_c',
       'abc343_d', 'abc343_e', 'abc343_g', 'abc344_a', 'abc344_b', 'abc344_c', 'abc344_d', 'abc345_a', 'abc345_b',
       'abc345_c', 'abc345_d', 'abc345_e', 'abc345_f', 'abc346_b', 'abc346_c', 'abc346_d', 'abc346_e', 'abc346_f',
       'abc346_g', 'abc347_a', 'abc347_b', 'abc347_c', 'abc347_d', 'abc347_e', 'abc347_f', 'abc347_g', 'abc348_a',
       'abc348_b', 'abc348_c', 'abc348_d', 'abc349_a', 'abc349_b', 'abc349_c', 'abc349_d', 'abc349_f', 'abc349_g',
       'abc350_a', 'abc350_b', 'abc350_c', 'abc350_d', 'abc350_f', 'abc350_g', 'abc351_b', 'abc351_c', 'abc352_a',
       'abc352_b', 'abc352_d', 'abc352_f', 'abc353_a', 'abc353_b', 'abc353_c', 'abc353_d', 'abc353_f', 'abc353_g',
       'abc354_a', 'abc354_b', 'abc354_c', 'abc354_d', 'abc354_e', 'abc354_f', 'abc354_g', 'abc355_a', 'abc355_b',
       'abc355_c', 'abc355_d', 'abc355_e', 'abc356_a', 'abc356_b', 'abc356_c', 'abc356_d', 'abc356_e', 'abc356_f',
       'abc356_g', 'abc357_a', 'abc357_b', 'abc357_c', 'abc357_d', 'abc357_e', 'abc357_f', 'abc358_a', 'abc358_b',
       'abc358_c', 'abc358_d', 'abc358_f', 'abc358_g', 'abc359_b', 'abc359_c', 'abc359_e', 'abc359_f', 'abc359_g',
       'abc360_a', 'abc360_b', 'abc360_c', 'abc360_d', 'abc360_e', 'abc360_f', 'abc360_g', 'agc055_a', 'agc056_c',
       'agc057_a', 'agc057_b', 'agc058_a', 'agc058_b', 'agc059_a', 'agc059_b', 'agc060_a', 'agc060_b', 'agc061_a',
       'agc062_a', 'agc062_c', 'agc063_a', 'agc063_b', 'agc063_c', 'agc064_a', 'agc065_a', 'agc065_c', 'agc066_a',
       'agc066_b', 'agc066_c', 'arc128_b', 'arc128_c', 'arc128_d', 'arc128_e', 'arc129_a', 'arc129_b', 'arc129_c',
       'arc129_d', 'arc130_b', 'arc130_f', 'arc131_a', 'arc131_b', 'arc131_c', 'arc131_d', 'arc131_e', 'arc132_a',
       'arc132_b', 'arc132_c', 'arc132_d', 'arc133_a', 'arc133_b', 'arc133_d', 'arc134_a', 'arc134_b', 'arc134_d',
       'arc135_a', 'arc135_b', 'arc135_c', 'arc135_d', 'arc136_a', 'arc136_b', 'arc136_c', 'arc136_d', 'arc137_a',
       'arc137_b', 'arc137_c', 'arc137_d', 'arc138_a', 'arc138_b', 'arc138_c', 'arc138_d', 'arc139_a', 'arc139_b',
       'arc139_c', 'arc139_d', 'arc140_a', 'arc140_b', 'arc140_c', 'arc140_d', 'arc140_e', 'arc141_a', 'arc141_b',
       'arc141_c', 'arc141_e', 'arc142_a', 'arc142_b', 'arc142_c', 'arc143_a', 'arc143_c', 'arc143_d', 'arc144_a',
       'arc144_b', 'arc145_a', 'arc145_b', 'arc145_c', 'arc145_d', 'arc145_e', 'arc146_a', 'arc146_b', 'arc146_c',
       'arc146_d', 'arc147_b', 'arc147_c', 'arc147_e', 'arc148_a', 'arc148_b', 'arc148_c', 'arc148_d', 'arc148_e',
       'arc149_a', 'arc149_c', 'arc149_d', 'arc150_a', 'arc150_b', 'arc150_e', 'arc151_a', 'arc151_b', 'arc151_c',
       'arc151_e', 'arc152_a', 'arc152_b', 'arc152_c', 'arc152_d', 'arc153_a', 'arc153_b', 'arc153_c', 'arc154_b',
       'arc154_c', 'arc154_d', 'arc155_a', 'arc155_b', 'arc155_c', 'arc156_a', 'arc156_b', 'arc156_c', 'arc156_d',
       'arc157_a', 'arc157_b', 'arc157_d', 'arc157_e', 'arc158_a', 'arc158_b', 'arc158_c', 'arc158_d', 'arc159_a',
       'arc159_b', 'arc159_c', 'arc159_d', 'arc160_a', 'arc160_b', 'arc160_c', 'arc161_a', 'arc161_b', 'arc161_c',
       'arc161_d', 'arc162_b', 'arc162_c', 'arc162_e', 'arc163_a', 'arc163_b', 'arc163_c', 'arc164_a', 'arc164_c',
       'arc164_d', 'arc164_e', 'arc165_a', 'arc165_b', 'arc165_c', 'arc165_d', 'arc166_a', 'arc166_b', 'arc166_c',
       'arc166_d', 'arc167_a', 'arc167_b', 'arc167_e', 'arc168_a', 'arc168_b', 'arc168_d', 'arc168_e', 'arc170_a',
       'arc170_b', 'arc170_c', 'arc170_d', 'arc171_a', 'arc171_b', 'arc171_c', 'arc171_d', 'arc172_a', 'arc172_b',
       'arc172_c', 'arc173_a', 'arc173_b', 'arc173_c', 'arc173_d', 'arc173_e', 'arc174_a', 'arc174_b', 'arc174_c',
       'arc174_d', 'arc174_e', 'arc175_a', 'arc175_b', 'arc175_c', 'arc175_d', 'arc175_e', 'arc176_a', 'arc176_b',
       'arc176_d', 'arc177_a', 'arc177_b', 'arc177_d', 'arc178_b', 'arc178_c', 'arc178_d', 'arc179_a', 'arc179_b',
       'arc179_d', 'arc180_a', 'arc180_b', 'arc180_c']

var2=os.listdir("/Users/tom/Downloads/ConDefects-4b05fb96a46baacf48be160621121412c795b1de/Code")

#read task date
task_date = {}
with open("./data/date.txt", 'r') as f:
    for line in f.readlines():
        if len(line.strip()) == 0:
            continue
        items = line.strip().split()
        task_date[items[0]] = items[1]

result=[]
for name in var:
    if is_date_in_range(task_date[name]) and name in var2:
        result.append(name)
def main1():
    print("2021-10-01~2023-09-30,","2023-09-30~,","2021-10-01~2023-09-30 old")
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
    # print([x for x in meta_info.keys() if is_date_in_range(date_info[x])])
    print("元信息",len([x for x in meta_info.keys() if is_date_in_range(date_info[x])]),len([x for x in meta_info.keys() if not is_date_in_range(date_info[x])]),len([x for x in meta_info.keys() if is_date_in_range(date_info[x]) and x in result]))
    print([x for x in meta_info.keys() if is_date_in_range(date_info[x]) and x in result])
    pass_result, fail_result = [], []
    tasks = set()
    with jsonlines.open("./data/condefects_mcts_result.jsonl", 'r') as reader:
        for line in reader:
            tasks.add(line['project'])
            if line["eval"] == "PASS":
                pass_result.append((line['project'], line['bug_id']))
            else:
                fail_result.append((line['project'], line['bug_id']))
    print("运行的",len([x for x in tasks if is_date_in_range(date_info[x])]),len([x for x in tasks if not is_date_in_range(date_info[x])]),len([x for x in tasks if is_date_in_range(date_info[x]) and x in result]))
    pass_num = 0
    pass_num_new = 0
    pass_num_old = 0
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
            if p in result:
                pass_num_old+=1
    print("修复的",pass_num, pass_num_new,pass_num_old)
    # print([x for x in meta_info.keys() if len(meta_info[x])==1])

def main2():
    with jsonlines.open("./data/condefects_mcts_result.jsonl", 'r') as reader,jsonlines.open("./data/condefects_mcts_result2.jsonl", 'w') as writer:
        for line in reader:
            if line["eval"]=="PASS":
                writer.write(line)

main1()



#
# print(len(result))
