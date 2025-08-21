## Requirements
+ Gradle: 8.11.1
```shell
wget https://services.gradle.org/distributions/gradle-8.11.1-bin.zip
unzip gradle-8.11.1-bin.zip
sudo vim ~/.bashrc
# add following line to .bashrc
export GRADLE_HOME=/root/autodl-tmp/gradle-8.11.1
export PATH=${PATH}:${GRADLE_HOME}/bin

source ~/.bashrc
# check gradle version
gradle -v
```
+ HuggingFace
```shell
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --token $HF_TOKEN --resume-download $MODEL_NAME --local-dir $LOCAL_DIR
```

+ Defects4j
```shell
git clone https://github.com/rjust/defects4j
cd defects4j
cpanm --installdeps .
./init.sh
export PATH=$PATH:"path2defects4j"/framework/bin
```

+ ConDefects
  + [paper](https://arxiv.org/abs/2310.16253)
  + [code](https://github.com/appmlk/ConDefects)
  + git clone & replace the original files with ConDefects_Replace to support multi-threaded testing.


+ other artifacts needed to install
  + less 
  + expect 
  + svn

## Run

+ Defects4J (APRMCTS)
```shell

python mcts.py \
    --policy_model gpt-4o-mini \
    --max_rollout 16 \
    --max_expansion 3 \
    --branch 1 \
    --exploration_constant 0.7 \
    --logger logs/mcts.log \
    --output_file results/result.jsonl

```

+ Defects4J (Beam Search + Patch Pool)
```shell
python beam_search.py \
    --policy_model qwen2.5-coder-32b \
    --reward_model qwen2.5-coder-32b \
    --output_path ./data/beam_search_results.jsonl \
    --plausible_save_path ./plausible \
    --beam_width 5 \
    --max_tokens 16000 \
    --max_iter 4 \
    --pool_size 4 \
```

+ QuixBugs

```shell
Python quixbugs_repair.py
```

+ ConDefects

```shell
python condefects_mcts.py \
    --policy_model gpt-4o-mini \
    --max_rollout 32 \
    --max_expansion 3 \
    --branch 10 \
    --exploration_constant 0.7 \
    --program_id xxx \
    --task_id xxx\
    --mask_mode unmasked \
    --logger logs/mcts.log \
    --output_file results/condefects_result.jsonl
```

+ SWE-bench lite
```shell
# prepare data
python swe_utils.py
# run swe
python swe_mcts.py
```

## Results

| Method                                                   | Model         | Patch Size | Defects4J-v1.2 | Defects4J-v2 | Total   | QuixBugs |
|----------------------------------------------------------|---------------|------------|----------------|--------------|---------|----------|
| [SelfAPR](https://arxiv.org/abs/2203.12755)              | T5            | 150        | 65/74          | 45/47        | 110/121 | -        |
| [CURE](https://arxiv.org/abs/2103.00073)                 | GPT-2         | 5000       | 57/-           | 19/-         | 76/-    | 26       |
| [RAPGen](https://arxiv.org/abs/2309.06057)               | CodeT5        | -          | 72/-           | 53/-         | 125/-   | -        |
| [RewardRepair](https://arxiv.org/abs/2105.04123)         | Transformer   | 200        | 45/-           | 45/-         | 90/-    | 20       |
| [Recoder](https://arxiv.org/abs/2106.08253)              | TreeGen       | 100        | 53/-           | 19/-         | 72/-    | 31       |
| [Repatt](https://ieeexplore.ieee.org/document/10457332/) | -             | 1200       | 40/70          | 35/68        | 75/138  | -        |
| [GAMMA](https://arxiv.org/abs/2309.09308)                | ChatGPT       | 250        | 82/108         | 45/-         | 127/-   | 22       |
| Incoder                                                  | Incoder       | -          | 78/-           | 39/-         | 117/-   | 37       |
| [ChatRepair](https://arxiv.org/abs/2304.00385)           | ChatGPT       | 500        | 114/-          | 48/-         | 162/-   | 40       |
| [RepairAgent](https://arxiv.org/abs/2403.17134)          | GPT-3.5  | 117        | 92/96          | 72/90        | 164/186 | -        |
| **APRMCTS (GPT-4o-mini)**                                | GPT-4o-mini   | 16         | 80/108         | 78/100       | 158/208 | 40       |
| **APRMCTS (GPT-3.5, 16 patch)**                    | GPT-3.5 | 16         | 86/112         | 73/104       | 159/216 | 40       |
| **APRMCTS (GPT-3.5, 32 patch)**                    | GPT-3.5 | 32         | 108/146        | 93/134       | 201/280 | 40       |
| **APRMCTS (GPT-3.5, 500 patch)**                    | GPT-3.5 | 500         | 112/153        | 100/143       | 212/296 | 40       |

