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
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --token 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 --resume-download TechxGenus/starcoder2-3b-instruct --local-dir /root/autodl-tmp/starcoder2-3b-instruct
```

## QuixBugs-Java Result

#### API

| Model       | SL (Single Line) | SF (Single Function) | ALL          | 
|-------------|------------------|----------------------|--------------|
| GPT-4o-mini | 87.5 (28/32)     | 87.5 (7/8)           | 87.5 (35/40) | 
| GPT-4o      | 96.87 (31/32)    | 87.5 (7/8)           | 95 (38/40)   | 


#### 3B

| Model                               | SL (Single Line) | SF (Single Function) | ALL          | 
|-------------------------------------|------------------|----------------------|--------------|
| meta-llama/Llama-3.2-3B-Instruct    | 46.87 (15/32)    | 62.5 (5/8)           | 52.5 (21/40) |
| Qwen/Qwen2.5-Coder-3B-Instruct      | 68.75 (22/32)    | 62.5 (5/8)           | 67.5 (27/40) |
| stabilityai/stable-code-instruct-3b | 62.5 (20/32)     | 0 (0/8)              | 50 (20/40)   |
| MaziyarPanahi/calme-3.1-instruct-3b | 43.75 (14/32)    | 62.5 (5/8)           | 47.5 (19/40) |
| TechxGenus/starcoder2-3b-instruct   | 56.25 (18/32)    | 0 (0/8)              | 45 (18/40)   |

#### 7B
| Model                                    | SL (Single Line) | SF (Single Function) | ALL          | 
|------------------------------------------|------------------|----------------------|--------------|
| deepseek-ai/deepseek-coder-6.7b-instruct | 65.62 (21/32)    | 75 (6/8)             | 67.5 (27/40) | 
| Qwen/Qwen2.5-Coder-7B-Instruct           | 59.37 (19/32)    | 75 (6/8)             | 62.5 (25/40) |
| 01-ai/Yi-Coder-9B-Chat                   | 75 (24/32)       | 87.5 (7/8)           | 77.5 (31/40) |
| meta-llama/Llama-3.1-8B-Instruct         | 62.5 (20/32)     | 62.5 (5/8)           | 62.5 (25/40) |
| Deci/DeciLM-7B-instruct                  | 53.12 (17/32)    | 25 (2/8)             | 47.5 (19/40) |
| tiiuae/falcon-7b-instruct                | 12.5 (4/32)      | 0 (0/8)              | 10 (4/40)    |
| microsoft/Phi-3.5-mini-instruct          | 46.87 (15/32)    | 50 (4/8)             | 47.5 (19/40) |


## D4J Result

#### API

| Model       | SL (Single Line) | SH (Single Hunk) | SF (Single Function) | ALL             | 
|-------------|------------------|------------------|----------------------|-----------------|
| GPT-4o-mini | 43.37 (72/166)   | 34.21 (39/114)   | 32.36 (67/207)       | 36.55 (178/487) | 
| GPT-4o      | (/166)           | (/114)           | (/207)               | (/487)          |

#### 3B

| Model                               | SL (Single Line) | SH (Single Hunk) | SF (Single Function) | ALL    | 
|-------------------------------------|------------------|------------------|----------------------|--------|
| Qwen/Qwen2.5-Coder-3B-Instruct      | (/166)           | (/114)           | (/207)               | (/487) |
| meta-llama/Llama-3.2-3B-Instruct    | (/166)           | (/114)           | (/207)               | (/487) |
| stabilityai/stable-code-instruct-3b | (/166)           | (/114)           | (/207)               | (/487) |
| MaziyarPanahi/calme-3.1-instruct-3b | (/166)           | (/114)           | (/207)               | (/487) |
| TechxGenus/starcoder2-3b-instruct   | (/166)           | (/114)           | (/207)               | (/487) |

#### 7B

| Model                                    | SL (Single Line) | SH (Single Hunk) | SF (Single Function) | ALL             | 
|------------------------------------------|------------------|------------------|----------------------|-----------------|
| deepseek-ai/deepseek-coder-6.7b-instruct | 48.19 (80/166)   | 15.78 (18/114)   | 20.77 (43/207)       | 28.95 (141/487) | 
| Qwen/Qwen2.5-Coder-7B-Instruct           | 39.15 (65/166)   | 15.78 (18/114)   | 25.60 (53/207)       | 27.92 (136/487) | 
| 01-ai/Yi-Coder-9B-Chat                   | 48.79 (81/166)   | 28.07 (32/114)   | 31.40 (65/207)       | 36.55 (178/487) |
| meta-llama/Llama-3.1-8B-Instruct         | 42.77 (71/166)   | 21.05 (24/114)   | 28.98 (60/207)       | 31.82 (155/487) | 
| Deci/DeciLM-7B-instruct                  | 30.72 (51/166)   | 7.89 (9/114)     | 11.11 (23/207)       | 17.04 (83/487)  |
| tiiuae/falcon-7b-instruct                | 20.48 (34/166)   | 8.77 (10/114)    | 0.96 (2/207)         | 9.44 (46/487)   |
| microsoft/Phi-3.5-mini-instruct          | (/166)           | (/114)           | (/207)               | (/487)          |
