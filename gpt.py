from openai import OpenAI
import tiktoken
from utils import *

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") 
token_statistics_file="./token_statistics.txt"


def make_model(base_url="https://api5.xhub.chat/v1", api_key="sk-LHj3bkbrBiTyPKpsR6SrWi8zCaW5xTlkWMHCDZ2BR96U1iZx"):
    _client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    return _client

gpt_client = make_model()

def generate_patches(prompt, model_name="gpt-4o-mini", timeout=100, retry=100, num_samples=1):
    # prompt token数量
    prompt_tokens = encoding.encode(str(prompt))
    prompt_token_count = len(prompt_tokens)
    write_line_to_txt(token_statistics_file,f"input {prompt_token_count} tokens")

    for i in range(retry):
        completion = gpt_client.chat.completions.create(
            model=model_name,
            messages=prompt,
            max_tokens=8000,
            temperature=0.95,
            n=num_samples,
            timeout=timeout
        )
        if not completion.choices or len(completion.choices) == 0:
            continue
        else:
            texts = [x.message.content for x in completion.choices]
            # 响应token数量
            response_token_count=[len(encoding.encode(str(x))) for x in texts]
            write_line_to_txt(token_statistics_file,f"output {str(response_token_count)} tokens")
            return texts
    print("No reply from GPT")
    return ""


def generate(prompt, model_name="gpt-4o-mini", timeout=100, retry=100):
     # prompt token数量
    prompt_tokens = encoding.encode(str(prompt))
    prompt_token_count = len(prompt_tokens)
    write_line_to_txt(token_statistics_file,f"input {prompt_token_count} tokens")
    for i in range(retry):
        completion = gpt_client.chat.completions.create(
            model=model_name,
            messages=prompt,
            max_tokens=8000,
            temperature=0.0,
            timeout=timeout
        )
        if not completion.choices or len(completion.choices) == 0:
            continue
        else:
            text = completion.choices[0].message.content
            # 响应token数量
            response_token_count=len(encoding.encode(str(text))) 
            write_line_to_txt(token_statistics_file,f"output {response_token_count} tokens")
            return text
    print("No reply from GPT")
    return ""
