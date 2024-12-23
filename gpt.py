from openai import OpenAI


def make_model(base_url="https://api.nextapi.fun/v1", api_key=""):
    _client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    return _client


gpt_client = make_model()


def generate_patches(prompt, model_name="gpt-4o-mini", timeout=10, retry=10, num_samples=1):
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
            return texts
    print("No reply from GPT")
    return ""


def generate(prompt, model_name="gpt-4o-mini", timeout=10, retry=10):
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
            return text
    print("No reply from GPT")
    return ""
