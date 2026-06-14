import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()

prompt = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
client = InferenceClient(
    provider="featherless-ai",
    api_key=os.environ["HF_TOKEN"],
)

result = client.text_generation(
    prompt,
    model="codellama/CodeLlama-7b-hf",
    max_new_tokens=20
)

print(result)