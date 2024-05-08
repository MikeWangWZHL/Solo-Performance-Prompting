import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
import logging  

from transformers import AutoTokenizer
import transformers
import torch
import uuid



# Configure logging  
logging.basicConfig(level=logging.INFO)  
  
# Error callback function
def log_retry_error(retry_state):  
    logging.error(f"Retrying due to error: {retry_state.outcome.exception()}")  



DEFAULT_GPT_CONFIG = {
    "engine": "devgpt4-32k",
    "temperature": 0.0,
    "max_tokens": 5000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

class OpenAIWrapper:
    def __init__(self, config = DEFAULT_GPT_CONFIG, system_message=""):
        # TODO: set up your API key with the environment variable OPENAIKEY
        openai.api_key = os.environ.get("OPENAI_API_KEY")      

        if os.environ.get("USE_AZURE")=="True":
            print("using azure api")
            openai.api_type = "azure"
        openai.api_base = os.environ.get("API_BASE")
        openai.api_version = os.environ.get("API_VERSION")

        self.config = config
        print("api config:", config, '\n')

        # count total tokens
        self.completion_tokens = 0
        self.prompt_tokens = 0

        # system message
        self.system_message = system_message # "You are an AI assistant that helps people find information."

    # retry using tenacity
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), retry_error_callback=log_retry_error)
    def completions_with_backoff(self, **kwargs):
        # print("making api call:", kwargs)
        # print("====================================")
        return openai.ChatCompletion.create(**kwargs)

    def run(self, prompt, n=1, system_message=""):
        """
            prompt: str
            n: int, total number of generations specified
        """
        try:
            # overload system message
            if system_message != "":
                sys_m = system_message
            else:
                sys_m = self.system_message
            if sys_m != "":
                # print("adding system message:", sys_m)
                messages = [
                    {"role":"system", "content":sys_m},
                    {"role":"user", "content":prompt}
                ]
            else:
                messages = [
                    {"role":"user","content":prompt}
                ]
            text_outputs = []
            raw_responses = []
            while n > 0:
                cnt = min(n, 10) # number of generations per api call
                n -= cnt
                res = self.completions_with_backoff(messages=messages, n=cnt, **self.config)
                text_outputs.extend([choice["message"]["content"] for choice in res["choices"]])
                # add prompt to log
                res['prompt'] = prompt
                if sys_m != "":
                    res['system_message'] = sys_m
                raw_responses.append(res)
                # log completion tokens
                self.completion_tokens += res["usage"]["completion_tokens"]
                self.prompt_tokens += res["usage"]["prompt_tokens"]

            return text_outputs, raw_responses
        except Exception as e:
            print("an error occurred:", e)
            return [], []

    def compute_gpt_usage(self):
        engine = self.config["engine"]
        if engine == "devgpt4-32k":
            cost = self.completion_tokens / 1000 * 0.12 + self.prompt_tokens / 1000 * 0.06
        else:
            cost = 0 # TODO: add custom cost calculation for other engines
        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens, "cost": cost}


DEFAULT_LLAMA2_CONFIG = {
    "task": "text-generation",
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "torch_dtype": torch.float16,
    "device_map": "auto",
    "do_sample": False
}

class Llama2Wrapper:
    def __init__(self, config = DEFAULT_LLAMA2_CONFIG):
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
        self.pipeline = transformers.pipeline(**config)
        self.config = config

    def run(self, prompt, n=1, system_message=""):
        #TODO: make this configurable
        sequences = self.pipeline(
            prompt,
            do_sample=self.config["do_sample"],
            num_return_sequences=n,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=3999,
        )
        # convert generation output into the same format as GPT raw response
        text_outputs = []
        raw_responses = []
        for seq in sequences:
            # remove prompt from the generated text
            gen_text = seq['generated_text'][len(prompt):]
            text_outputs.append(gen_text)
            mock_id = str(uuid.uuid4())
            mock_gpt_response_obj = {
                "id": mock_id,
                "object": "text-generation",
                "created": mock_id,
                "model": self.config["model"],
                "choices": [
                    {
                        "index":0,
                        "finish_reason": "stop",
                        "message":{
                            "role": "assistant",
                            "content":gen_text
                        }
                    }
                ],
                "usage": {},
                "prompt":prompt,
                "system_message":system_message
            }
            raw_responses.append(mock_gpt_response_obj)
        return text_outputs, raw_responses
    
    def compute_gpt_usage(self):
        return {}


if __name__ == "__main__":
    llama = Llama2Wrapper()
    prompt = '''I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'''
    text_outputs, raw_responses = llama.run(prompt)
    print(text_outputs)
    print('\n\n')
    print(raw_responses)
