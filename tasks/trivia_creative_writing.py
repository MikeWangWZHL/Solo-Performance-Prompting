import os
import re
from tasks.base import Task, DATA_PATH
from prompts.trivia_creative_writing import standard_prompt, cot_prompt, spp_prompt, spp_prompt_profile, spp_prompt_fixed_persona
import json
# from models import gpt

class TriviaCreativeWritingTask(Task):
    def __init__(self, file='trivia_creative_writing_100_n_5.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'trivia_creative_writing', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        questions = datapoint["questions"]
        topic = datapoint["topic"]
        n = len(questions)
        questions_str = " ".join(questions)
        
        if method == "standard":
            input_prompt = standard_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "cot":
            input_prompt = cot_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "spp":
            input_prompt = spp_prompt.format(n=n, questions=questions_str, topic=topic)
        elif method == "spp_fixed_persona":
            input_prompt = spp_prompt_fixed_persona.format(n=n, questions=questions_str, topic=topic)
        elif method == "spp_profile":
            input_prompt = spp_prompt_profile.format(n=n, questions=questions_str, topic=topic)
        else:
            raise NotImplementedError(f"method {method} not implemented")
        
        return input_prompt

    def test_output(self, idx: int, output: str):
        # test whether the output includes all the answers of the trivia questions
        instance = self.data[idx]
        correct_count = 0
        question_count = len(instance["answers"])
        for ans_to_question in instance["answers"]:
            for ans in ans_to_question:
                # compare all to lower
                if ans.lower() in output.lower():
                    correct_count += 1
                    break
        info = {'correct_count': correct_count, 'question_count': question_count}
        return info

    @staticmethod
    def prompt_unwrap(response: str, method: str):
        '''
            response: raw genration from the model
            return:
                - str: the story
                - bool: whether the story is successfully parsed from the raw genration
        '''
        if method == "standard":
            return response, True
        
        elif method == "cot":
            if "Story:" in response:
                return response.split("Story:")[1].strip(), True
            elif "story:" in response:
                return response.split("story:")[1].strip(), True
            else:
                return response, False
        
        elif method in ["spp","spp_profile","spp_fixed_persona"]:
            if "Final answer:" in response:
                return response.split("Final answer:")[1].strip(), True
            elif "final answer:" in response:
                return response.split("final answer:")[1].strip(), True
            else:
                return response, False
        
        else:
            raise NotImplementedError(f"method {method} not implemented")