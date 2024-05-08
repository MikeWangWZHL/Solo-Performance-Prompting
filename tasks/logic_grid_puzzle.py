import os
import re
from tasks.base import Task, DATA_PATH
from prompts.logic_grid_puzzle import *
import json


target_aliases = {
    "1": "first",
    "2": "second",
    "3": "third",
    "4": "fourth",
    "5": "fifth",
    "6": "sixth",
    "7": "seventh",
    "8": "eighth",
    "9": "ninth",
    "10": "tenth"
}

class LogicGridPuzzleTask(Task):
    def __init__(self, file='logic_grid_puzzle_200.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'logic_grid_puzzle', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        input_str = datapoint['inputs']
        
        input_str = input_str.replace("\nA:", "")

        if method == "standard":
            input_prompt = standard_prompt.format(input=input_str)
        elif method == "cot":    
            input_prompt = cot_prompt.format(input=input_str)
        elif method == "spp":
            input_prompt = spp_prompt.format(input=input_str)
        elif method == "spp_less_demo":
            input_prompt = spp_prompt_less_demo.format(input=input_str)
        elif method == "spp_fixed_persona":
            input_prompt = spp_prompt_fixed_persona.format(input=input_str)
        elif method == "spp_profile":
            input_prompt = spp_prompt_profile.format(input=input_str)
        elif method == "self_refine":
            phase = kwargs["phase"]
            if phase == "init":
                input_prompt = standard_prompt.format(input=input_str)
            elif phase == "feedback":
                input_prompt = self_refine_feedback_prompt.format(question_answer=kwargs["question_answer"])
            elif phase == "refine":
                input_prompt = self_refine_refinement_prompt.format(question_answer=kwargs["question_answer"], feedback=kwargs["feedback"])
        else:
            raise NotImplementedError(f"method {method} not implemented")
        
        return input_prompt

    def test_output(self, idx: int, output: str):
        # test whether the output includes all the answers of the trivia questions
        instance = self.data[idx]
        target = instance["targets"][0]
        targets = [target]
        if target in target_aliases:
            targets.append(target_aliases[target])
        
        # get all other candidates
        not_targets = []
        for i in range(1, 11):
            if str(i) not in targets:
                not_targets.append(str(i))
                not_targets.append(target_aliases[str(i)])
        # print("targets", targets)
        # print("negatives", not_targets)
        info = {'correct': False}
        for target in targets:
            if target.lower().strip() in output.lower().strip(): # if the target is in the output
                info['correct'] = True
                # and if all the other targets are not in the output
                for not_target in not_targets:
                    if not_target.lower().strip() in output.lower().strip():
                        info['correct'] = False
                        break
                break
        return info

    @staticmethod
    def prompt_unwrap(response: str, method: str, **kwargs):
        '''
            response: raw genration from the model
            return:
                - str: the story
                - bool: whether the story is successfully parsed from the raw genration
        '''
        # take only the first few characters (enough for successfully parsed output) -> aviod unparsed result to have high accuracy when test output
        if method in ["standard", "cot"]:
            if "Answer:" in response:
                return response.split("Answer:")[1].strip(), True
            elif "answer:" in response:
                return response.split("answer:")[1].strip(), True
            else:
                return response, False
        
        elif method in ["spp", "spp_profile", "spp_fixed_persona", "spp_less_demo"]:
            if "Final answer:" in response:
                return response.split("Final answer:")[1].strip(), True
            elif "final answer:" in response:
                return response.split("final answer:")[1].strip(), True
            else:
                return response, False
        
        elif method == "self_refine":
            phase = kwargs["phase"]
            if phase == "feedback":
                return response, True
            else:
                if "Answer:" in response:
                    return response.split("Answer:")[1].strip(), True
                elif "answer:" in response:
                    return response.split("answer:")[1].strip(), True
                else:
                    return response, False
        else:
            raise NotImplementedError(f"method {method} not implemented")