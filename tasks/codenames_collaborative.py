import os
import re
from tasks.base import Task, DATA_PATH
from prompts.codenames_collaborative import (
    standard_prompt_spymaster,
    cot_prompt_spymaster,
    spp_prompt_spymaster,
    spp_prompt_spymaster_fixed_persona,
    spp_prompt_spymaster_profile,
    standard_prompt_guesser,
    cot_prompt_guesser,
    spp_prompt_guesser,
    spp_prompt_guesser_fixed_persona,
    spp_prompt_guesser_profile
)
import json

class CodenamesCollaborativeTask(Task):
    def __init__(self, file='codenames_50.jsonl'):
        super().__init__()
        path = os.path.join(DATA_PATH, 'codenames_collaborative', file)
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int):
        return self.data[idx]

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        datapoint = self.data[idx]
        word_list = datapoint['word_list']
        word_list_str = ", ".join(word_list)
        target_words = datapoint['target_words']
        target_words_str = ", ".join(target_words)

        # for guesser
        assert 'role' in kwargs
        role = kwargs['role']
        if role == 'guesser':
            assert 'hint_word' in kwargs
            hint_word = kwargs['hint_word']
        else:
            hint_word = None

        n = len(target_words)
        if role == 'spymaster':
            if method == "standard":
                input_prompt = standard_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "cot":
                input_prompt = cot_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "spp":
                input_prompt = spp_prompt_spymaster.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "spp_fixed_persona":
                input_prompt = spp_prompt_spymaster_fixed_persona.format(n = n, target_words = target_words_str, word_list = word_list_str)
            elif method == "spp_profile":
                input_prompt = spp_prompt_spymaster_profile.format(n = n, target_words = target_words_str, word_list = word_list_str)
            else:
                raise NotImplementedError(f"method {method} not implemented for spymaster role")
        elif role == 'guesser':
            if method == "standard":
                input_prompt = standard_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "cot":
                input_prompt = cot_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "spp":
                input_prompt = spp_prompt_guesser.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "spp_fixed_persona":
                input_prompt = spp_prompt_guesser_fixed_persona.format(n = n, hint_word = hint_word, word_list = word_list_str)
            elif method == "spp_profile":
                input_prompt = spp_prompt_guesser_profile.format(n = n, hint_word = hint_word, word_list = word_list_str)
            else:
                raise NotImplementedError(f"method {method} not implemented for guesser role")
        else:
            raise NotImplementedError(f"role {role} not implemented; choose from 'spymaster' or 'guesser'")
        return input_prompt

    def test_output(self, idx: int, output: str):
        # test whether the output includes all the answers of the trivia questions
        datapoint = self.data[idx]
        target_words = datapoint['target_words']
        target_words = [word.strip().lower() for word in target_words]

        predicted_words = output.split(",")
        predicted_words = [word.strip().replace(".","").lower() for word in predicted_words]
        
        # ground truth set
        target_words_set = set(target_words)
        # predicted set
        predicted_words_set = set(predicted_words)
        
        common_words = predicted_words_set.intersection(target_words_set)
        common_words = list(common_words)
        info = {"matched_words":common_words, "matched_count":len(common_words), "target_count":len(target_words_set)}
        return info

    @staticmethod
    def prompt_unwrap(response: str, method: str):
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
                return response, True
        
        elif method in ["spp", "spp_profile", "spp_fixed_persona"]:
            if "Final answer:" in response:
                return response.split("Final answer:")[1].strip(), True
            elif "final answer:" in response:
                return response.split("final answer:")[1].strip(), True
            else:
                return response, False
        
        else:
            raise NotImplementedError(f"method {method} not implemented")