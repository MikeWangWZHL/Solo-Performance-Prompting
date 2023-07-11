import os
import json
import argparse
from models import OpenAIWrapper
from tasks import get_task
import time


SLEEP_RATE = 30 # sleep between calls


def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def _post_process_raw_response(task, raw_output_batch, method):
    unwrapped_output_batch = []
    if_success_batch = []
    for output in raw_output_batch:
        unwrapped_output, if_success_flag = task.prompt_unwrap(output, method)
        unwrapped_output_batch.append(unwrapped_output)
        if_success_batch.append(if_success_flag)
    return unwrapped_output_batch, if_success_batch

def _run_task(task_name, gpt, task, i, method, num_generation):
    if task_name in ['trivia_creative_writing', 'logic_grid_puzzle']:
        # get prompt
        prompt = task.get_input_prompt(i, method=method)
        # get raw response
        raw_output_batch, raw_response_batch = gpt.run(prompt=prompt, n=num_generation)
        if raw_output_batch == [] or raw_response_batch == []: # handle exception
            return {}    
        # get parsed response, and the success flags (whether or not the parsing is success) (standard prompt always success)
        unwrapped_output_batch, if_success_batch = _post_process_raw_response(task, raw_output_batch, method)
        # compute automatic metric (different for each task), e.g., if the output contains all the answers
        test_output_infos = [task.test_output(i, output) for output in unwrapped_output_batch]
        # log output
        log_output = {
            "idx": i,
            "raw_response": raw_response_batch,
            "unwrapped_output": unwrapped_output_batch,
            "parsing_success_flag": if_success_batch,
            "test_output_infos": test_output_infos
        }
    elif task_name == 'codenames_collaborative':
        # get spymaster hint word
        spymaster_prompt = task.get_input_prompt(i, method=method, role='spymaster')
        raw_spymaster_output, raw_response_spymaster = gpt.run(prompt=spymaster_prompt, n=1)
        # raw_spymaster_output, raw_response_spymaster = gpt.run(prompt=spymaster_prompt, n=1, system_message="You are an AI assistant that plays the Spymaster role in Codenames.")
        if raw_spymaster_output == [] or raw_response_spymaster == []: # handle exception
            return {}
        spymaster_output, if_success_batch_spymaster = _post_process_raw_response(task, raw_spymaster_output, method)
        hint_word = spymaster_output[0].replace(".", "").strip()
        print(f"\tidx: {i} | done spymaster, hint word: {hint_word}")
        # sleep before calling guesser
        time.sleep(SLEEP_RATE)
        # get guesser result
        guesser_prompt = task.get_input_prompt(i, method=method, role='guesser', hint_word=hint_word)
        raw_guesser_output, raw_response_batch_guesser = gpt.run(prompt=guesser_prompt, n=num_generation)
        # raw_guesser_output, raw_response_batch_guesser = gpt.run(prompt=guesser_prompt, n=num_generation, system_message="You are an AI assistant that plays the Guesser role in Codenames.")
        if raw_guesser_output == [] or raw_response_batch_guesser == []: # handle exception
            return {}
        guesser_output_batch, if_success_batch_guesser = _post_process_raw_response(task, raw_guesser_output, method)
        # compute automatic metric (different for each task), e.g., if the output contains all the answers
        test_output_infos = [task.test_output(i, output) for output in guesser_output_batch]
        # log output
        log_output = {
            "idx": i,
            "raw_response_spymaster": raw_response_spymaster,
            "raw_response_guesser": raw_response_batch_guesser,
            "spymaster_output": spymaster_output,
            "guesser_output": guesser_output_batch,
            "hint_word": hint_word,
            "parsing_success_flag_spymaster": if_success_batch_spymaster,
            "parsing_success_flag_guesser": if_success_batch_guesser,
            "test_output_infos": test_output_infos
        }
    else:
        raise NotImplementedError(f"task {task_name} not implemented; please choose from ['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative']")

    # log everything else that is related
    log_output.update(args)
    log_output.update({"task_data":task.get_input(i)})
    return log_output

def run(args):
    # get configs
    gpt_config = args['gpt_config']
    task_name = args['task']
    method = args['method']
    start_idx, end_idx = args['task_start_index'], args['task_end_index']
    task_data_file = args['task_data_file']
    num_generation = args['num_generation']

    additional_output_note = args['additional_output_note']
    system_message = args['system_message']
    print(f"setting default system message: {system_message}")
    
    # setup gpt api
    gpt = OpenAIWrapper(config=gpt_config, system_message=system_message)

    # setup log file
    if system_message == "":
        log_file = f"logs/{task_name}/{task_data_file}__method-{method}_engine-{gpt_config['engine']}_temp-{gpt_config['temperature']}_topp-{gpt_config['top_p']}_start{start_idx}-end{end_idx}{additional_output_note}__without_sys_mes.jsonl"
    else:
        log_file = f"logs/{task_name}/{task_data_file}__method-{method}_engine-{gpt_config['engine']}_temp-{gpt_config['temperature']}_topp-{gpt_config['top_p']}_start{start_idx}-end{end_idx}{additional_output_note}__with_sys_mes.jsonl"
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # setup task
    task = get_task(task_name, file=task_data_file)
    
    all_logs = []
    print("start running ... log file:", log_file)

    print()
    start = max(start_idx, 0)
    end = min(end_idx, len(task))
    print("total num of instances:", end - start)
    for i in range(start, end):
        log_output = _run_task(task_name, gpt, task, i, method, num_generation)
        all_logs.append(log_output)
        print("\tidx:", i, "done | usage so far:", gpt.compute_gpt_usage())
        # output log at each iteration
        output_log_jsonl(log_file, all_logs)
        # sleep
        time.sleep(SLEEP_RATE)



# TODO: add your custom model config here:
gpt_configs = {
    "gpt4-32k": {
        "engine": "devgpt4-32k",
        "temperature": 0.0,
        "max_tokens": 5000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    }
}

default_gpt_config = {
    "engine": None,
    "temperature": 0.0,
    "max_tokens": 5000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": None
}

def parse_args():
    model_choices = list(gpt_configs.keys())
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=model_choices, required=True)
    args.add_argument('--method', type=str, choices=['standard','cot','spp','spp_profile', 'spp_fixed_persona'], required=True)
    args.add_argument('--task', type=str, choices=['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative'], required=True)
    args.add_argument('--task_data_file', type=str, required=True)
    args.add_argument('--task_start_index', type=int, required=True)
    args.add_argument('--task_end_index', type=int, required=True)
    args.add_argument('--num_generation', type=int, default=1)
    args.add_argument('--additional_output_note', type=str, default="")
    args.add_argument('--temperature', type=int, default=0.0)
    args.add_argument('--top_p', type=int, default=1.0)
    args.add_argument('--system_message', type=str, default="")
    
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = vars(parse_args())
    model_name = args['model']
    
    if model_name in gpt_configs:
        args['gpt_config'] = gpt_configs[model_name] # our configs
    else:
        args['gpt_config'] = default_gpt_config
        args['gpt_config']['engine'] = model_name
    
    # overwrite temperature and top_p
    args['gpt_config']['temperature'] = args['temperature']
    args['gpt_config']['top_p'] = args['top_p']
    print("run args:", args)
    
    run(args)