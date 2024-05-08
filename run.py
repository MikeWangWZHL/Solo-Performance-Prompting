import os
import json
import argparse
from models import OpenAIWrapper, Llama2Wrapper
from tasks import get_task
import time
from configs import gpt_configs, llama_configs, default_gpt_config, default_llama_config


SLEEP_RATE = 10 # sleep between calls


def output_log_jsonl(log_file, all_logs):
    with open(log_file, "w") as f:
        for log in all_logs:
            f.write(json.dumps(log) + "\n")

def _post_process_raw_response(task, raw_output_batch, method, **kwargs):
    unwrapped_output_batch = []
    if_success_batch = []
    for output in raw_output_batch:
        unwrapped_output, if_success_flag = task.prompt_unwrap(output, method, **kwargs)
        unwrapped_output_batch.append(unwrapped_output)
        if_success_batch.append(if_success_flag)
    return unwrapped_output_batch, if_success_batch


### default task runners ###

def _get_response_default(model, task, i, method, num_generation, prompt, test_output=True, **kwargs):
    raw_output_batch, raw_response_batch = model.run(prompt=prompt, n=num_generation)
    if raw_output_batch == [] or raw_response_batch == []: # handle exception
        return {}    
    # get parsed response, and the success flags (whether or not the parsing is success) (standard prompt always success)
    unwrapped_output_batch, if_success_batch = _post_process_raw_response(task, raw_output_batch, method, **kwargs)
    # compute automatic metric (different for each task), e.g., if the output contains all the answers
    if test_output:
        test_output_infos = [task.test_output(i, output) for output in unwrapped_output_batch]
    else:
        test_output_infos = []
    # log output
    log_output = {
        "idx": i,
        "raw_response": raw_response_batch,
        "unwrapped_output": unwrapped_output_batch,
        "parsing_success_flag": if_success_batch,
        "test_output_infos": test_output_infos
    }
    return log_output

def _run_task_default(model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, test_output=True):
    # get prompt
    prompt = task.get_input_prompt(i, method=method)
    # get response and parsed output 
    return _get_response_default(model, task, i, method, num_generation, prompt, test_output=test_output)

def _run_task_codenames(model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, test_output=True):
    # get spymaster hint word
    spymaster_prompt = task.get_input_prompt(i, method=method, role='spymaster')
    raw_spymaster_output, raw_response_spymaster = model.run(prompt=spymaster_prompt, n=1)
    if raw_spymaster_output == [] or raw_response_spymaster == []: # handle exception
        return {}
    spymaster_output, if_success_batch_spymaster = _post_process_raw_response(task, raw_spymaster_output, method)
    hint_word = spymaster_output[0].replace(".", "").strip()
    print(f"\tidx: {i} | done spymaster, hint word: {hint_word}")
    # sleep before calling guesser
    time.sleep(sleep_rate)
    # get guesser result
    guesser_prompt = task.get_input_prompt(i, method=method, role='guesser', hint_word=hint_word)
    raw_guesser_output, raw_response_batch_guesser = model.run(prompt=guesser_prompt, n=num_generation)
    if raw_guesser_output == [] or raw_response_batch_guesser == []: # handle exception
        return {}
    guesser_output_batch, if_success_batch_guesser = _post_process_raw_response(task, raw_guesser_output, method)
    # compute automatic metric (different for each task), e.g., if the output contains all the answers
    if test_output:
        test_output_infos = [task.test_output(i, output) for output in guesser_output_batch]
    else:
        test_output_infos = []
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
    return log_output

##############################

### self_refine task runners ###

def _run_self_refine_default(model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, num_refine=1, **kwargs):
    print("\tidx:", i, "start self refine...")
    log_outputs = {}
    ## get initial response
    init_prompt = task.get_input_prompt(i, method=method, phase="init", **kwargs)
    init_output = _get_response_default(model, task, i, method, num_generation=1, prompt=init_prompt, test_output=True, phase="init")
    if init_output == {}:
        return {}
    log_outputs["answer_0"] = init_output

    time.sleep(sleep_rate)
    context_prompt = init_output['raw_response'][0]['prompt'] + "\n" + init_output["raw_response"][0]['choices'][0]['message']['content'] # Q + A0
    for j in range(num_refine):
        print("\t\tstep:", j)
        # get feedback
        feedback_prompt = task.get_input_prompt(i, method=method, phase="feedback", question_answer=context_prompt, **kwargs)
        feedback_output = _get_response_default(model, task, i, method, num_generation=1, prompt=feedback_prompt, test_output=False, phase="feedback")
        if feedback_output == {}:
            return log_outputs
        log_outputs[f"feedback_{j}"] = feedback_output
        time.sleep(sleep_rate)

        # get refined response
        refine_prompt = task.get_input_prompt(i, method=method, phase="refine", question_answer=context_prompt, feedback=feedback_output["unwrapped_output"][0], **kwargs) # Q + A0 + F
        refine_output = _get_response_default(model, task, i, method, num_generation=1, prompt=refine_prompt, test_output=True, phase="refine")
        if refine_output == {}:
            return log_outputs
        log_outputs[f"answer_{j+1}"] = refine_output
        time.sleep(sleep_rate)

        # update context
        context_prompt = refine_prompt + refine_output["raw_response"][0]['choices'][0]['message']['content'] # Q + A0 + F + A1

    return log_outputs

def _run_self_refine_codenames(model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, num_refine=1, test_output=True):
    # get spymaster hint word
    spy_master_log_outputs = _run_self_refine_default(model, task, i, method, num_generation, sleep_rate, num_refine, role='spymaster')
    if f"answer_{num_refine}" not in spy_master_log_outputs:
        return {}
    hint_word = spy_master_log_outputs[f"answer_{num_refine}"]["unwrapped_output"][0].replace(".", "").strip()
    print(f"\tidx: {i} | num_refine: {num_refine} | done spymaster, hint word: {hint_word}")
    # sleep before calling guesser
    time.sleep(sleep_rate)
    # get guesser result
    guesser_log_outputs = _run_self_refine_default(model, task, i, method, num_generation, sleep_rate, num_refine, role='guesser', hint_word=hint_word)
    if f"answer_{num_refine}" not in guesser_log_outputs:
        return {}
    guesser_output = guesser_log_outputs[f"answer_{num_refine}"]["unwrapped_output"][0]
    # compute automatic metric (different for each task), e.g., if the output contains all the answers
    if test_output:
        test_output_infos = [task.test_output(i, guesser_output)]
    else:
        test_output_infos = []
    # log output
    log_output = {
        "idx": i,
        "spymaster_logs": spy_master_log_outputs,
        "guesser_logs": guesser_log_outputs,
        "hint_word": hint_word,
        "parsing_success_flag_spymaster": spy_master_log_outputs[f"answer_{num_refine}"]["parsing_success_flag"],
        "parsing_success_flag_guesser": guesser_log_outputs[f"answer_{num_refine}"]["parsing_success_flag"],
        "test_output_infos": test_output_infos
    }
    return log_output
##############################



def _run_task(task_name, model, task, i, method, num_generation, sleep_rate=SLEEP_RATE, **kwargs):
    if task_name in ['trivia_creative_writing', 'logic_grid_puzzle']:
        if method == "self_refine":
            log_output = _run_self_refine_default(model, task, i, method, num_generation, sleep_rate, num_refine = kwargs['num_refine'])
        else:
            log_output = _run_task_default(model, task, i, method, num_generation, sleep_rate)
    elif task_name == 'codenames_collaborative':
        if method == "self_refine":
            log_output = _run_self_refine_codenames(model, task, i, method, num_generation, sleep_rate, num_refine = kwargs['num_refine'])
        else:
            log_output = _run_task_codenames(model, task, i, method, num_generation, sleep_rate)
    else:
        raise NotImplementedError(f"task {task_name} not implemented; please choose from ['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative']")

    # log everything else that is related
    if "llama_config" in args:
        args["llama_config"]["torch_dtype"] = str(args["llama_config"]["torch_dtype"])
    log_output.update(args)
    log_output.update({"task_data":task.get_input(i)})
    return log_output

def run(args):
    # get configs
    model_type = args['model_type']
    task_name = args['task']
    method = args['method']
    start_idx, end_idx = args['task_start_index'], args['task_end_index']
    task_data_file = args['task_data_file']
    num_generation = args['num_generation']
    
    output_dir = args['output_dir']
    if output_dir == "":
        output_dir = f"logs/{task_name}"

    additional_output_note = args['additional_output_note']
    system_message = args['system_message']
    print(f"setting default system message: {system_message}")
    
    # setup model and output log file
    if model_type == 'gpt':
        model_config = args['gpt_config']
        model = OpenAIWrapper(config=model_config, system_message=system_message)
        # setup log file
        model_name_for_output = model_config['engine'].replace("/", "-")
        if system_message == "":
            log_file = os.path.join(output_dir, f"{task_data_file}__method-{method}_engine-{model_name_for_output}_temp-{model_config['temperature']}_topp-{model_config['top_p']}_start{start_idx}-end{end_idx}{additional_output_note}__without_sys_mes.jsonl")
        else:
            log_file = os.path.join(output_dir, f"{task_data_file}__method-{method}_engine-{model_name_for_output}_temp-{model_config['temperature']}_topp-{model_config['top_p']}_start{start_idx}-end{end_idx}{additional_output_note}__with_sys_mes.jsonl")
        sleep_rate = SLEEP_RATE

    elif model_type == 'llama2':
        model_config = args['llama_config']
        model = Llama2Wrapper(config=model_config)
        # setup log file
        model_name_for_output = model_config['model'].replace("/", "-")
        log_file = os.path.join(output_dir, f"{task_data_file}__method-{method}_engine-{model_name_for_output}_start{start_idx}-end{end_idx}{additional_output_note}__without_sys_mes.jsonl")
        sleep_rate = 0

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # setup task
    task = get_task(task_name, file=task_data_file)
    
    all_logs = []
    print("start running ... log file:", log_file)
    print("sleep rate:", sleep_rate)

    print()
    start = max(start_idx, 0)
    end = min(end_idx, len(task))
    print("total num of instances:", end - start)
    print("method:", method)
    for i in range(start, end):
        log_output = _run_task(task_name, model, task, i, method, num_generation, sleep_rate, num_refine = args['num_refine'])
        all_logs.append(log_output)
        print("\tidx:", i, "done | usage so far:", model.compute_gpt_usage())
        # output log at each iteration
        output_log_jsonl(log_file, all_logs)
        # sleep
        time.sleep(sleep_rate)


def parse_args():
    model_choices = list(gpt_configs.keys()) + list(llama_configs.keys())
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, choices=model_choices, required=True)
    args.add_argument('--output_dir', type=str, required=False, default="")
    args.add_argument('--model_type', type=str, choices=['gpt','llama2'], default='gpt')
    args.add_argument('--method', type=str, choices=['standard','cot','spp','spp_profile', 'spp_fixed_persona', 'self_refine', 'spp_less_demo'], required=True)
    args.add_argument('--task', type=str, choices=['trivia_creative_writing', 'logic_grid_puzzle', 'codenames_collaborative'], required=True)
    args.add_argument('--task_data_file', type=str, required=True)
    args.add_argument('--task_start_index', type=int, required=True)
    args.add_argument('--task_end_index', type=int, required=True)
    args.add_argument('--num_generation', type=int, default=1)
    args.add_argument('--additional_output_note', type=str, default="")
    args.add_argument('--temperature', type=float, default=0.0)
    args.add_argument('--top_p', type=float, default=1.0)
    args.add_argument('--system_message', type=str, default="")
    args.add_argument('--num_refine', type=int, default=1) # Perform how many iterations of the self-refinement
    
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = vars(parse_args())
    model_name = args['model']
    model_type = args['model_type']
    
    ### gpt config ###
    if model_type == 'gpt':
        if model_name in gpt_configs:
            args['gpt_config'] = gpt_configs[model_name] # gpt configs
        else:
            args['gpt_config'] = default_gpt_config
            args['gpt_config']['engine'] = model_name

        # overwrite temperature and top_p
        args['gpt_config']['temperature'] = args['temperature']
        args['gpt_config']['top_p'] = args['top_p']
    
    elif model_type == 'llama2':
        ### llama config ###
        if model_name in llama_configs:
            args['llama_config'] = llama_configs[model_name] # llama configs
        else:
            args['llama_config'] = default_llama_config
            args['llama_config']['model'] = model_name

    print("run args:", args)
    run(args)