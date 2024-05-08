MODEL="gpt4-32k" # your engine name: gpt4-32k, gpt35-turbo, or meta-llama/Llama-2-13b-chat-hf
MODEL_TYPE="gpt" # 'gpt' or 'llama2'

DATA_FILE="logic_grid_puzzle_200.jsonl"

START_IDX=0
END_IDX=200

METHOD="spp" # ['standard','cot','spp', 'spp_profile', 'spp_fixed_persona']

# w/ or w/o system message (spp works better w/o system message)
SYSTEM_MESSAGE="" # or e.g., "You are an AI assistant that helps people find information."

python run.py \
    --model ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --method ${METHOD} \
    --task logic_grid_puzzle \
    --task_data_file ${DATA_FILE} \
    --task_start_index ${START_IDX} \
    --task_end_index ${END_IDX} \
    --system_message "${SYSTEM_MESSAGE}" \
    ${@}

