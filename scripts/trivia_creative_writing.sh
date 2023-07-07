
MODEL="gpt4-32k" # ['gpt3.5-4k','gpt4-32k','gpt4-8k']

DATA_FILE="trivia_creative_writing_100_n_5.jsonl" # ['trivia_creative_writing_100_n_5.jsonl', 'trivia_creative_writing_100_n_10.jsonl']

START_IDX=0
END_IDX=100

# choose method
METHOD="spp" # ['standard','cot','spp', 'spp_profile', 'spp_fixed_persona']

# w/ or w/o system message (spp works better w/ system message)
SYSTEM_MESSAGE="You are an AI assistant that helps people find information." # or "" (empty string)

python run.py \
    --model ${MODEL} \
    --method ${METHOD} \
    --task trivia_creative_writing \
    --task_data_file ${DATA_FILE} \
    --task_start_index ${START_IDX} \
    --task_end_index ${END_IDX} \
    --system_message "${SYSTEM_MESSAGE}" \
    ${@}

