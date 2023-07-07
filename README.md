# Repo for paper [Solo Performance Prompting]()

## Setup
- Install dependencies
    ```
    pip install -r requirements.txt
    ```
- Set up OpenAI API key and store in environment variable `OPENAI_API_KEY`
- Config OpenAI API deployment following Line 39-44 in `model.py`
- Config model/engine by adding an entry in `gpt_config` (line 139-168) in `run.py`

## Quick Start
We provide running scripts for each of the three tasks, please check out the comments in the ".sh" scripts for more information: 
- Trivia Creative Writing: `bash scripts/trivia_creative_writing.sh`
- Codenames Collaborative: `bash scripts/codenames_collaborative.sh`
- Logic Grid Puzzle: `bash scripts/logic_grid_puzzle.sh`

## Prompts
All prompts can be found in the `prompts/` folder. 

## Paper Experiment Results
Experimental results in the paper for each task can be found in the `logs/` folder. Each task has two subdirs `w_sys_mes` and `wo_sys_mes` indicating the two inference settings: with and without the system message: "You are an AI assistant that helps people find information.".

## Citations
Please cite the paper and star this repo if you find this work interesting/helpful.
```
```

## Acknowledgements
This codebase referenced the structure of the [Tree-of-thought official repo](https://github.com/princeton-nlp/tree-of-thought-llm). We thank the authors for their open-sourcing efforts.

