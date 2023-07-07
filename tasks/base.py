DATA_PATH = './data'

class Task:
    def __init__(self):
        pass

    def __len__(self) -> int:
        pass

    def get_input_prompt(self, idx: int, method: str, **kwargs) -> str:
        pass

    def test_output(self, idx: int, output: str):
        pass