def get_task(name, file=None):
    if name == 'trivia_creative_writing':
        from .trivia_creative_writing import TriviaCreativeWritingTask
        return TriviaCreativeWritingTask(file)
    elif name == 'logic_grid_puzzle':
        from .logic_grid_puzzle import LogicGridPuzzleTask
        return LogicGridPuzzleTask(file)
    elif name == 'codenames_collaborative':
        from .codenames_collaborative import CodenamesCollaborativeTask
        return CodenamesCollaborativeTask(file)
    else:
        raise NotImplementedError