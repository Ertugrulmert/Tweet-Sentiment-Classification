class Runner:
    """Runner class
    
    Encapsulates all the behavior needed to run experiments.
    The train() method fits model parameters to the task at hand and outputs model checkpoints
    The eval() method generates the submission file
    """
    def train():
        raise NotImplementedError()

    def evaluate():
        raise NotImplementedError()