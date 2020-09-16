class NotFittedException(Exception):
    def __init__(self, model):
        self.message = f"Model {model.__class__.__name__} has not been fitted."
