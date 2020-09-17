class NotFittedException(Exception):
    def __init__(self, model):
        self.message = f"Model {model.__class__.__name__} has not been fitted."


class InsufficientDataException(Exception):
    def __init__(self, n_features_per_subset, n_row):
        self.message = f"Model requires the number of rows to be greater or equal to the 'n_features_per_subset', "
        f"which is set to {n_features_per_subset}, but we only have {n_row} row(s)."

class DimNotMatchException(Exception):
    def __init__(self, mat_dim, idx_dim):
        self.message = f"The dimension (number of rows) of the matrix should match the length of the index, "
        f"but they are {mat_dim} and {idx_dim} now."
