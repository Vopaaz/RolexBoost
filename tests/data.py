import numpy as np

data_attributes = {
    "breast-cancer-coimbra": {"rotation": 0.7263, "flex": 0.7681, "rolex": 0.7693},
    "breast-cancer-wisconsin-diagnose": {"rotation": 0.9508, "flex": 0.9698, "rolex": 0.9604},
    "breast-cancer-wisconsin": {"rotation": 0.9616, "flex": 0.9657, "rolex": 0.9770},
    "haberman": {"rotation": 0.7157, "flex": 0.7397, "rolex": 0.7561},
    "heart-disease": {"rotation": 0.8026, "flex": 0.8129, "rolex": 0.8187},
    "ILPD": {"rotation": 0.7016, "flex": 0.7025, "rolex": 0.7110},
    "ionosphere": {"rotation": 0.9341, "flex": 0.9334, "rolex": 0.9425},
    "monks": {"rotation": 0.6564, "flex": 0.6607, "rolex": 0.6725},
    "parkinson": {"rotation": 0.8738, "flex": 0.9099, "rolex": 0.9070},
    "somerville": {"rotation": 0.5858, "flex": 0.6014, "rolex": 0.6290},
    "sonar": {"rotation": 0.8094, "flex": 0.8276, "rolex": 0.8386},
    "spine": {"rotation": 0.8347, "flex": 0.8364, "rolex": 0.8503},
    "transfusion": {"rotation": 0.7788, "flex": 0.7952, "rolex": 0.7849},
    # The following datasets are 3-class problems, but they are splitted into 2-class problems in the paper,
    # and 3 accuracy scores are reported.
    # We are not sure about how to split the problem and compare the performance, so they are temperally not in use.
    #
    # "auto_mpg",
    # "cmc",
    # "iris",
    # "seed",
    # "wine",
}


def load_data(name: str):
    data = np.load(f"data/{name}.npy")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y
