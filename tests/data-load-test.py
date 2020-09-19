import numpy as np
import pandas as pd

name_list = [
    'breast-cancer-coimbra',
    'breast-cancer-wisconsin-diagnose',
    'breast-cancer-wisconsin',
    'haberman',
    'heart-disease',
    'ILPD',
    'ionosphere',
    'monks',
    'parkinson',
    'somerville',
    'sonar',
    'spine',
    'transfusion'
]

def load_data(name='haberman'):
    data = np.load(f'..\data\{name}.npy')
    X = data[:,:-1]
    y = data[:,-1]
    return X, y 