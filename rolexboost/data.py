import numpy as np
import pandas as pd

name_list = [
    'breast-cancer-wisconsin-diagnose',
    'breast-cancer-wisconsin',
    'haberman'
]

def load_data(name='haberman'):
    data = np.load(f'..\data\{name}.npy')
    X = data[:,:-1]
    y = data[:,-1]
    print(X[:10])
    print(y[:10])
    print('Data Shape: ',X.shape)
    print('Label Shape: ',y.shape)
    return X, y 



# Data process


def process_haberman(path='..\data\haberman.data'):
    d = np.loadtxt(path, delimiter=',')
    np.save('..\data\haberman.npy',d)

def process_breast_cancer_wisconsin(path=r'../data/breast-cancer-wisconsin.data'):
    with open(path, 'r') as f:
        data = [[int(item) for item in line.strip().split(',')] for line in f if '?' not in line]
    data = np.array(data)
    data = data[:,1:] # 去掉id列，保留9个attribute，与paper统一
    print(data.shape)
    print(data[:10])
    for line in data: # malign is 2 & benign is 1
        if line[-1] == 2: line[-1] = 1
        else: line[-1] = 2
    np.save(r'../data/breast-cancer-wisconsin.npy',data)

def process_breast_cancer_wisconsin_diagnose(path=r'../data/breast-cancer-wisconsin-diagnose.data'):
    with open(path, 'r') as f:
        data = [line.strip().split(',')[1:] for line in f if '?' not in line]
    for line in data:
        if line[0] == 'M': line[0] = 2 # malign is 2 & benign is 1
        else: line[0] = 1
        for i in range(1,len(line)): line[i] = float(line[i])
    data = np.array(data)
    label = data[:,0:1]
    print(label[:3])
    data = data[:,1:]
    new_data = np.concatenate([data,label],axis=1)
    print(new_data.shape)
    np.save(r'../data/breast-cancer-wisconsin-diagnose.npy', new_data)

if __name__=='__main__':
    # process_haberman()
    # process_breast_cancer_wisconsin()
    process_breast_cancer_wisconsin_diagnose()
    X, y = load_data(name = name_list[0])