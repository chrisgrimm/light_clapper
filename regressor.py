import numpy as np
import dill as pickle
from sklearn.svm import SVC

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = []
        for x in f.readlines():
            data_line = [float(y) for y in x.split(',')]
            data.append(data_line)
    mode_length = sorted([len(x) for x in data])[len(data)//2]
    data = [x for x in data if len(x) == mode_length]
    data = np.array(data)
    print(data.shape)
    return data



def train_classifier():
    clf = SVC()
    negative_examples = load_data('loud_noises.txt')
    negative_classes = [0 for _ in range(len(negative_examples))]
    positive_examples = load_data('claps.txt')
    positive_classes = [1 for _ in range(len(positive_examples))]

    X = np.concatenate([positive_examples, negative_examples], axis=0)
    X /= 100000000000.
    y = np.concatenate([positive_classes, negative_classes], axis=0)
    print(X.shape, y.shape)
    clf.fit(X, y)
    return clf


def save_classifier(clf):
    with open('clf.pickle', 'wb') as f:
        pickle.dump(clf, f)


def load_classifier():
    with open('clf.pickle', 'rb') as f:
        clf = pickle.load(f)
    return clf


if __name__ == '__main__':
    clf = train_classifier()

    save_classifier(clf)