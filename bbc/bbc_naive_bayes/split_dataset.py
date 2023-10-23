import random


def split_dataset(articles, labels, test_size=0.2):
    random.seed(42)

    num_samples = len(labels)
    indices = list(range(num_samples))
    random.shuffle(indices)

    test_samples = int(test_size * num_samples)

    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    X_train = [articles[i] for i in train_indices]
    X_test = [articles[i] for i in test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]

    return X_train, X_test, y_train, y_test
