# Feature extraction: Bag of words

import numpy as np
import random
import matplotlib.pyplot as plt

# Read the bbc.classes file
with open('../bbc_data/bbc.classes', 'r') as file:
    classes_data = file.readlines()

# Extract the class labels
class_labels = []
for line in classes_data[4:]:
    label = line.strip().split()[1]
    class_labels.append(int(label))
    
# Read the bbc.mtx file
with open('../bbc_data/bbc.mtx', 'r') as file:
    mtx_data = file.readlines()

# Extract the matrix data
matrix_data = list({int(line.strip().split()[0]):{} for line in mtx_data[3:]}.values())
for line in mtx_data[3:]:
    row, col, value = line.strip().split()
    row = int(row)-1
    col = int(col)-1
    value = float(value)
    matrix_data[row].update({col: value})

# Vectorization
def vectorize(matrix_data, class_labels):
    feature_vectors = []
    for i in range(len(class_labels)):
        vector = []
        for j in range(len(matrix_data)):
            vector.append(matrix_data[j][i] if i in matrix_data[j] else 0.0)
        feature_vectors.append(vector)
    return feature_vectors

# Split the Data (with shuffling)
def split_data(feature_vectors, labels, train_ratio=0.8):
    data = list(zip(feature_vectors, labels))
    random.shuffle(data)

    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)

    return X_train, y_train, X_test, y_test

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Model Training (Logistic Regression - One-vs-Rest)
def train_logistic_regression(X_train, y_train, learning_rate=0.1, num_iterations=10):
    num_features = len(X_train[0])
    num_classes = len(set(y_train))
    
    weights = [[0.0] * num_features for _ in range(num_classes)]
    bias = [0.0] * num_classes

    for _ in range(num_iterations):
        for i in range(len(X_train)):
            for class_label in range(num_classes):
                z = bias[class_label]
                for j in range(num_features):
                    z += weights[class_label][j] * X_train[i][j]

                y_pred = sigmoid(z)
                error = y_pred - int(y_train[i] == class_label)

                bias[class_label] -= learning_rate * error
                for j in range(num_features):
                    weights[class_label][j] -= learning_rate * error * X_train[i][j]
    return weights, bias

# Model Evaluation
def evaluate_model(X_test, y_test, weights, bias):
    num_correct = 0
    for i in range(len(X_test)):
        class_scores = []
        for class_label in range(len(weights)):
            z = bias[class_label]
            for j in range(len(weights[class_label])):
                z += weights[class_label][j] * X_test[i][j]

            class_scores.append(sigmoid(z))

        predicted_class = class_scores.index(max(class_scores))
        if predicted_class == y_test[i]:
            num_correct += 1

    accuracy = num_correct / len(X_test)
    return accuracy

if __name__=='__main__':
    feature_vectors = vectorize(matrix_data, class_labels)
    X_train, y_train, X_test, y_test = split_data(feature_vectors, class_labels)
    weights, bias = train_logistic_regression(X_train, y_train)
    print('Accuracy:', evaluate_model(X_test, y_test, weights, bias))