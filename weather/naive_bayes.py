from math import log

import feature_extraction_naive_test
from feature_extraction_naive import featureize

features, labels, lens = featureize()
features_test, labels_test, lens_test = feature_extraction_naive_test.featureize()

train_features = features
train_labels = labels

test_features = features_test
test_labels = labels_test
correct = 0



def begin():
    label_counts = {}

    for label in train_labels:

        if label not in label_counts:
            label_counts[label] = 0

        label_counts[label] += 1


    feature_counts = {}


    for i, features in enumerate(train_features):
        label = train_labels[i]

        if label not in feature_counts:
            feature_counts[label] = {}


        for j, value in enumerate(features):

            if j not in feature_counts[label]:
                feature_counts[label][j] = {}

            if value not in feature_counts[label][j]:
                feature_counts[label][j][value] = 0

            feature_counts[label][j][value] += 1

    return label_counts, feature_counts



def train(test_features, label_counts, train_labels, lap1 = 1, lap2 = 3):
    global correct

    for i, features in enumerate(test_features):
        
        posteriors = {}
        for label in label_counts:
            
            prior = log(label_counts[label] / len(train_labels))
        
            likelihood = 0
            for j, value in enumerate(features):
                count = feature_counts[label][j].get(value, 0)
                likelihood += log((count + lap1) / (label_counts[label] + lap2))
        
            posteriors[label] = prior + likelihood
    
        prediction = max(posteriors, key=posteriors.get)

        if prediction == test_labels[i]:
            correct += 1

    return correct

laplace = [ 0.1, 0.5, 1.0, 10, 100, 99]
for lap in range(1, len(laplace)):

    label_counts, feature_counts = begin()
    correct = 0
    correct = train(test_features, label_counts, train_labels, laplace[lap - 1], laplace[lap])
    accuracy = correct / len(test_labels)
    print("Naive Bayes accuracy:", accuracy)