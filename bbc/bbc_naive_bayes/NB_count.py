import math
import matplotlib.pyplot as plt

from load_data import load_data
from split_dataset import split_dataset


class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = None
        self.class_likelihoods = None

    def train(self, X_train, y_train, laplace=1.0):
        # Calculate class priors
        self.class_priors = {}
        total_samples = len(y_train)
        for label in y_train:
            self.class_priors[label] = self.class_priors.get(label, 0) + 1

        # Add Laplace smoothing variable
        for label in self.class_priors:
            self.class_priors[label] = (
                self.class_priors[label] + laplace) / (total_samples + laplace * len(self.class_priors))

        # Calculate class likelihoods
        self.class_likelihoods = {}
        num_features = len(X_train[0])
        for label in self.class_priors:
            class_samples = [X_train[i]
                             for i in range(total_samples) if y_train[i] == label]
            class_feature_counts = [0] * num_features
            total_class_samples = 0
            for sample in class_samples:
                for i, count in enumerate(sample):
                    class_feature_counts[i] += count
                    total_class_samples += count
            self.class_likelihoods[label] = [
                (count + laplace) / (total_class_samples + laplace * num_features) for count in class_feature_counts]

    def predict(self, X_test):
        y_pred = []
        for sample in X_test:
            probabilities = {}
            for label in self.class_priors:
                probability = math.log(self.class_priors[label])
                for i, count in enumerate(sample):
                    if count > 0:
                        probability += math.log(
                            self.class_likelihoods[label][i]) * count
                probabilities[label] = probability
            predicted_label = max(probabilities, key=probabilities.get)
            y_pred.append(predicted_label)
        return y_pred


if __name__ == "__main__":
    vocab, articles, labels, keys = load_data()
    X_train, X_test, Y_train, y_test = split_dataset(
        articles, labels, test_size=0.2)

    laplace_values = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    accuracy_values = []

    # To collect values to plot
    for alpha in laplace_values:
        fold_accuracies = []
        num_folds = 5
        fold_size = len(X_train) // num_folds

        for fold in range(num_folds):
            start = fold * fold_size
            end = start + fold_size

            fold_X_train = X_train[:start] + X_train[end:]
            fold_y_train = Y_train[:start] + Y_train[end:]
            fold_X_val = X_train[start:end]
            fold_y_val = Y_train[start:end]

            nb_classifier = NaiveBayesClassifier()

            nb_classifier.train(fold_X_train, fold_y_train, laplace=alpha)

            fold_y_pred = nb_classifier.predict(fold_X_val)

            fold_accuracy = sum(1 for pred, true in zip(
                fold_y_pred, fold_y_val) if pred == true) / len(fold_y_val)
            fold_accuracies.append(fold_accuracy)

        average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        accuracy_values.append(average_accuracy)

    best_laplace = laplace_values[accuracy_values.index(max(accuracy_values))]
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(X_train, Y_train, laplace=best_laplace)

    y_pred = nb_classifier.predict(X_test)

    accuracy = sum(1 for pred, true in zip(y_pred, y_test)
                   if pred == true) / len(y_test)

    print("Best Laplace value:", best_laplace)
    print("Test Accuracy:", accuracy)

    plt.plot(laplace_values, accuracy_values, marker='o')
    plt.xlabel('Laplace Value')
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes Classifier with Word Count')
    plt.grid(True)
    plt.show()
