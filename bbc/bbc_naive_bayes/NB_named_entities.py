import math

from matplotlib import pyplot as plt

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
        for label in self.class_priors:
            self.class_priors[label] = (self.class_priors[label] + laplace) / (
                total_samples + laplace * len(self.class_priors)
            )

        # Calculate class likelihoods
        self.class_likelihoods = {}
        num_features = len(X_train[0])
        for label in self.class_priors:
            class_samples = [X_train[i]
                             for i in range(total_samples) if y_train[i] == label]
            class_feature_counts = [
                sum(sample[j] for sample in class_samples) + laplace for j in range(num_features)]
            total_class_samples = sum(class_feature_counts)
            self.class_likelihoods[label] = [
                count / total_class_samples for count in class_feature_counts]

    def predict(self, X_test):
        y_pred = []
        for sample in X_test:
            probabilities = {}
            for label in self.class_priors:
                probability = math.log(self.class_priors[label])
                for i, feature in enumerate(sample):
                    if feature > 0:
                        probability += math.log(
                            self.class_likelihoods[label][i])
                probabilities[label] = probability
            predicted_label = max(probabilities, key=probabilities.get)
            y_pred.append(predicted_label)
        return y_pred


def extract_named_entities(text):
    entities = []
    words = text.split()
    current_entity = ""
    current_label = None
    for word in words:
        if word.isupper():
            if current_entity:
                entities.append((current_entity, current_label))
                current_entity = ""
            current_entity = word
            current_label = "ORG"
        elif current_entity:
            current_entity += " " + word
    if current_entity:
        entities.append((current_entity, current_label))
    return entities


if __name__ == "__main__":
    vocab, articles, labels, keys = load_data()
    preprocessed_articles = [
        extract_named_entities(" ".join(map(lambda x: "" if x == 0 else vocab[index], article))) for index, article in enumerate(articles)
    ]

    # Split the preprocessed dataset
    X_train, X_test, y_train, y_test = split_dataset(
        preprocessed_articles, labels, test_size=0.2)

    laplace_list = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    accuracy_list = []

    for laplace in laplace_list:
        fold_accuracies = []
        num_folds = 5
        fold_size = len(X_train) // num_folds

        for fold in range(num_folds):
            start = fold * fold_size
            end = start + fold_size

            fold_X_train = X_train[:start] + X_train[end:]
            fold_y_train = y_train[:start] + y_train[end:]
            fold_X_val = X_train[start:end]
            fold_y_val = y_train[start:end]

            nb_classifier = NaiveBayesClassifier()
            nb_classifier.train(fold_X_train, fold_y_train, laplace=laplace)

            fold_y_pred = nb_classifier.predict(fold_X_val)
            fold_accuracy = sum(1 for pred, true in zip(
                fold_y_pred, fold_y_val) if pred == true) / len(fold_y_val)
            fold_accuracies.append(fold_accuracy)

        average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        accuracy_list.append(average_accuracy)

    best_laplace = laplace_list[accuracy_list.index(max(accuracy_list))]

    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(X_train, y_train, alpha=best_laplace)
    y_pred = nb_classifier.predict(X_test)

    accuracy = sum(1 for pred, true in zip(y_pred, y_test)
                   if pred == true) / len(y_test)

    print("Best Laplace value:", best_laplace)
    print("Accuracy:", accuracy)

    plt.plot(laplace_list, accuracy_list, marker="o")
    plt.xlabel('Laplace Value')
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes Classifier with Named entities')
    plt.grid(True)
    plt.show()
