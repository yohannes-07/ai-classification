import numpy as np
import matplotlib.pyplot as plt
from utils import accuracy_score
from keras.datasets import mnist


class LogisticRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000, regularization=None, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_param = lambda_param

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.weights = np.zeros((self.num_classes, X.shape[1]))
        self.bias = np.zeros((self.num_classes,))

        for i in range(self.num_iterations):
            scores = self._softmax(np.dot(X, self.weights.T) + self.bias)

            if self.regularization == 'l2':
                regularization_term = (self.lambda_param / X.shape[0]) * self.weights
                gradient = (1 / X.shape[0]) * np.dot((scores - self._one_hot_encode(y)).T, X) + regularization_term
            else:
                gradient = (1 / X.shape[0]) * np.dot((scores - self._one_hot_encode(y)).T, X)

            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * np.mean(scores - self._one_hot_encode(y), axis=0)

    def predict(self, X):
        scores = np.dot(X, self.weights.T) + self.bias
        return np.argmax(scores, axis=1)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot_encode(self, y):
        one_hot = np.zeros((y.shape[0], self.num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot


def calculate_gradient(image):
    dx = np.gradient(image, axis=1)
    dy = np.gradient(image, axis=0)

    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    gradient_orientation = np.arctan2(dy, dx) * 180 / np.pi

    return gradient_magnitude, gradient_orientation


def calculate_histogram(gradient_magnitude, gradient_orientation, num_bins):
    bins = np.linspace(0, 180, num_bins + 1)

    histogram = np.zeros(num_bins)
    for i in range(gradient_magnitude.shape[0]):
        for j in range(gradient_magnitude.shape[1]):
            orientation = gradient_orientation[i, j]
            magnitude = gradient_magnitude[i, j]

            bin_index = int(np.digitize(orientation, bins)) - 1
            bin_index = bin_index % num_bins

            histogram[bin_index] += magnitude

    return histogram


def normalize_histogram(histogram):
    norm = np.linalg.norm(histogram)
    if norm != 0:
        histogram /= norm
    return histogram


def calculate_pca_features(images, num_bins=9, cell_size=(8, 8), block_size=(2, 2), block_stride=(1, 1)):
    pca_features = []
    for image in images:
        gradient_magnitude, gradient_orientation = calculate_gradient(image)

        num_cells_y = image.shape[0] // cell_size[0]
        num_cells_x = image.shape[1] // cell_size[1]

        pca_feature = []

        for i in range(num_cells_y):
            for j in range(num_cells_x):
                cell_start_y = i * cell_size[0]
                cell_start_x = j * cell_size[1]
                cell_end_y = cell_start_y + cell_size[0]
                cell_end_x = cell_start_x + cell_size[1]

                cell_gradient_magnitude = gradient_magnitude[cell_start_y:cell_end_y, cell_start_x:cell_end_x]
                cell_gradient_orientation = gradient_orientation[cell_start_y:cell_end_y, cell_start_x:cell_end_x]

                histogram = calculate_histogram(cell_gradient_magnitude, cell_gradient_orientation, num_bins)

                pca_feature.extend(histogram)

        pca_feature = normalize_histogram(np.array(pca_feature))

        pca_features.append(pca_feature)

    return np.array(pca_features)



(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train_pca = calculate_pca_features(x_train)


x_test_pca = calculate_pca_features(x_test)


learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
regularizations = [None, 'l2']


accuracies = []


for lr in learning_rates:
    for regularization in regularizations:
        
        model = LogisticRegression(learning_rate=lr, regularization=regularization, num_iterations=1000)
        model.fit(x_train_pca, y_train)

        y_pred = model.predict(x_test_pca)


        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)


x_ticks = np.arange(len(regularizations))
width = 0.15
plt.figure(figsize=(10, 6))
for i, lr in enumerate(learning_rates):
    plt.bar(x_ticks + (i * width), accuracies[i * len(regularizations):(i + 1) * len(regularizations)],
            width, label=f"LR={lr}")

plt.xlabel("Regularization")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Learning Rates and Regularization")
plt.xticks(x_ticks + width * (len(learning_rates) / 2), regularizations)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
