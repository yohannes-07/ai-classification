import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.datasets import mnist

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, penalty='l2', C=1.0):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.penalty = penalty
        self.C = C

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / len(X)) * (np.dot(X.T, (y_pred - y)) + (self.C / len(X)) * self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * np.mean(y_pred - y)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.round(y_pred)


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


def calculate_hog_features(images, num_bins=9, cell_size=(8, 8), block_size=(2, 2), block_stride=(1, 1)):
    hog_features = []
    for image in images:
        gradient_magnitude, gradient_orientation = calculate_gradient(image)
        num_cells_y = image.shape[0] // cell_size[0]
        num_cells_x = image.shape[1] // cell_size[1]
        hog_feature = []
        for i in range(num_cells_y):
            for j in range(num_cells_x):
                cell_start_y = i * cell_size[0]
                cell_start_x = j * cell_size[1]
                cell_end_y = cell_start_y + cell_size[0]
                cell_end_x = cell_start_x + cell_size[1]
                cell_gradient_magnitude = gradient_magnitude[cell_start_y:cell_end_y, cell_start_x:cell_end_x]
                cell_gradient_orientation = gradient_orientation[cell_start_y:cell_end_y, cell_start_x:cell_end_x]
                histogram = calculate_histogram(cell_gradient_magnitude, cell_gradient_orientation, num_bins)
                hog_feature.extend(histogram)
        hog_feature = normalize_histogram(np.array(hog_feature))
        hog_features.append(hog_feature)
    return np.array(hog_features)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_hog = calculate_hog_features(x_train)
x_test_hog = calculate_hog_features(x_test)

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
laplace_values = [0.1, 0.5, 1.0, 10, 100]

accuracies = []

for lr in learning_rates:
    for laplace in laplace_values:
        model = LogisticRegression(learning_rate=lr, num_iterations=1000, penalty='l2', C=1 / (lr * len(x_train_hog)))
        model.fit(x_train_hog, y_train)
        y_pred = model.predict(x_test_hog)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

x_ticks = np.arange(len(laplace_values))
width = 0.15
plt.figure(figsize=(10, 6))
for i, lr in enumerate(learning_rates):
    plt.bar(x_ticks + (i * width), accuracies[i * len(laplace_values):(i + 1) * len(laplace_values)], width, label=f"LR={lr}")

plt.xlabel("Laplace Smoothing")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Learning Rates and Laplace Smoothing")
plt.xticks(x_ticks + width * (len(learning_rates) / 2), laplace_values)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
