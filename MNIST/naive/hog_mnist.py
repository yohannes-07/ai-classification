import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np

def calculate_gradients(image):
    dx = np.gradient(image, axis=0)
    dy = np.gradient(image, axis=1)
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    return magnitude, angle

def calculate_histogram(gradient_magnitude, gradient_angle):
    histogram = np.zeros(9)
    angle_bins = np.linspace(-np.pi, np.pi, num=10)
    
    for i in range(gradient_magnitude.shape[0]):
        for j in range(gradient_magnitude.shape[1]):
            magnitude = gradient_magnitude[i, j]
            angle = gradient_angle[i, j]
            for k in range(9):
                if angle >= angle_bins[k] and angle < angle_bins[k+1]:
                    histogram[k] += magnitude
                    break
    
    return histogram

def calculate_hog_features(images):
    hog_features = []
    for image in images:
        gradient_magnitude, gradient_angle = calculate_gradients(image)
        histogram = calculate_histogram(gradient_magnitude, gradient_angle)
        hog_features.append(histogram)
    return np.array(hog_features)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_hog = calculate_hog_features(x_train)
x_test_hog = calculate_hog_features(x_test)

laplace_values = [0.1, 0.5, 1.0, 10, 100]

accuracies = []

for laplace in laplace_values:
    class_counts = np.bincount(y_train)
    class_probs = class_counts / len(y_train)
    conditional_probs = np.zeros((10, 9))

    for i in range(10):
        class_indices = np.where(y_train == i)[0]
        class_histograms = x_train_hog[class_indices]
        class_histogram_sum = np.sum(class_histograms, axis=0)
        conditional_probs[i] = (class_histogram_sum + laplace) / (np.sum(class_histogram_sum) + 9 * laplace)

    scores = np.dot(x_test_hog, np.log(conditional_probs.T)) + np.log(class_probs)
    y_pred = np.argmax(scores, axis=1)

    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

x_ticks = np.arange(len(laplace_values))
plt.figure(figsize=(8, 6))
plt.bar(x_ticks, accuracies)
plt.xlabel("Laplace Smoothing")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Laplace Smoothing Values (HOG Features)")
plt.xticks(x_ticks, laplace_values)
plt.tight_layout()
plt.show()
