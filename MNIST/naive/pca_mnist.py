import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from sklearn.metrics import accuracy_score


def pca(X, n_components):
    X = X.astype(np.float64)
    X -= np.mean(X, axis=0)
    cov = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    indices = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, indices[:n_components]]
    X_transformed = np.dot(X, components)
    return X_transformed


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_raw = x_train.reshape(x_train.shape[0], -1)
x_test_raw = x_test.reshape(x_test.shape[0], -1)

n_components = 200
x_train_pca = pca(x_train_raw, n_components)
x_test_pca = pca(x_test_raw, n_components)

laplace_values = [0.1, 0.5, 1.0, 10, 100]

accuracies = []

for laplace in laplace_values:
    class_counts = np.bincount(y_train)
    class_probs = class_counts / len(y_train)
    conditional_probs = np.zeros((x_test_pca.shape[0], 10))

    for i in range(10):
        class_indices = np.where(y_train == i)[0]
        class_data = x_train_pca[class_indices]
        class_mean = np.mean(class_data, axis=0)
        class_cov = np.cov(class_data.T)
        inv_class_cov = np.linalg.inv(class_cov + laplace * np.eye(n_components))
        log_likelihood = -0.5 * np.sum(
            np.dot(x_test_pca - class_mean, inv_class_cov) * (x_test_pca - class_mean),
            axis=1
        )
        prior = np.log(class_probs[i])
        conditional_probs[:, i] = log_likelihood + prior

    y_pred = np.argmax(conditional_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

x_ticks = np.arange(len(laplace_values))
plt.figure(figsize=(8, 6))
plt.bar(x_ticks, accuracies)
plt.xlabel("Laplace Smoothing")
plt.ylabel("Accuracy")
plt.title("Accuracy for Different Laplace Smoothing Values (PCA Features)")
plt.xticks(x_ticks, laplace_values)
plt.tight_layout()
plt.show()
