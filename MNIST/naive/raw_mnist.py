import time
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np

title = "Laplace Smoothing \n using RAW"
y_lable = "Accuracy"

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_raw = x_train.reshape(x_train.shape[0], -1)
x_test_raw = x_test.reshape(x_test.shape[0], -1)

laplace_values = [0.1, 0.5, 1.0, 10, 100]

accuracies = []

start_time = time.time()

for laplace in laplace_values:
    count = np.zeros(10)
    correct = np.zeros(10)

    for i, x in enumerate(x_train_raw):
        label = y_train[i]
        count[label] += 1

        correct[label] += np.sum(x[x > 0] == label)

    predictions = []

    for x in x_test_raw:
        probabilities = []

        for label in range(10):
            prob = correct[label] / count[label]

            prob *= np.prod(correct[label] / count[label] * (x > 0))

            probabilities.append(prob)

        predicted_label = np.argmax(probabilities)
        predictions.append(predicted_label)

    correct_count = np.sum(np.array(y_test) == np.array(predictions))
    accuracy = correct_count / len(y_test)
    accuracies.append(accuracy )

end_time = time.time()
total_time = end_time - start_time

x_ticks = range(len(laplace_values))
plt.figure(figsize=(8, 6))
plt.bar(x_ticks, accuracies)
plt.xlabel(title)
plt.ylabel(y_lable)
plt.title("Accuracy for Different Laplace Smoothing Values\nTotal Running Time: {:.2f} seconds".format(total_time))
plt.xticks(x_ticks, laplace_values)
plt.tight_layout()
plt.show()
