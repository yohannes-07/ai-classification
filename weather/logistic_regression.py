from math import exp, log

import feature_extraction
import feature_extraction_logic_test

X, y = feature_extraction.featureize()
X_test, y_test =  feature_extraction_logic_test.featureize()


def softmax(z):

    exp_z = [exp(x) for x in z]
    sum_exp_z = sum(exp_z)
    softmax = [x / sum_exp_z for x in exp_z]
    # print(len(softmax), "here")
    return softmax
    
    

def loss(y, y_hat):
    loss_values = []
    for y_val, y_hat_val in zip(y, y_hat): 
        loss_values.append(y_val * log(y_hat_val) - (1 - y_val) * log(1 - y_hat_val))

    loss_mean = - (sum(loss_values) / len(y))

    return loss_mean

    
def gradients(X, y, y_hat):
    
    dataset_num = len(X)

    
    feature_num = len(X[0])

   
    dw = [[],[],[]]
    db = []
    for k in range(3):
        for j in range(feature_num):
            dwj = (1 / dataset_num) * sum((y_hat[i][k] - y[i][k]) * X[i][j] for i in range(dataset_num))
            dw[k].append(dwj)
        db.append((1 / dataset_num) * sum(y_hat[i][k] - y[i][k] for i in range(dataset_num)))

   
    

    return dw, db

def transpose(matrix):
    res = []
    
    for i in range(len(matrix[0])):
        temp = []
        for j in range(len(matrix)):
            temp.append(matrix[j][i])

        res.append(temp)
    return res




def normalize (X):
    dataset_num, features_num = len(X), len(X[0])

    for j in range(features_num):
        feature_values = [X[i][j] for i in range(dataset_num)]

        mean = sum(feature_values) / dataset_num
        std = (sum((value - mean) ** 2 for value in feature_values) / dataset_num) ** 0.5

        for i in range(dataset_num):
            X[i][j] = (X[i][j] - mean) / (std + 0.1)

    return X


def dot(W, X):
    return sum([W[i]*X[i] for i in range(len(X))])


def train(X, y, bs, iteration, learning_rate):
    dataset_num, feature_num = len(X), len(X[0])  

    weights =[ [0] * feature_num for i in range(len(y[0]))]
    bias = [0,0,0]


    X = normalize(X)

    
    losses = []

   
    for _ in range(iteration):
        for i in range((dataset_num - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]

            
            sm_ = 0
            res = []

            
            for i in range(len(xb)):
                xi  = xb[i]
                sum_ = 0
                
               
                temp =  []
                for k, W in enumerate(weights):
                    temp.append(dot(W, xi) + bias[k])
                res.append(temp)


            y_hat = list(map(softmax, res))

            
            dw, db = gradients(xb, yb, y_hat)
           

            
            for k in range(3):
                weights[k] = [wi - learning_rate * dwi for wi, dwi in zip(weights[k], dw[k])]
                bias[k] -= learning_rate * db[k]

        
    return weights, bias, losses






def predict (X, weights, bias):
    x = normalize(X)


    pred = []
    
    for x in X:
        temp = []
        for k in range(3):
            temp.append(dot(x, weights[k]) + bias[k])
        pred.append(softmax(temp))

    pred_class = []


    for p in pred:
        max_ = max(p)
        s = []
        for i in p:
            if i == max_:
                s.append(1)
            else:
                s.append(0)
        pred_class.append(s)

    

    return pred_class



lr_ = [1.5, 1.0, 0.0001, 0.001, 0.01, 0.1]

weights, bias, l = train(X, y, 50, 10000, 0.01)


def accuracy(y, y_hat):
   

    correct_count = 0

    for i in range(len(y)):
        if y[i] == y_hat[i]:
            correct_count+=1

    
    accuracy = correct_count / len(y)
    return accuracy
# print(y[1], predict(X, weights, bias)[1])
# print(accuracy(y, y_hat=predict(X, weights, bias)))

for i in lr_:
    weights, bias, l = train(X, y, 50, 10000, i)

    print(accuracy(y, y_hat=predict(X, weights, bias)))


for i in lr_:
    weights, bias, l = train(X_test, y_test, 50, 10000, i)
    print("Logistic_Reg_Accuracy :", accuracy(y, y_hat=predict(X, weights, bias)))


