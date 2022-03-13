import numpy as np
import matplotlib.pyplot as plt

def adagrad(X, y) :
    
    rows, columns = X.shape
    learning_rate = 1
    weights = np.zeros(columns)
    bias = 0
    epochs = 1000
    cost = [ ]
    itera = [ ]
    prev_update_weight = [0]
    prev_update_bias = 0
    c = 0.09
    
    for _ in range(epochs) :
        y_p = np.dot(X ,weights) + bias
        cost1 = np.mean(np.square(y - y_p))
        
        dw = -2/rows * (np.dot(X.T, (y - y_p)))
        db = -2/rows *  sum((y- y_p))

        update_w = prev_update_weight + (dw**2) 
        update_b = prev_update_bias + db ** 2

        prev_update_weights = update_w
        prev_update_bias = update_b
        
        weights = weights - (learning_rate / np.sqrt(c + update_w) ) * dw
        bias = bias - (learning_rate / np.sqrt(c + update_b) )  * db
        
        if _ % 10 == 0 :            
            cost.append(cost1)
            itera.append(_)
    return weights, bias, itera, cost1


def plotting_adagrad(X, y) :
    print(X)
    weight, bias, itera, cost = momentum_based_gradient_descent(X, y)
    print(cost, "cost")
    print(f"Weights : {weight}")
    print(f"bias : {bias}")
    plt.plot(itera, cost)
