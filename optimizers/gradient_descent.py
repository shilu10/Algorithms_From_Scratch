import numpy as np
import matplotlib.pyplot as plt
def gradient_descent(X, y) :
    rows, columns = X.shape
    learning_rate = 0.01
    weights = np.zeros(columns)
    bias = 0
    epochs = 1000
    cost = [ ]
    itera = [ ]
    for _ in range(epochs) :
        y_p = np.dot(X ,weights) + bias
        cost1 = np.mean(np.square(y - y_p))
       
        dw = -2/rows * (np.dot(X.T, (y - y_p)))
        db = -2/rows *  sum((y- y_p))
        
        weights = weights - dw * learning_rate
        bias = bias - db * learning_rate
        
        if _ % 20 == 0 :     
            cost.append(cost1)
            itera.append(_)
    return weights, bias, itera, cost

def plotting_gradient(X, y) :
    weight, bias, itera, cost = gradient_descent(X, y)
    print(f"Weight : {weight}")
    print(f"bias : {bias}")
    plt.plot(itera, cost)

