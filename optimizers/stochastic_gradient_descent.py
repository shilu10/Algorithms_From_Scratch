import random
import numpy as np
import matplotlib.pyplot as plt

def stochastic_gradient_descent(X, y) :
    rows, columns = X.shape
    learning_rate = 0.001
    weights = np.zeros(columns)
    bias = 0
    epochs = 1000
    cost = [ ]
    itera = [ ]
    
    for _ in range(epochs) :
        x = random.choice(X)
        idx = np.where(X == x)[0][0]     
        y_p =np.dot(x, weights) + bias
        yt = y[idx]
        cost1 = np.square(yt - y_p)         
        dw = -2/rows * np.dot(x, (yt - y_p))
        db = -2/rows * (yt- y_p)
        
        weights = weights - dw * learning_rate
        bias = bias - db * learning_rate
        
        if _ % 30 == 0 :        
            cost.append(cost1)
            itera.append(_)

    return weights, bias, itera, cost

def plotting_stochastic_gradient(X, y) :
    weight, bias, itera, cost = stochastic_gradient_descent(X, y)
    print(f"Weight : {weight}")
    print(f"bias : {bias}")
    plt.plot(itera, cost)
