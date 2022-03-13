import pandas as pd
import numpy as np
def mini_batch_gradient_descent(X, y) :
    rows, columns = X.shape
    weights = np.zeros(columns)
    bias = 0
    learning_rate = 0.01
    epochs = 1000
    batch_size = 5
    cost = []
    itera = []
    
    for _ in range(epochs) :
        no_of_records_in_batch = rows // 5
        x = pd.DataFrame(X).sample(no_of_records_in_batch)
        idx = x.index
        yt = y[idx]
        x = np.array(x)
        y_p = np.dot(x, weights) + bias 
        
        dw = -2/rows * (np.dot(x.T, (yt - y_p)))
        db = -2/rows * sum(yt- y_p)
        
        cost1 = np.mean(np.square(yt-y_p))
        if _ % 30 == 0 :
            cost.append(cost1)
            itera.append(_)
        
        weights = weights - dw * learning_rate
        bias = bias - db * learning_rate
    return weights, bias, itera, cost

def plotting_minibatch_gradient(X, y) :
    weight, bias, itera, cost = mini_batch_gradient_descent(X, y)
    print(f"Weight : {weight}")
    print(f"bias : {bias}")
    plt.plot(itera, cost)
