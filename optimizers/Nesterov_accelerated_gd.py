# nesterov accelarated gradient descent
# w_look_ahead = w - update_t-1
# update_t = update_t-1 * gamma + derv(w_look_ahead) * learning rate
# w = w - update_t
# same for the bias also
import numpy as np
import matplotlib.pyplot as plt

def nesterov_accelarated_gradient_descent(X, y) :
    rows, columns = X.shape
    learning_rate = 0.01
    weights = np.zeros(columns)
    bias = 0
    epochs = 300
    cost = [ ]
    itera = [ ]
    prev_update_weight = [0]
    prev_update_bias = 0
    gamma = 0.001
    
    for _ in range(epochs) :
        
       
        w_look_ahead = weights - sum(prev_update_weight)
        b_look_ahead = bias - prev_update_bias
        
        y_p = np.dot(X ,w_look_ahead) + b_look_ahead
        
        dw = -2/rows * (np.dot(X.T, (y - y_p)))
        db = -2/rows *  sum((y- y_p))

        update_w = prev_update_weight + learning_rate * dw 
        update_b = prev_update_bias + learning_rate * db

        prev_update_weights = update_w
        prev_update_bias = update_b
        
        weights = weights - update_w
        bias = bias - update_b
        
        cost1 = np.mean(np.square(y - y_p))
        
        if _ % 10 == 0 :            
            cost.append(cost1)
            itera.append(_)
    return weights, bias, itera, cost
