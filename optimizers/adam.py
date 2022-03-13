import numpy as np
import matplotlib.pyplot as plt

def adam(X, y) :
    
    rows, columns = X.shape
    learning_rate = 0.0001
    weights = np.zeros(columns)
    bias = 0
    epochs = 2500
    cost = [ ]
    itera = [ ]
    prev_vw  = [0]
   # prev_ vb = 0
    prev_mw = [0]
  #  prev_mb = 0
    gamma1 = 0.95
    gamma2 = 0.005
    c = 10
    
    for _ in range(epochs) :
        y_p = np.dot(X ,weights) + bias
        cost1 = np.mean(np.square(y - y_p))
        
        dw = -2/rows * (np.dot(X.T, (y - y_p)))
        db = -2/rows *  sum((y- y_p))

        v_w = (prev_vw * gamma1 ) + (1 - gamma1) * (dw ** 2) 
        m_w = (prev_mw * gamma2 ) + (1 - gamma2) * (dw)

        prev_vw = v_w
        prev_mw = m_w
        
        # bias correction
        m_hat = m_w / (1 - gamma2)
        v_hat = v_w / (1 - gamma1)
        
        # weight and bias updation
        weights = weights - (learning_rate / np.sqrt(c + v_hat) ) * m_hat
        bias = bias - learning_rate  * db
        
        if _ % 10 == 0 :            
            cost.append(cost1)
            itera.append(_)
    return weights, bias, itera, cost1


def plotting_adam(X, y) :
    weight, bias, itera, cost = momentum_based_gradient_descent(X, y)
    print(cost, "cost")
    print(f"Weights : {weight}")
    print(f"bias : {bias}")
    plt.plot(itera, cost)
