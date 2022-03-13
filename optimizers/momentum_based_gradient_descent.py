# momentum based gradient descent

# update t =  gamma * update t -1 + learning rate * derv(w)
# w = w - uodate_t

def momentum_based_gradient_descent(X, y) :
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
        y_p = np.dot(X ,weights) + bias
        cost1 = np.mean(np.square(y - y_p))
        
        dw = -2/rows * (np.dot(X.T, (y - y_p)))
        db = -2/rows *  sum((y- y_p))

        update_w = prev_update_weight + learning_rate * dw 
        update_b = prev_update_bias + learning_rate * db

        prev_update_weights = update_w
        prev_update_bias = update_b
        
        weights = weights - update_w
        bias = bias - update_b
        
        if _ % 10 == 0 :            
            cost.append(cost1)
            itera.append(_)
    return weights, bias, itera, cost


def plotting_momentum_gradient(X, y) :
    weight, bias, itera, cost = momentum_based_gradient_descent(X, y)
    print(f"Weights : {weight}")
    print(f"bias : {bias}")
    plt.plot(itera, cost)
