class Adaline :
    def __init__(self, epos = 50, eta = 0.1) :
        self.epos = epos
        self.eta = eta
        self.activation_function = self._linear_activation_function
        self.weights = None
        self.bias = None
        
    
    def _linear_activation_function(self, X) :
        return self.net_input_function(X)
    
    def net_input_function(self, X) :
        return np.dot(self.weights, X.T) + self.bias
    
    def fit(self, X, y) :
        rows, columns = X.shape
        self.weights = np.zeros(columns)
        self.bias = 0
        
        for _ in range(self.epos) :
            y_hat = self.net_input_function(X)
            errors = y - y_hat
            cost =  sum((errors) ** 2)/2 
            dw = self.eta * X.T.dot(errors)
            db = self.eta * sum(errors)
            self.weights += dw
            self.bias += db
            
    def predict(self, X) :
        return  np.where(self.activation_function(X) >= 0, 1, 0)
