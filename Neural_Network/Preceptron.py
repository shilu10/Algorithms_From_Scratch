class Perceptron :
    def __init__(self, eta = 0.1, epos = 1000) :
        self.eta = eta 
        self.epos = epos
        self.activation_function = self._unit_step_function
        self.weights = None
        self.bias = None
        self.activation_function_output = self._unit_step_function1
    
    def _unit_step_function1(self, value) :
        return np.where(value < 0, 0, 1)
        
    def _unit_step_function(self, value) :
        return np.where(value <= 0, -1, 1)
    
    def fit(self, X, y) :
        y_new = [1 if record == 1 else -1 for record in y]
        print(y_new, "y_new")
        no_row_vectors, no_column_vectors = X.shape
        self.weights = np.zeros(no_column_vectors)
        self.bias = 0
        
        for _ in range(self.epos) :
            for index, row_vector in enumerate(X) :
                value = np.dot(row_vector, self.weights) + self.bias
                y_prediction = self.activation_function(value)
                
                # updating the weights parellely
                updated_weight = (self.eta * (y_new[index] - y_prediction))
                self.bias = self.eta * (y_new[index] - y_prediction)
                self.weights += updated_weight * row_vector
                
        
    
    def predict(self, X) :
        print(self.weights, self.bias)
        value = np.dot(X, self.weights) + self.bias
        output = self.activation_function_output(value)
        return output
        
