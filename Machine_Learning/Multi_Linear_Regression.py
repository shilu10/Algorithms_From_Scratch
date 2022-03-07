import numpy as np

class LinearRegression :
    
    def __init__(self, learning_rate = 0.01, iteration = 100) :
        self.learning_rate = learning_rate
        self.slope = None
        self.constant = None
        self.iteration = iteration
        
    def gradient_descent(self, x, y ):
        no_of_row_vectors, no_of_column_vectors = x.shape
        self.slope = np.zeros(no_of_column_vectors)
        self.constant = 0 
        
        x = np.array(x)
        y= np.array(y)
        
        for i in range(self.iteration) :
            y_prediction = np.dot(x, self.slope) + self.constant
        # least square function
            least_square = 1/no_of_row_vectors * (sum([difference ** 2 for difference in (y_prediction-y)]))
        # derivatives of slope and constant respective to least square
            slope_derivative = -(2/no_of_row_vectors) * sum(np.dot(x.T, (y-y_prediction)))
            constant_derivative = -(2/no_of_row_vectors)*sum(y-y_prediction)
       # updating the values of constant and the slope     
            self.slope = self.slope - self.learning_rate * slope_derivative          
            self.constant = self.constant - self.learning_rate * constant_derivative      
      
        return self.slope, self.constant

    def fit(self, x_train, y_train):
            self.slope, self.constant=self.gradient_descent( x_train, y_train)

    def predict(self, x_test) :
            x_test = np.array(x_test)
            y_pred =np.dot(x_test, self.slope) + self.constant
            return y_pred
    
# testing  
linear_regression = LinearRegression()
linear_regression.fit(X, y)
prediction = linear_regression.predict(X)
