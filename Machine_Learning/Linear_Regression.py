
import numpy as np

class LinearRegression :
    
    
    def gradient_descent(self, learning_rate, x, y ,iteration, slope = 0, constant = 0):
       # print(slope,"slope")
        self.slope = slope
        self.constant = constant
        n = len(x)   
        x = np.array(x)
        y= np.array(y)
        for i in range(iteration) :
            y_prediction = self.slope*x + self.constant
            least_square = 1/n * (sum([difference ** 2 for difference in (y_prediction-y)]))
            slope_derivative = -(2/n)*sum(x*(y-y_prediction))
            constant_derivative = -(2/n)*sum(y-y_prediction)
            self.slope = self.slope - learning_rate * slope_derivative       
            self.constant = self.constant - learning_rate * constant_derivative      
        print(self.slope, self.constant,"s")
        return self.slope, self.constant

    def fit(self, x_train, y_train, iteration = 100000):
            self.slope, self.constant=self.gradient_descent(0.08, x_train, y_train, iteration )

    def predict(self, x_test) :
            x_test = np.array(x_test)
            y_pred = [i for i in self.slope * x_test + self.constant] 
            return y_pred
