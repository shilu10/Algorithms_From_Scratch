class PCA1 :
    def __init__(self, n_components = 2) :
        self.n_components = n_components
        self.PC = None
        
    def fit(self, X) :
        # Mean centering
        X_Centred = X - np.mean(X , axis = 0)
        print(X_Centred)
        #constructing the covariance matrix
        A = np.cov(X_Centred,  rowvar = False)
        
        # finding a eigenvalues and eigenvectors for the Matrix A 
        eigenvalues, eigenvectors = np.linalg.eig(A)      
        desc_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[desc_indices]
        eigenvectors = eigenvectors[desc_indices]
        print(eigenvalues)
        print(eigenvectors)
        #selecting the pc's
        self.PC = eigenvectors[:self.n_components]
        
    def transform(self, X) :
        Centered_mean = X - np.mean(X, axis =0)
        return(np.dot(Centered_mean, self.PC.T))
