class LDA :
    
    def __init__(self, n_components = 2) :
        self.n_components = n_components
        self.lda = None
    
    def fit(self, X, y) :
        
        overall_mean = np.mean(X)
        class_labels = np.unique(y)
        n_features = np.shape(X)[1]
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for cls in class_labels :
            
            rows_cls = X[y == cls] 
            mean_cls = np.mean(rows_cls, axis = 0)
            S_W += (rows_cls - mean_cls).T.dot(rows_cls - mean_cls)
        
            ni = np.shape(rows_cls)[0]
            diff = (mean_cls - overall_mean).reshape(n_features, 1)
            print(diff.shape)
            S_B += ni*(diff).dot(diff.T)
            
        lda_matrix = S_W.dot(np.linalg.inv(S_B))   
        eigenvalues, eigenvectors = np.linalg.eig(lda_matrix)
        desc_indexes = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[desc_indexes]
        eigenvectors = eigenvectors[desc_indexes]
        
       # print(eigenvectors)
        
        self.lda = eigenvectors[:self.n_components]
        #print(self.lda)
        #return self.lda
    
    def transform(self) :
        return np.dot(X, self.lda.T)
