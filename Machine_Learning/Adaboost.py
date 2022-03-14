from copy import deepcopy
import numpy as np
import pandas as pd


class Adaboost :
    AOS = []
    CLFS = []
    
    def __init__(self) : 
        pass
    
    def sample_weight(self, dataframe) :
        self.weight = 1 / len(dataframe)
        return self.weight
    
    def error(self, points) :
        self.errorr = np.sum(points)
        return self.errorr
    
    def amount_of_say(self, error) :
        self.aos = 1/2*(np.log((1-error)/error))
        return self.aos
    
    def error_new_weight(self, aos, point) :
        self.new_weight =np.array(point) * (2.718281828459045 ** aos)
        return self.new_weight
    
    def other_new_weight(self, aos, point) :
        self.new_weight =np.array(point) * (2.718281828459045 ** -aos)
        return self.new_weight
    
    def normalized_weight(self, error_list, other_list) :
        err = np.sum(error_list)
        othr = np.sum(other_list)
        
        deno = err + othr
        errorweightlist = error_list/deno
        otherweightlist = other_list/deno
        return errorweightlist, otherweightlist
    
    def fit(self, test1, test2) :
        i = 0
        weight = self.sample_weight(test1)
        
        dic = {}
        clfs = [ ]
        x = pd.DataFrame(test1)
        y = deepcopy(test2)
        
        x['weight'] = weight
        self.new = pd.DataFrame(y)
        self.df = pd.concat([x, self.new], axis = 'columns')
        
        print(x)
        print(y)
        from sklearn.tree import DecisionTreeClassifier
        for _ in range(1000) :
           
            if _ == 0:
                x['weight'] = weight
                
            classifer = DecisionTreeClassifier(max_depth = 1)
            classifer.fit(x, y, sample_weight=x['weight'])
            plot_tree(classifer, filled = True)
            prediction = classifer.predict(x)
            wrong_prediction = np.sum(y != prediction)
           
            point = (np.array(self.df[y != prediction]['weight']))
           
            if len(point) <= 0:
                break
                
            other_point = (np.array(self.df[y == prediction]['weight']))
            error = self.error(point)
           # print(error)
            aos = self.amount_of_say(error)
            Adaboost.AOS.append(aos)
            Adaboost.CLFS.append(classifer)

            dic[classifer] = aos

            errow = self.error_new_weight(aos, point)
           # print(errow)
            otherw = self.other_new_weight(aos, other_point)
            #print(otherw)
            no_of_other_points = len(x)-wrong_prediction
            new_error_weight, new_other_weight = self.normalized_weight(errow, otherw)

           # updated the weight successfully in the df we have created 
            self.df.loc[self.df[y.name] != prediction, 'weight'] = new_error_weight
            self.df.loc[self.df[y.name] == prediction, 'weight'] = new_other_weight


            x['weight'] = self.df['weight']
        del x 
        del y
        del test1
        del test2
            
    def prediction(self, test) :
        x1 = pd.DataFrame(test)
        x1['weight'] = 1/len(x1)
        print(x1)
        
        print(x1)
        
        from collections import Counter, defaultdict
        predc = [ ]
        
        for i in Adaboost.CLFS : 
            pred = i.predict(x1)
            predc.append(pred)
        np.array(predc)
        
        answer = defaultdict(list)
        for j in predc : 
            #print(j , "pred")
            for i in range(len(j)) : 
                answer[i].append(j[i]) 
        final_answer = [ ]        
        for i in answer.values() :
            #print(i, "firstnas")
            dic = defaultdict(list)
            for j in  range(len(i)) :
                dic[i[j]].append(Adaboost.AOS[j])
                
            print(dic)
            
            print()
            label = self.new[self.new.columns[len(self.new.columns)-1]]
            if sum(dic[label.unique()[0]]) > sum(dic[label.unique()[1]]) :
                final_answer.append(label.unique()[0])
            else:
                final_answer.append(label.unique()[1])      
        del x1 
        del test
        return final_answer
    
        
        
            
           
                
                
