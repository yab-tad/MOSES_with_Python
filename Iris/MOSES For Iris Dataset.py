import os
import pandas as pd
import numpy as np
import subprocess
import random
from pathlib import Path
from sklearn.metrics import confusion_matrix
from time import time
from datetime import datetime



class Iris_Dataset:
    
    def __init__(self, dataset_path):
        
        self.dataset_path = dataset_path
        self.flowers = pd.read_csv(self.dataset_path, delimiter='\t')
        self.flowers = pd.DataFrame(self.flowers)
        
    
    
# Parameter values for bin function    
#
#     -> data: self.flowers.SL, self.flowers.SW, self.flowers.PL, and self.flowers.PW
#
#     -> col: 'SL', 'SW', 'PL', and 'PW'

    def bins(self, data, col):
        
        number_of_bins = int((data.max() - data.min()) // data.std()) + 1

        std = data.std() # Standard deviation of a feature column
        col_min = data.min() # Minimum value of a feature column
        col_max = data.max() # Maximum value of a feature column
        mins = []
        mins.append(col_min)
        
        # Storing values of a feature column with a gap of the standard deviation starting from minimum to maximum value
        for k in range(number_of_bins):
            if (mins[k] + std) < col_max:
                mins.append(mins[k] + std)
            elif mins[k] + std >= col_max:
                mins.append(col_max)
                break

        bin_vals = dict()
        
        # Binarizing feature column elements in a OneHotEncoder manner
        for j in range(len(data)):
            for i in range(1, (number_of_bins+1)):
                if data[j] == mins[i-1] and i <= number_of_bins:
                    if f"{col}{i-1}" not in bin_vals:
                        bin_vals[f"{col}{i-1}"]=[0 for k in range(len(data))]
                    bin_vals[f"{col}{i-1}"][j] = 1

                elif data[j] > mins[i-1] and data[j] <= mins[i] and i < number_of_bins:
                    if f"{col}{i}" not in bin_vals:
                        bin_vals[f"{col}{i}"]=[0 for k in range(len(data))]
                    bin_vals[f"{col}{i}"][j]= 1

                elif data[j]== mins[number_of_bins-1] and i == number_of_bins:
                    if f"{col}{i-1}" not in bin_vals:
                        bin_vals[f"{col}{i-1}"]=[0 for k in range(len(data))]
                    bin_vals[f"{col}{i-1}"][j]= 1
                else:
                    pass

        return pd.DataFrame(bin_vals)        
    
    
# Preparing training and testing sets for each flower type
#    -> dataset_type values are training, testing_class and testing_feautures 
#
#    -> class_name values are Iris-setosa, Iris-versicolor and Iris-virginica

    def create_dataset(self, dataset_type, class_name=''):
    
        if dataset_type == 'training':
            training = self.flowers_bin.drop(self.test_index, axis=0).reset_index(drop=True)
            training['CLASS'] = training['CLASS'].apply(lambda x: 1 if x == class_name else 0)
            return training
        elif dataset_type == 'testing':
            testing = self.flowers_bin.loc[self.test_index].reset_index(drop=True)
            testing['CLASS'] = testing['CLASS'].apply(lambda x: 1 if x == class_name else 0)
            return testing
        else:
            return

        
# Saving prepared dataset:
# 
#    -> files values are: iris_setosa_train, iris_versicolor_train, iris_virginica_train
# 
#    -> file_name values are: iris_test_features, iris_setosa_test, iris_versicolor_test, iris_virginica_test

    def saving_file(self, file, file_name):
        if file_name == 'iris_setosa_train' or file_name == 'iris_versicolor_train' or file_name == 'iris_virginica_train':
            file.to_csv(str(Path().absolute())+"/Iris/Iris Flowers Dataset/Training Dataset/"+ file_name + ".csv", index=False)
        elif file_name == 'iris_virginica_test' or file_name == 'iris_setosa_test' or file_name == 'iris_versicolor_test':
            file.to_csv(str(Path().absolute())+"/Iris/Iris Flowers Dataset/Testing Dataset/"+ file_name + ".csv", index=False)
        elif file_name == 'iris_virginica_validation' or file_name == 'iris_setosa_validation' or file_name == 'iris_versicolor_validation':
            file.to_csv(str(Path().absolute())+"/Iris/Iris Flowers Dataset/Validation Dataset/"+ file_name + ".csv", index=False)
        else:
            return
        
    
    def preparing_dataset(self):
        
        # Assembling the binarized components of the dataset and merging them onto one binarized set
        flowers_SL_bin = self.bins(self.flowers.SL, "SL")
        flowers_SW_bin = self.bins(self.flowers.SW, "SW")
        flowers_PL_bin = self.bins(self.flowers.PL, "PL")
        flowers_PW_bin = self.bins(self.flowers.PW, "PW")
        
        self.flowers_bin = self.flowers[['CLASS']].join(flowers_SL_bin).join(flowers_SW_bin).join( flowers_PL_bin).join(flowers_PW_bin)
        
        # Setting the test indices and merging them all as test_index
        iris_setosa_test_idices = [20,34,30,28,32,26,0,5,4,15,24,45,19,33,47,46,13]
        ver = [10,12,27,9,43,19,6,31,3,46,18,21,24,44,11,14,35]
        iris_versicolor_test_idices = [x+50 for x in ver]
        vir = [27,31,4,48,32,13,46,24,45,20,26,44,38,34,49,5,12]
        iris_virginica_test_idices = [x+100 for x in vir]
        
        self.test_index = iris_setosa_test_idices + iris_versicolor_test_idices + iris_virginica_test_idices
        
        # Creating the training set for each flower type and removing the indices
        iris_setosa_train = self.create_dataset('training', 'Iris-setosa')
        iris_versicolor_train = self.create_dataset('training', 'Iris-versicolor')
        iris_virginica_train = self.create_dataset('training', 'Iris-virginica')
                 
        iris_setosa_train = iris_setosa_train.reset_index(drop=True)
        iris_versicolor_train = iris_versicolor_train.reset_index(drop=True)
        iris_virginica_train = iris_virginica_train.reset_index(drop=True)
        
        # Creating test sets for each flower type and removing the indices
        iris_setosa_test = self.create_dataset('testing', 'Iris-setosa')
        iris_versicolor_test = self.create_dataset('testing', 'Iris-versicolor')
        iris_virginica_test = self.create_dataset('testing', 'Iris-virginica')
        
        iris_setosa_test = iris_setosa_test.reset_index(drop=True)
        iris_versicolor_test = iris_versicolor_test.reset_index(drop=True)
        iris_virginica_test = iris_virginica_test.reset_index(drop=True)
        
        # Saving training set to disk
        self.saving_file(iris_setosa_train,'iris_setosa_train')
        self.saving_file(iris_versicolor_train,'iris_versicolor_train')
        self.saving_file(iris_virginica_train,'iris_virginica_train')
                
        # Saving test set to disk
        self.saving_file(iris_setosa_test,'iris_setosa_test')
        self.saving_file(iris_versicolor_test,'iris_versicolor_test')
        self.saving_file(iris_virginica_test,'iris_virginica_test')



class MOSES_Iris:
    
    def __init__(self, training_file_path, testing_file_path, combo_file_path):
        self.training_file_path = training_file_path
        self.testing_file_path = testing_file_path
        self.combo_file_path = combo_file_path
        
    
    def flower_type_identifier(self, path):
        
        if path[-21:-10] == 'iris_setosa' or path[-20:-9] == 'iris_setosa':
            path = 'iris_setosa'
        elif path[-25:-10] == 'iris_versicolor' or path[-24:-9] == 'iris_versicolor':
            path = 'iris_versicolor'
        elif path[-24:-10] == 'iris_virginica' or path[-23:-9] == 'iris_virginica':
            path = 'iris_virginica'
        else:
            return
        return path
    
    
    def train(self, target='CLASS', iter_=200000, n_combo=10):
    
        file_name = self.training_file_path
        start_time = time()

        s = subprocess.check_output(["moses", f"-i{file_name}", f"-u{target}", f"-m{iter_}", "-W1", f"--result-count={n_combo}"], text=True).split('\n')[:-1]

        training_time = f"{round(((time() - start_time) * 1000), 5)} ms"

        combo_codes = ''

        for i in range(len(s)):
            s[i]= s[i][2:].strip(' ')
            if s[i][0].isnumeric():
                s[i]= s[i][1:].strip(' ')
            combo_codes += s[i]+'\n'
        combo_codes = combo_codes[:-1]

        
        file_name = self.flower_type_identifier(file_name)
        
        with open(str(Path().absolute())+f"/Iris/Combo_Programs/{file_name}_combo.txt","w") as file:
            file.write(f"{combo_codes}")
        
        self.combo_file_path = str(Path().absolute())+f"/Iris/Combo_Programs/{file_name}_combo.txt"
        
        return training_time
    
    
    def eval_model(self, target='CLASS', n_combo=10):
    
        eval_data = self.testing_file_path
        combo_path = self.combo_file_path
        
        threshold = (n_combo // 2)
        combo_outputs = pd.DataFrame()
        with open(f'{combo_path}', 'r') as f:
            line = f.readlines()
            sample_combo = line[0]
            for j in range(0, len(line)):
                s = subprocess.check_output(["eval-table", f"-i{eval_data}", f"-u{target}", f"-c{line[j]}"], text=True)

                combo_out = s.split('\n')
                combo_out = combo_out[1:-1]
                combo_outputs['Combo'+str(j+1)] = combo_out

        eval_output = combo_outputs.astype(int).sum(axis=1)
        self.prediction = pd.DataFrame()
        self.prediction['Output'] = (eval_output.apply(lambda x: 1 if x >= threshold else 0)).reset_index(drop=True)

        setosa_prob = self.prediction.loc[0:17].sum(axis=0)[0]
        versicolor_prob = self.prediction.loc[17:34].sum(axis=0)[0]
        virginica_prob = self.prediction.loc[34:].sum(axis=0)[0]

        flower_type = ['Predicted flower type is: Iris Setosa', 'Predicted flower type is: Iris Versicolor', 'Predicted flower type is: Iris Virginica']


        if setosa_prob > versicolor_prob or setosa_prob > virginica_prob:
            predict = "".join(flower_type[0])
        elif versicolor_prob > setosa_prob or versicolor_prob > virginica_prob:
            predict = "".join(flower_type[1])
        elif virginica_prob > setosa_prob or virginica_prob > versicolor_prob:
            predict = "".join(flower_type[2])
        elif setosa_prob == versicolor_prob:
            predict = "".join(random.choices(flower_type[:2]))
        elif setosa_prob == virginica_prob:
            predict = "".join(random.choices(flower_type[::2]))
        elif versicolor_prob == virginica_prob:
            predict = "".join(random.choices(flower_type[1:3]))

        evaluated_flower_type = self.flower_type_identifier(eval_data)
        eval_time_date = datetime.now().strftime('%A %D [ %X ]')
        
        return evaluated_flower_type, predict, eval_time_date, sample_combo, self.scores()
    
    
    def scores(self):
            
        actual = pd.read_csv(self.testing_file_path, delimiter=(',')).CLASS
        TN, FP, FN, TP = confusion_matrix(actual, self.prediction).flatten()
            
        True_Positive_Ratio = f"{TP}/{TP + TN + FP + FN}"
        False_Positive_Ratio = f"{FP}/{TP + TN + FP + FN}"
        True_Negative_Ratio = f"{TN}/{TP + TN + FP + FN}"
        False_Negative_Ratio = f"{FN}/{TP + TN + FP + FN}"
        accuracy = ((TP + TN) / (TP + TN + FP + FN)) * 100
        precision = (TP / (TP + FP)) * 100
        recall = (TP / (TP + FN)) * 100
        F1_Score = 2 * ((precision * recall) / (precision + recall))
            
        Accuracy = f"{accuracy} %"
        Precision = f"{precision} %"
        Recall = f"{recall} %"
        
        return Accuracy, Recall, Precision, True_Positive_Ratio, False_Positive_Ratio, True_Negative_Ratio, False_Negative_Ratio

    
    def generate_output_files(self):
        
        file_path = self.testing_file_path
        flower_type = self.flower_type_identifier(file_path)
        
        training_time = self.train()
        evaluated_flower_type, prediction, eval_time_date, sample_combo, scores = self.eval_model()
        
        with open(f"Iris/Output/{flower_type} Output1.txt", "w") as file:
            file.write(f"Training time that MOSES took : {training_time}"+'\n')
            file.write(f"Date and Time Evaluation Finished : {eval_time_date}"+'\n')
            file.write(f"Evaluated Flower Type is : {evaluated_flower_type}"+'\n')
            file.write(f"Sample COMBO Program for {flower_type} : {sample_combo}"+'\n')
            file.write(f"Accuracy : {scores[0]}"+'\n')
            file.write(f"Recall : {scores[1]}"+'\n')
            file.write(f"Precision : {scores[2]}"+'\n')
            file.write(f"True_Positive_Ratio : {scores[3]}"+'\n')
            file.write(f"False_Positive_Ratio : {scores[4]}"+'\n')
            file.write(f"True_Negative_Ratio : {scores[5]}"+'\n')
            file.write(f"False_Negative_Ratio : {scores[6]}")
            
        with open(f"Iris/Output/Output2.txt", "w") as file:
            file.write(f"{prediction}"+'\n')
            file.write(f"Accuracy : {scores[0]}"+'\n')
            file.write(f"Date and Time Evaluation Finished : {eval_time_date}")



if __name__ == "__main__":

    dataset_path = str(Path().absolute())+"/Iris/Iris_dataset.txt"

    training_path_setosa = str(Path().absolute())+"/Iris/Iris Flowers Dataset/Training Dataset/iris_setosa_train.csv"
    training_path_versicolor = str(Path().absolute())+"/Iris/Iris Flowers Dataset/Training Dataset/iris_versicolor_train.csv"
    training_path_virginica = str(Path().absolute())+"/Iris/Iris Flowers Dataset/Training Dataset/iris_virginica_train.csv"
    
    testing_path_setosa = str(Path().absolute())+"/Iris/Iris Flowers Dataset/Testing Dataset/iris_setosa_test.csv"
    testing_path_versicolor = str(Path().absolute())+"/Iris/Iris Flowers Dataset/Testing Dataset/iris_versicolor_test.csv"
    testing_path_virginica = str(Path().absolute())+"/Iris/Iris Flowers Dataset/Testing Dataset/iris_virginica_test.csv"

    if os.path.isfile(training_path_setosa) and os.path.isfile(training_path_versicolor) and os.path.isfile(training_path_virginica) and os.path.isfile(testing_path_setosa) and os.path.isfile(testing_path_versicolor) and os.path.isfile(testing_path_virginica):
        pass
    else:
        create_files = Iris_Dataset(dataset_path)
        create_files.preparing_dataset()

    
    
    while True:
        
        training_flower_type = input("Choose which Iris Flower type you would like to FIT your model on:"+'\n'+"Press 1 for Iris Setosa, 2 for Iris Versicolor, 3 for Iris Virginica, or 0 to exit: ")
        
        if training_flower_type == '0':
            break
        elif training_flower_type == '1':
            training_path = training_path_setosa
        elif training_flower_type == '2':
            training_path = training_path_versicolor
        elif training_flower_type == '3':
            training_path = training_path_virginica
        else:
            print("Incorrect input for training flower type!")
        
        testing_flower_type = input("Choose which Iris Flower type you would like to EVALUATE your model on:"+'\n'+"Press 1 for Iris Setosa, 2 for Iris Versicolor, 3 for Iris Virginica, or 0 to exit: ")
        print('\n')
        
        if testing_flower_type == '0':
            break
        elif testing_flower_type == '1':
            testing_path = testing_path_setosa
        elif testing_flower_type == '2':
            testing_path = testing_path_versicolor
        elif testing_flower_type == '3':
            testing_path = testing_path_virginica
        else:
            print("Incorrect input for testing flower type!")

        combo_file_path = ''
        iris_moses = MOSES_Iris(training_path, testing_path, combo_file_path)
        iris_moses.generate_output_files()
