import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

#print (disease.describe().transpose())
#print (disease.shape)


neuron_number = [3, 11, 101]
max_i = 6
labels = ['Disease', 'Temperature', 'Occurrence of nausea', 'Lumbar pain', 'Urine pushing', 'Micturition pains', 'Burning of urethra']


def find_best(array):
    best = array[0]
    best_value = 0
    temp_value = 0


    for item in array:
        for i in range (0,4):
            temp_value = temp_value + item[i][i]
        if best_value < temp_value:
            best_value = temp_value
            best = item
        temp_value = 0

    return best   

def count_wrong(array):
    wrong = 0.0

    for i in range(0,4):
        for j in range(0,4):
            if i != j:
                wrong = wrong + array[i][j]

    return wrong


def features_ranking(disease):
    X = disease.drop('Disease',axis=1)
    y = disease['Disease']
    print(X.shape)
    #skb = SelectKBest(chi2, k=2)
    skb = SelectKBest(f_classif, k=2)
    F_new = skb.fit_transform(X, y)

    print(skb.scores_)
    print(skb.pvalues_)

    scores = skb.scores_
    arr = []
    best = 0
    
    for i in range(0,max_i):
        for j in range(0,max_i):
            if j not in arr:
                if scores[j] > best:
                    best = scores[j]
                    num = j
        arr.append(num)
        best = 0

    return arr


def n_features(number_of_features, arr, disease, File_result, neurons):

    table_conf_mat = []
    table_conf_mat_tmp = []
    list_to_cut = []
    list_to_cut.append(labels[0])
    for i in range(number_of_features, max_i):
        list_to_cut.append(labels[arr[i]+1])

    print(list_to_cut)
    for i in range (1, 6):
        X = disease.drop(list_to_cut,axis=1)
        y = disease['Disease']
        #print(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        T_train = X_test
        T_test = X_train
        t_train = y_test
        t_test = y_train

        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(X_train)

        # Now apply the transformations to the data:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(neurons), solver='sgd', activation = 'tanh', max_iter=100, shuffle = True, momentum = 0.9)


        mlp.fit(X_train,y_train)
        predictions = mlp.predict(X_test)

        #print(confusion_matrix(y_test,predictions))
        #print(classification_report(y_test,predictions))
        table_conf_mat_tmp.append(confusion_matrix(y_test,predictions))


        #scaler2 = StandardScaler()
        # Fit only to the training data
        scaler.fit(T_train)

        # Now apply the transformations to the data:
        T_train = scaler.transform(T_train)
        T_test = scaler.transform(T_test)
        mlp = MLPClassifier(hidden_layer_sizes=(neurons), solver='sgd', activation = 'tanh', max_iter=100, shuffle = True, momentum = 0.9)

        mlp.fit(T_train,t_train)
        predictions = mlp.predict(T_test)

        #print(confusion_matrix(t_test,predictions))
        #print(classification_report(t_test,predictions))
        table_conf_mat_tmp.append(confusion_matrix(t_test,predictions))

        table_conf_mat_tmp[0] = table_conf_mat_tmp[0] + table_conf_mat_tmp[1]
        table_conf_mat.append(table_conf_mat_tmp[0])
        table_conf_mat_tmp.clear()


    best_conf_matrix = find_best(table_conf_mat)
    print('---------------------------------------')
    print (best_conf_matrix)
    print('---------------------------------------')
    for item in table_conf_mat:
        print(item)
    print('---------------------------------------')

    master_matrix = table_conf_mat[0]/5

    table_conf_mat.pop(0)

    print('All confusion matrixes added and averaged:')
    for item in table_conf_mat:
        master_matrix = master_matrix + item/5
    print(master_matrix)

    master_matrix = np.around(master_matrix, decimals=2)
    wrong = count_wrong(master_matrix)

    File_result.write('Confusion matrix with best score:\n')
    #File_result.write(best_conf_matrix)
    for i in range (0, 4):
        File_result.write('%d & ' % i)
        best_conf_matrix[i].tofile(File_result, sep=' & ', format='%s')
        File_result.write('\\\\')
        File_result.write('\n')
    
    File_result.write('All confusion matrixes added and averaged:\n')
    #File_result.write(master_matrix)   
    for i in range (0, 4):
        File_result.write('%d & ' % i)
        master_matrix[i].tofile(File_result, sep=' & ', format='%s')
        File_result.write('\\\\')
        File_result.write('\n')     
    File_result.write('\nWrong acc: %f\n' % wrong)

    table_conf_mat.clear()

#Obsolte function
def all_features(disease):
    for i in range (1, 6):
        X = disease.drop('Disease',axis=1)
        y = disease['Disease']


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        T_train = X_test
        T_test = X_train
        t_train = y_test
        t_test = y_train

        scaler = StandardScaler()
        # Fit only to the training data
        scaler.fit(X_train)

        # Now apply the transformations to the data:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(neuron_number), solver='sgd', activation = 'tanh', max_iter=100, shuffle = True, momentum = 0.0 )


        mlp.fit(X_train,y_train)
        predictions = mlp.predict(X_test)

        #print(confusion_matrix(y_test,predictions))
        #print(classification_report(y_test,predictions))
        table_conf_mat.append(confusion_matrix(y_test,predictions))


        #scaler2 = StandardScaler()
        # Fit only to the training data
        scaler.fit(T_train)

        # Now apply the transformations to the data:
        T_train = scaler.transform(T_train)
        T_test = scaler.transform(T_test)
        mlp = MLPClassifier(hidden_layer_sizes=(neuron_number), solver='sgd', activation = 'tanh', max_iter=100, shuffle = True, momentum = 0.0 )

        mlp.fit(T_train,t_train)
        predictions = mlp.predict(T_test)

        #print(confusion_matrix(t_test,predictions))
        #print(classification_report(t_test,predictions))
        table_conf_mat.append(confusion_matrix(t_test,predictions))


    best_conf_matrix = find_best(table_conf_mat)
    print('---------------------------------------')
    print (best_conf_matrix)
    print('---------------------------------------')
    for item in table_conf_mat:
        print(item)
    print('---------------------------------------')

    master_matrix = table_conf_mat[0]/10

    table_conf_mat.pop(0)

    print('All confusion matrixes added and averaged:')
    for item in table_conf_mat:
        master_matrix = master_matrix + item/10
    print(master_matrix)        

    File_result.write('Confusion matrix with best score:\n')
    #File_result.write(best_conf_matrix)
    for i in range (0, 4):
        best_conf_matrix[i].tofile(File_result, sep='\t', format='%s')
        File_result.write('\n')
    
    File_result.write('All confusion matrixes added and averaged:\n')
    #File_result.write(master_matrix)   
    for i in range (0, 4):
        master_matrix[i].tofile(File_result, sep='\t', format='%s')
        File_result.write('\n')   

def main():
    disease = pd.read_csv('data2.csv', names = labels)

    disease.head()

    File_result = open('result_file_nm3', 'w')
    File_result.write('Start of testing\n')

    #all_features(disease, File_result)
    rank_array = features_ranking(disease)
    #print (rank_array)
    for n in range(0,len(neuron_number)):
        File_result.write('+++++++++++++++++++++++++++++++++++++++++++\n')
        File_result.write('Test for %d number of neurons\n' % neuron_number[n])
        for i in range(1,max_i+1):
            File_result.write('-----------------------------\n')
            File_result.write('Results for %d features\n' % i)
            n_features(i, rank_array, disease, File_result, neuron_number[n])

#    for i in range(1,max_i+1):
#        File_result.write('-----------------------------\n')
#        File_result.write('Results for %d features\n' % i)
#        n_features(i, rank_array, disease, File_result, 13)


    File_result.close()
    

if __name__ == "__main__":
    main()
