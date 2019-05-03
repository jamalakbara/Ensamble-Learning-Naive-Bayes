import csv
import random
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

def open_file(file):
    global x1, x2, y
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file)
        skip_row = 0

        if file == 'TrainsetTugas4ML.csv':
            x1, x2, y = [], [], []

            for row in csv_reader:
                if skip_row == 0:
                    skip_row = 1
                else:
                    att1 = row[0]
                    att2 = row[1]
                    klass = row[2]
                    x1.append(att1)
                    x2.append(att2)
                    y.append(klass)

            return x1, x2, y
        elif file == 'TestsetTugas4ML.csv':
            x1, x2 = [], []

            for row in csv_reader:
                if skip_row == 0:
                    skip_row = 1
                else:
                    att1 = row[0]
                    att2 = row[1]
                    x1.append(att1)
                    x2.append(att2)

            return x1, x2

def save_file(hasil):
    with open("TebakanTugas4ML.csv", 'w', newline='') as csv_file:
        writeCsv = csv.writer(csv_file)

        for row in hasil:
            writeCsv.writerow([row])

#bikin bag
def bagging(x1,x2,y):
    bag_x1, bag_x2, bag_y = [], [], []
    for i in range(round(len(x1)*0.6)):
        rand_index = random.randint(0,len(x1))
        if rand_index == 298:
            rand_index = 297

        bag_x1.append(x1[rand_index])
        bag_x2.append(x2[rand_index])
        bag_y.append(y[rand_index])

    return bag_x1, bag_x2, bag_y

#naive bayes nya
def naive_bayes(x1, x2, y, test1, test2):
    #bikin label encoder
    le = preprocessing.LabelEncoder()

    #convert isi dari datatrain jadi diskret number
    x1_encode = le.fit_transform(x1)
    x2_encode = le.fit_transform(x2)
    y_encode = le.fit_transform(y)

    #gabungin x1 sama x2
    features = list(zip(x1_encode, x2_encode))

    #bikin gaussian classifier
    model = GaussianNB()

    #train model dengan datatrain
    model.fit(features, y_encode)

    # convert isi dari datatest jadi diskret number
    test1_encode = le.fit_transform(test1)
    test2_encode = le.fit_transform(test2)

    hasil_prediksi = []
    #prediksi/klasifikasi naive bayes
    for i in range(len(test1)):
        predicted = model.predict([[test1_encode[i], test2_encode[i]]])
        hasil_prediksi.append(predicted[0])

    return hasil_prediksi

trainset = open_file('TrainsetTugas4ML.csv')

#read testset
testset = open_file('TestsetTugas4ML.csv')

#bikin 5 bags berbasis naive bayes buat bagging nya
bag1 = bagging(trainset[0], trainset[1], trainset[2]); bag2 = bagging(trainset[0], trainset[1], trainset[2]); bag3 = bagging(trainset[0], trainset[1], trainset[2]); bag4 = bagging(trainset[0], trainset[1], trainset[2]); bag5 = bagging(trainset[0], trainset[1], trainset[2])

#memasukkan hasil prediksi/klasifikasi naive bayes ke matriks
matriks = np.matrix(
    [naive_bayes(bag1[0],bag1[1],bag1[2],testset[0],testset[1]),
     naive_bayes(bag2[0],bag2[1],bag2[2],testset[0],testset[1]),
     naive_bayes(bag3[0],bag3[1],bag3[2],testset[0],testset[1]),
     naive_bayes(bag4[0],bag4[1],bag4[2],testset[0],testset[1]),
     naive_bayes(bag5[0],bag5[1],bag5[2],testset[0],testset[1])]
)

#voting dari setiap bags untuk dipilih class/label yg akan dipake
final_pref = []
for i in range(len(testset[0])):
    class1, class2 = 0, 0
    #ngitung banyakan class yang 2 apa 1
    for j in range(len(matriks)):
        if matriks[j, i] == 0:
            class2 += 1
        else:
            class1 += 1

    if class1 > class2:
        final_pref.append(1)
    else:
        final_pref.append(2)

save_file(final_pref)