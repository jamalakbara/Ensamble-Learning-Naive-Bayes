import csv
import random
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

#bikin bag
def bagging(x1,x2,y):
    bag = []
    for i in range(round(len(x1)*0.6)):
        rand_index = random.randint(0,len(x1))
        if rand_index == 298:
            rand_index = 297
        bag.append([x1[rand_index], x2[rand_index], y[rand_index]])

    return bag

#naive bayes nya
def naive_bayes(x1, x2, y):
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

    predicted = model.predict([[231, 61]])

#bikin 5 bags berbasis buat bagging nya
# bag1 = bagging(); bag2 = bagging(); bag3 = bagging(); bag4 = bagging(); bag5 = bagging()

trainset = open_file('TrainsetTugas4ML.csv')

bag1 = bagging(trainset[0], trainset[1], trainset[2]); bag2 = bagging(trainset[0], trainset[1], trainset[2]); bag3 = bagging(trainset[0], trainset[1], trainset[2]); bag4 = bagging(trainset[0], trainset[1], trainset[2]); bag5 = bagging(trainset[0], trainset[1], trainset[2])
