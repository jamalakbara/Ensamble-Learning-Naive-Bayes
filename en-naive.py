import csv
import random
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

#bikin bag
def bagging(atribut1,atribut2,label):
    bag_x1 = []
    bag_x2 = []
    bag_y = []

    panjang_bag = round(len(atribut1)*0.4)
    for i in range(0, panjang_bag):
        rand_index = random.randint(0,len(atribut1))
        if rand_index == 298:
            rand_index = 298 - random.randint(0,298)

        bag_x1.append(atribut1[rand_index])
        bag_x2.append(atribut2[rand_index])
        bag_y.append(label[rand_index])

    return bag_x1, bag_x2, bag_y

#naive bayes nya
def naive_bayes(train_att1, train_att2, y, test_att1, test_att2):
    #bikin label encoder
    le = preprocessing.LabelEncoder()

    #convert isi dari datatrain jadi diskret number
    train_att1_encode = le.fit_transform(train_att1)
    train_att2_encode = le.fit_transform(train_att2)
    y_encode = le.fit_transform(y)

    #gabungin train_att1 sama train_att2
    features = list(zip(train_att1_encode, train_att2_encode))

    #bikin gaussian classifier
    model = GaussianNB()

    #train model dengan datatrain
    model.fit(features, y_encode)

    # convert isi dari datatest jadi diskret number
    test_att1_encode = le.fit_transform(test_att1)
    test_att2_encode = le.fit_transform(test_att2)

    hasil_prediksi = []
    #prediksi/klasifikasi naive bayes
    for i in range(0, len(test_att1)):
        predicted = model.predict([[test_att1_encode[i], test_att2_encode[i]]])
        hasil_prediksi.append(predicted[0])

    return hasil_prediksi

with open('TrainsetTugas4ML.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    x1_train = []
    x2_train = []
    y_train = []

    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            att1 = row[0]
            att2 = row[1]
            label = row[2]
            x1_train.append(att1)
            x2_train.append(att2)
            y_train.append(label)

with open('TestsetTugas4ML.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    x1_test = []
    x2_test = []

    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            att1 = row[0]
            att2 = row[1]
            x1_test.append(att1)
            x2_test.append(att2)

#bikin 4 bags berbasis naive bayes buat bagging nya
bag1 = bagging(x1_train, x2_train, y_train)
bag2 = bagging(x1_train, x2_train, y_train)
bag3 = bagging(x1_train, x2_train, y_train)
bag4 = bagging(x1_train, x2_train, y_train)

#memasukkan hasil prediksi/klasifikasi naive bayes ke variable
hasil_bag1 = naive_bayes(bag1[0],bag1[1],bag1[2],x1_test,x2_test)
hasil_bag2 = naive_bayes(bag2[0],bag2[1],bag2[2],x1_test,x2_test)
hasil_bag3 = naive_bayes(bag3[0],bag3[1],bag3[2],x1_test,x2_test)
hasil_bag4 = naive_bayes(bag4[0],bag4[1],bag4[2],x1_test,x2_test)

#voting dari setiap bags untuk dipilih class/label yg akan dipake
array_hasil = []
for idx in range(0, len(hasil_bag1)):
    count1 = 0
    count2 = 0
    if hasil_bag1[idx] == 1:
        count1 += 1
    elif hasil_bag1[idx] == 0:
        count2 += 1

    if count1 > count2:
        array_hasil.append(1)
    elif count1 < count2:
        array_hasil.append(2)

#write file into csv
with open("TebakanTugas4ML.csv", 'w', newline='') as csv_file:
    writeCsv = csv.writer(csv_file)

    for row in array_hasil:
        writeCsv.writerow([row])