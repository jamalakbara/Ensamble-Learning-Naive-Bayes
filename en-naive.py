import csv
import random
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

#naive bayes nya menggunakan multinomialNB
def NB(x1, x2, label, x1_test, x2_test):
    #bikin label encoder
    pre = preprocessing.LabelEncoder()

    #convert isi dari datatrain jadi diskret number
    x1_encoded = pre.fit_transform(x1)
    x2_encoded = pre.fit_transform(x2)
    label_encoded = pre.fit_transform(label)

    #gabungin x1 sama x2
    features = list(zip(x1_encoded, x2_encoded))

    #bikin multinomial classifier
    multi = MultinomialNB()

    #train modul dengan datatrain
    multi.fit(features, label_encoded)

    # convert isi dari datatest jadi diskret number
    x1_test_encoded = pre.fit_transform(x1_test); x2_test_encoded = pre.fit_transform(x2_test)

    result = []
    i = 0
    #prediksi/klasifikasi naive bayes
    while i < len(x1_test):
        predicted = multi.predict([[x1_test_encoded[i], x2_test_encoded[i]]])
        result.append(predicted[0])

        i += 1

    return result

#bikin modulnya ini
def bagging_modul(trainset1,trainset2,train_label):
    baggingAtribut1 = []; baggingAtribut2 = []; bagginLabel = []

    length_bagging = round(len(trainset1)*0.5)

    i = 0
    while i < length_bagging:
        idx = random.randint(0,len(trainset1))
        if idx == 298:
            idx = random.randint(random.randint(0, 10), random.randint(50, 100))

        baggingAtribut1.append(trainset1[idx]); baggingAtribut2.append(trainset2[idx]); bagginLabel.append(train_label[idx])

        i += 1

    return baggingAtribut1, baggingAtribut2, bagginLabel

with open('TrainsetTugas4ML.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    lineSkip = 0
    x1_train, x2_train, y_train = [], [], []

    for row in csv_reader:
        if lineSkip == 0:
            lineSkip = 1
        else:
            x1_train.append(row[0])
            x2_train.append(row[1])
            y_train.append(row[2])

with open('TestsetTugas4ML.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    lineSkip = 0
    x1_test, x2_test = [], []

    for row in csv_reader:
        if lineSkip == 0:
            lineSkip = 1
        else:
            x1_test.append(row[0])
            x2_test.append(row[1])


#bikin 3 bags berbasis naive bayes buat bagging bagging_modul nya
bagging = [
    bagging_modul(x1_train, x2_train, y_train),
    bagging_modul(x1_train, x2_train, y_train),
    bagging_modul(x1_train, x2_train, y_train)
]

#memasukkan hasil prediksi/klasifikasi naive bayes ke array
hasil_bagging = []
for i in range(len(bagging)):
    hasil_bagging.append(NB(bagging[i][0],bagging[i][1],bagging[i][2],x1_test,x2_test))

#voting dari setiap bags untuk dipilih class/label yg akan dipake
i = 0
hasil = []
while i < len(hasil_bagging[0]):
    pilih1 = 0; pilih2 = 0
    j = 0
    while j < len(hasil_bagging):
        if hasil_bagging[j][i] == 0:
            pilih2 += 1
        else:
            pilih1 += 1

        j += 1

    if pilih1 > pilih2:
        hasil.append(1)
    else:
        hasil.append(2)

    i += 1

with open("TebakanTugas4ML.csv", 'w', newline='') as file:
    save = csv.writer(file)

    for line in hasil:
        save.writerow([line])