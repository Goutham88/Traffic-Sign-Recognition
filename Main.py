import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Activation,Flatten
import os
from keras import backend as K
from skimage import io,color
import csv
aaaaaa=[]
with open("test_lab.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        aaaaaa.append((row[0].split(';'))[-1])


model=Sequential()
model.add(Conv2D(100, kernel_size=(3, 3), strides=(1,1),input_shape=(48,48,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(150,(4,4),strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(250,(3,3),strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
conv_out=Conv2D(200,(4,4),strides=(1,1))
model.add(conv_out)
model.add(Flatten())
model.add(Dense(43))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

keras.models.load_model('my_model.h5')

folders=[[] for _ in range(43)]
times=[]
for i in range(43):
    if (i < 10):
        path = '0000' + str(i)
    else:
        path = '000' + str(i)
    dirs = os.listdir('C:\\Users\\Goutham Chunduru\\Desktop\\Project\\TSR\\TrafficDataset\\' + path)
    folders[i]=dirs
    times.append(len(folders[i])//30)

# Defining Y
y=[[0 for x in range(43)] for _ in range(1356)]
q=0
for ert in range(43):
    for er in range(times[ert]):
        y[er+q][ert]=1
    q+=times[ert]
y=np.array(y)

qwqeqweqw=0
for i in range(30):
    print (i)
    x=[]
    for j in range(43):
        #print(j)
        if (j < 10):
            path = '0000' + str(j)
        else:
            path = '000' + str(j)
        for k in range(times[j]):
            #print (k,qwqeqweqw)
            qwqeqweqw+=1
            impath = 'C:\\Users\\Goutham Chunduru\\Desktop\\Project\\TSR\\TrafficDataset\\' + path + '\\' + folders[j][times[j]*i+k]
            rgb = io.imread(impath)
            lab = color.rgb2lab(rgb)
            p = [[0] * len(lab[0])] * (len(lab))
            for q in range(len(lab)):
                for qq in range(len(lab[0])):
                    p[q][qq] = (lab[q][qq][0] + lab[q][qq][1] + lab[q][qq][2]) / 3
            pad = [[0] * 48] * 48
            for q in range(len(lab)):
                for qq in range(len(lab[0])):
                    pad[q][qq] += p[q][qq]
            pad = np.array(pad)
            #y = np.array(y)
            x.append(pad)
    x=np.array(x)
    x=x.reshape(1356,48,48,1)
    y=y.reshape(1356,43)
    #print(x.shape,y.shape)
    model.fit(x, y, epochs=3)
    print("HI")

    get_9th_layer_output = K.function([model.layers[0].input], [model.layers[9].output])
    layer_output = get_9th_layer_output([x])[0]
    layer_output = np.array(layer_output)
    # print(layer_output)
    layer_output = layer_output.reshape(1356, 200)
    np.savetxt(""+str(i)+".csv", layer_output, delimiter=",")

model.save('my_model.h5')

#Testing
y_test=[[0 for x in range(43)] for y in range(12630)]
for iop in range(12630):
    y_test[iop][int(aaaaaa[iop])]=1

xx=[]

dir=os.listdir('C:\\Users\\Goutham Chunduru\\Desktop\\Project\\TSR\\TrafficTestDataset\\')
for j in dir:
    print (3)
    impath = 'C:\\Users\\Goutham Chunduru\\Desktop\\Project\\TSR\\TrafficTestDataset\\' + j
    rgb = io.imread(impath)
    lab = color.rgb2lab(rgb)
    p = [[0] * len(lab[0])] * (len(lab))
    p = [[0] * len(lab[0])] * (len(lab))
    for q in range(len(lab)):
        for qq in range(len(lab[0])):
            p[q][qq] = (lab[q][qq][0] + lab[q][qq][1] + lab[q][qq][2]) / 3
    pad = [[0] * 48] * 48
    for q in range(len(lab)):
        for qq in range(len(lab[0])):
            pad[q][qq] += p[q][qq]
    pad = np.array(pad)
    xx.append(pad)
xx=np.array(xx)
xx=xx.reshape(12630,48,48,1)
y_test=np.array(y_test)
y_test=y_test.reshape(12630,43)
score=model.evaluate(xx,y_test)
print (score)


get_9th_layer_output = K.function([model.layers[0].input], [model.layers[9].output])
layer_output = get_9th_layer_output([xx])[0]
layer_output = np.array(layer_output)
print(3)
layer_output = layer_output.reshape(12630, 200)
print(4)
np.savetxt("Test.csv", layer_output, delimiter=",")

