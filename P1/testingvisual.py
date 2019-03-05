import pandas as pd
import seaborn
import matplotlib.pyplot as plt

file1 = open('./files/iris.data', 'r')
datalist =  []
line = '0'
while line != '':
    line = file1.readline()
    datalist.append(line)
# print(len(datalist))
file1.close()

samplenumber= 150
while (len(datalist)-samplenumber) != 0:
    del datalist[len(datalist)-1]

SL = []
SW = []
PL = []
PW = []
CLASS = []
for i in range(len(datalist)):
    datalist[i] = datalist[i].replace(',',' ')
    datalist[i] = datalist[i].replace('\n','')
    SL.append(datalist[i].split()[0])
    SW.append(datalist[i].split()[1])
    PL.append(datalist[i].split()[2])
    PW.append(datalist[i].split()[3])
    CLASS.append(datalist[i].split()[4])
# For next steps in order to work, seaborn is required
