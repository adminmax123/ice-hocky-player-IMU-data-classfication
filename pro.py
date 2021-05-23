import pandas as pd
import os

# os.chdir("D:/learn")


def process(name):
    zph = pd.read_csv(name+'.csv')
    Leftax = []
    Leftay = []
    Leftaz = []
    Leftgx = []
    Leftgy = []
    Leftgz = []
    Rightax = []
    Rightay = []
    Rightaz = []
    Rightgx = []
    Rightgy = []
    Rightgz = []
    a = zph.Leftay[1].split('[')[1].split(']')[0].split(",")
    # print(len(a))
    for i in zph.Rightay:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Rightay.append(int(j))
    Rightay = arr_size(Rightay, 112)

    for i in zph.Rightax:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Rightax.append(int(j))
    Rightax = arr_size(Rightax, 112)

    for i in zph.Rightaz:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Rightaz.append(int(j))
    Rightaz = arr_size(Rightaz, 112)

    for i in zph.Rightgy:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Rightgy.append(int(j))
    Rightgy = arr_size(Rightgy, 112)

    for i in zph.Rightgx:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Rightgx.append(int(j))
    Rightgx = arr_size(Rightgx, 112)

    for i in zph.Rightgz:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Rightgz.append(int(j))
    Rightgz = arr_size(Rightgz, 112)

    for i in zph.Leftay:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Leftay.append(int(j))
    Leftay = arr_size(Leftay, 112)

    for i in zph.Leftax:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Leftax.append(int(j))
    Leftax = arr_size(Leftax, 112)

    for i in zph.Leftaz:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Leftaz.append(int(j))
    Leftaz = arr_size(Leftaz, 112)

    for i in zph.Leftgy:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Leftgy.append(int(j))
    Leftgy = arr_size(Leftgy, 112)

    for i in zph.Leftgx:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Leftgx.append(int(j))
    Leftgx = arr_size(Leftgx, 112)

    for i in zph.Leftgz:
        a = i.split('[')[1].split(']')[0].split(",")
        for j in a:
            Leftgz.append(int(j))
    Leftgz = arr_size(Leftgz, 112)

    zph['Rightgx'] = Rightgx
    zph['Rightgy'] = Rightgy
    zph['Rightgz'] = Rightgz
    zph['Rightax'] = Rightax
    zph['Rightay'] = Rightay
    zph['Rightaz'] = Rightaz
    zph['Leftgx'] = Leftgx
    zph['Leftgy'] = Leftgy
    zph['Leftgz'] = Leftgz
    zph['Leftax'] = Leftax
    zph['Leftay'] = Leftay
    zph['Leftaz'] = Leftaz
    zph.drop(['Time'], axis=1, inplace=True)
    return zph


def arr_size(arr, size):
    s = []
    for i in range(0, int(len(arr)), size):
        c = arr[i:i+size]
        s.append(c)
    return s
