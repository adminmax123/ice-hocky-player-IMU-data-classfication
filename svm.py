import math
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, make_scorer
from scipy.fftpack import fft, ifft
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pro
import warnings
import time


warnings.filterwarnings("ignore")
zph = pro.process('zph')
xzzyb = pro.process('xzzyb')
xyzyb = pro.process('xyzyb')
zph['Sort'] = 0
xzzyb['Sort'] = 1
xyzyb['Sort'] = 2

frames = [zph, xzzyb, xyzyb]
train = pd.concat(frames, ignore_index=True)
Sort = train['Sort'].values.tolist()
# Leftax = train['Leftax'].values.tolist()
# Rightgy = train['Rightgy'].values.tolist()
# Rightax = train['Rightax'].values.tolist()
# Rightay = train['Rightay'].values.tolist()
# Rightgx = train['Rightgx'].values.tolist()
# Leftaz = train['Leftaz'].values.tolist()
# Leftgz = train['Leftgz'].values.tolist()
# Leftgy = train['Leftgy'].values.tolist()
# Leftgx = train['Leftgx'].values.tolist()
# Leftay = train['Leftay'].values.tolist()
# Rightaz = train['Rightaz'].values.tolist()
# Rightgz = train['Rightgz'].values.tolist()
features = ['Leftgx', 'Leftgy', 'Leftgz', 'Rightgx', 'Rightgy', 'Rightgz',
            'Leftax', 'Leftay', 'Leftaz', 'Rightax', 'Rightay', 'Rightaz']
Train = train[features]


def change(df):
    combine = []
    for i in df.values:
        combine.extend(i)
    return combine


def Left_A(df):
    x = df['Leftax']
    y = df['Leftay']
    z = df['Leftaz']
    res = list_add(x, y, z)
    return res


def Left_G(df):
    x = df['Leftgx']
    y = df['Leftgy']
    z = df['Leftgz']
    res = list_add(x, y, z)
    return res


def Right_A(df):
    x = df['Rightax']
    y = df['Rightay']
    z = df['Rightaz']
    res = list_add(x, y, z)
    return res


def Right_G(df):
    x = df['Rightgx']
    y = df['Rightgy']
    z = df['Rightgz']
    res = list_add(x, y, z)
    return res


def AX(df):
    x = df['Leftax']
    y = df['Leftay']
    z = df['Leftaz']
    res = list_abs(x, y, z)
    return res


def AY(df):
    x = df['Leftgx']
    y = df['Leftgy']
    z = df['Leftgz']
    res = list_abs(x, y, z)
    return res


def AZ(df):
    x = df['Rightax']
    y = df['Rightay']
    z = df['Rightaz']
    res = list_abs(x, y, z)
    return res


def AW(df):
    x = df['Rightgx']
    y = df['Rightgy']
    z = df['Rightgz']
    res = list_abs(x, y, z)
    return res


def GX(df):
    """
    z=Lx-Rx
    param:Lx Rx
    return: z
    """
    x = df['Rightgx']
    y = df['Leftgx']
    z = list_minus(x, y)
    return z


def GY(df):
    """
    z=Lx-Rx
    param:Lx Rx
    return: z
    """
    x = df['Rightgy']
    y = df['Leftgy']
    z = list_minus(x, y)
    return z


def GZ(df):
    """
    z=Lx-Rx
    param:Lx Rx
    return: z
    """
    x = df['Rightgz']
    y = df['Leftgz']
    z = list_minus(x, y)
    return z


def list_minus(a, b):
    d = []
    for i in range(len(a)):
        d.append(abs(a[i])-abs(b[i]))
    return d


def list_add(a, b, c):
    d = []
    for i in range(len(a)):
        d.append(math.sqrt(a[i]**2+b[i]**2+c[i]**2))
    return d


def list_abs(a, b, c):
    d = []
    for i in range(len(a)):
        d.append(abs(a[i])+abs(b[i])+abs(c[i]))
    return d


# ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
Train = train[features]
Train['Right_G'] = Train.apply(Right_G, axis=1)
Train['Right_A'] = Train.apply(Right_A, axis=1)
Train['Left_G'] = Train.apply(Left_G, axis=1)
Train['Left_A'] = Train.apply(Left_A, axis=1)
Train['GX'] = Train.apply(GX, axis=1)
Train['GY'] = Train.apply(GY, axis=1)
Train['GZ'] = Train.apply(GZ, axis=1)
Train['AX'] = Train.apply(AX, axis=1)
Train['AY'] = Train.apply(AY, axis=1)
Train['AZ'] = Train.apply(AZ, axis=1)
Train['AW'] = Train.apply(AZ, axis=1)
Train['after'] = Train.apply(change, axis=1)


y_orgin = []
y_after_fft = []
for i in range(0, len(Train['after'])):
    y_orgin = Train['after'][i]
    yy = fft(y_orgin)
    yreal = yy.real               # ??????????????????
    yimag = yy.imag               # ??????????????????
    test_y = yy
    for i in range(len(fft(y_orgin))):
        if i >= 15:
            test_y[i] = 0
    test = np.fft.ifft(test_y)  # ???????????????????????????ifft???????????????????????????????????????????????????
    y = test
    test = test.real
    test = test.tolist()
    y_after_fft.append(test)


start = time.clock()
x_train, x_test, y_train, y_test = train_test_split(
    y_after_fft, Sort, train_size=0.8, random_state=3)
model = SVC(C=1, degree=2, gamma=1, kernel='poly')
# linear?????? poly????????? rbf??????????????? sigmoid????????????
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
target_names = ['?????????', '?????????', '?????????']
print(classification_report(y_test, y_pred, target_names=target_names))
print(accuracy_score(y_test, y_pred))

end = time.clock()
print('Running time: %s Seconds' % (end-start))
