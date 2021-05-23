import math
from scipy.fftpack import fft, ifft
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import pro
import warnings
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


# ——————————————————————————————————————————————————————
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

Train['after'] = Train.apply(change, axis=1)

y_orgin = []
y_after_fft = []
y_after_fft = []
for i in range(0, len(Train['after'])):
    y_orgin = Train['after'][i]
    yy = fft(y_orgin)
    yreal = yy.real               # 获取实数部分
    yimag = yy.imag               # 获取虚数部分
    test_y = yy
    for i in range(len(fft(y_orgin))):
        if i >= 15:
            test_y[i] = 0
    test = np.fft.ifft(test_y)  # 对变换后的结果应用ifft函数，应该可以近似地还原初始信号。
    y = test
    test = test.real
    test = test.tolist()
    y_after_fft.append(test)

weights = ['distance', 'uniform']
n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
p = [1, 2, 3, 4, 5, 6, 7]

param_grid = [
    {
        'weights': weights,
        'n_neighbors': n_neighbors,
        # 'algorithm': algorithm,
        'p': p
    }

]
my_scorer = make_scorer(
    accuracy_score,  greater_is_better=True)
x_train, x_test, y_train, y_test = train_test_split(
    y_after_fft, Sort, train_size=0.8, random_state=2)
# linear线性 poly多项式 rbf高斯径向基 sigmoid双曲正切
model = KNeighborsClassifier()
gridsearch = GridSearchCV(model, param_grid, scoring=my_scorer,
                          cv=5, n_jobs=-1, verbose=2,)
gridsearch.fit(x_train, y_train)
print(gridsearch.best_score_, gridsearch.best_estimator_)

#y_pred = model.predict(x_test)
#target_names = ['正平滑', '左压步', '右压步']
#print(classification_report(y_test, y_pred, target_names=target_names))
