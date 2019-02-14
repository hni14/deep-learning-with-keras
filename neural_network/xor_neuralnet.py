#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np 

# XORゲートの2層ネットワークの定義
model = Sequential()
# 入力層および隠れ層
model.add(Dense(input_dim=2,                # 入力層のノード数
                units=2,                    # 隠れ層のノード数
                kernel_initializer='zeros', # 隠れ層の重み初期値
                bias_initializer='zeros',   # 隠れ層のバイアス初期値
                activation='sigmoid'))      # 隠れ層の活性化関数
# 出力層
model.add(Dense(units=1,                    # 出力層のノード数
                kernel_initializer='zeros', # 出力層の重み初期値
                bias_initializer='zeros',   # 出力層のバイアス初期値
                activation='sigmoid'))      # 出力層の活性化関数

# 重みとバイアスを表示する。
for layer in model.layers:
    print('Weights:', np.array(layer.get_weights()[0]).flatten())
    print('Biases:', layer.get_weights()[1])

# 実行
x = np.array([[0,0],[0,1],[1,0],[1,1]]) # 入力データ
y_pred = model.predict(x) # 
print('x1  x2  y:\n', np.hstack((x, y_pred)))