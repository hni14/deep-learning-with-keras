#!/usr/bin/env python
from keras.backend import relu
import tensorflow
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np

# ステップ関数
def step(x):
   x = tensorflow.sign ( tensorflow.sign( x ) + 0.1 )
   return relu(x, alpha=0., max_value=1, threshold=0.)

# ANDゲートの1層ネットワークの定義
model = Sequential()

# 入力層および出力層
model.add(Dense(input_dim=2,                 # 入力層のノード数
                units=1,                     # 出力層のノード数
                kernel_initializer='zeros',  # 出力層の重み初期値
                bias_initializer='zeros',    # 出力層のバイアス初期値
                activation=step))            # 出力層の活性化関数

# 出力層の重み、バイアスの設定
## AND 
## x1に対する重み： 0.5
## x2に対する重み： 0.5
## バイアス： -0.7
output_layer = model.layers[0]
output_layer_weights = output_layer.get_weights()
output_layer_weights[0][0][0] = 0.5
output_layer_weights[0][1][0] = 0.5
output_layer_weights[1][0] = -0.7
output_layer.set_weights(output_layer_weights)

# 出力層の重みとバイアスを表示する。
print('Output Layer Weights:', np.array(output_layer.get_weights()[0]).flatten())
print('Output Layer Biases:', output_layer.get_weights()[1])

# 実行
x = np.array([[0,0],[0,1],[1,0],[1,1]]) # 入力データ
y_pred = model.predict(x) # 順伝搬
print('x1  x2  y:\n', np.hstack((x, y_pred)))
