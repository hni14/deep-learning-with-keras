#!/usr/bin/env python
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import numpy as np 

# 訓練データ
x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([0,0,0,1])

# テストデータ
x_test = x_train
y_test = y_train

# ANDゲートの1層ネットワークの定義
model = Sequential()

## 出力層
model.add(Dense(input_dim=2,                    # 入力層のノード数
                units=1,                        # 出力層のノード数
                kernel_initializer='zeros',     # 出力層の重み初期値
                bias_initializer='zeros',       # 出力層のバイアス初期値
                activation='sigmoid'))          # 出力層の活性化関数

## 定義したニューラルネットワークに、学習のための機構を組み合わせる。
model.compile(loss='mean_squared_error',        # 損失関数: 訓練データでのネットワークの性能を評価し、ネットワークの重みをどの方向に更新するか決める。
              optimizer=Adam(lr=0.1),           # オプティマイザ：与えられたデータと損失関数に基づいてネットワークが自身の重みを更新する方法
              metrics=['binary_accuracy'])      # メトリックス： 訓練とテストを監視するための評価指標

# モデルの訓練
history = model.fit(x_train, y_train,
                    batch_size=1,
                    epochs=1000,
                    verbose=1,
                    validation_data=(x_test, y_test))

# テスト
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

## 重みとバイアスを表示する。
for layer in model.layers:
    print('Weights:', np.array(layer.get_weights()[0]).flatten())
    print('Biases:', layer.get_weights()[1])

# 実行
y_pred = model.predict(x_train)
print('x1  x2  y:\n', np.hstack((x_train, y_pred)))