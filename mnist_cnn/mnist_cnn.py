#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# ミニバッチのサイズ
batch_size = 128

# 分類するクラス数
num_classes = 10

# エポック数
epochs = 12

# 入力画像の次元数
img_rows, img_cols = 28, 28

# MNISTの画像データ、クラスラベルを訓練データ、テストデータとして読み込む。
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 画像データのテンソルに深さ(カラーチャネル)の軸を加える。
# RBGでは赤、青、緑のカラー値ごとに、それぞれチャネルがあり、3つのカラーチャネルだが、グレースケールでは1つのチャネル。
# https://helpx.adobe.com/jp/photoshop/using/image-essentials.html#color_channels
if K.image_data_format() == 'channels_first':
    # KerasバックエンドがTheanoの場合
    # データ・セットの形状(n、 幅、高さ)を(n、深さ、幅、高さ)に変換する。
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # KerasバックエンドがTensorflowの場合
    # データ・セットの形状(n、 幅、高さ)を(n、幅、高さ、深さ)に変換する。
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 画像データのグレースケール値を正規化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# クラスベクトル(0〜9)をcategorical_crossentropyで用いるため、
# バイナリ(0, 1)のクラス行列に変換する。
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# モデルアーキテクチャの定義
# タンジュ
# 過学習に対して、ドロップアウト法
# 出力層の活性化関数にソフトマックスを使用する(10分類)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# モデルをコンパイルする。
# 定義したニューラルネットワークに、学習のための機構を組み合わせる。
# 損失関数: 訓練データでのネットワークの性能を評価し、ネットワークの重みをどの方向に更新するか決める。
# オプティマイザ：与えられたデータと損失関数に基づいてネットワークが自身の重みを更新する方法
# メトリックス： 訓練とテストを監視するための評価指標(ここでは画像が正しく分類された割合である正解率)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 訓練の実施
# モデルを訓練データに適合させるために、訓練画像データ、クラスラベルを渡す。
# ミニバッチ学習のためのバッチサイズ、および、エポック数を定義する。
# テスト画像データ、クラスラベルを検証データとして渡しており、ハイパーパラメータの更新に利用される。
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 訓練後のモデルの汎化性能をテストデータに対して評価する。
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 学習後のモデルの保存
model.save('mnist_cnn_model.h5')