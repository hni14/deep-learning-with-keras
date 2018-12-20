'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# ミニバッチのサイズ
batch_size = 128

# 分類するクラス数
num_classes = 10

# エポック数
epochs = 20

# MNISTの画像データ、クラスラベルを訓練データ、テストデータとして読み込む。
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# サンプル画像のテンソル化
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 画像データのグレースケール値を正規化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# クラスベクトル(0〜9)をcategorical_crossentropyで用いるため、
# バイナリ(0, 1)クラス行列に変換する。
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ニューラルネットワークの作成
# 過学習に対して、ドロップアウト法
# 他クラス分類問題のため、出力層の活性化関数にソフトマックスを使用する(10分類)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# 作成したニューラルネットワークに、学習のための機構を組み合わせる。
# 損失関数: 訓練データでのネットワークの性能を評価し、ネットワークの重みをどの方向に更新するか決める。
# オプティマイザ：与えられたデータと損失関数に基づいてネットワークが自身の重みを更新する方法
# メトリックス： 訓練とテストを監視するための評価指標(ここでは画像が正しく分類された割合である正解率)
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 訓練の実施
# モデルを訓練データに適合させるために、訓練画像データ、クラスラベルを渡す。
# ミニバッチ学習のためのバッチサイズ、および、エポック数を定義する。
# テスト画像データ、クラスラベルを検証データとして渡しており、ハイパーパラメータの更新に利用される。
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# 訓練後のモデルの汎化性能をテストデータに対して評価する。
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
