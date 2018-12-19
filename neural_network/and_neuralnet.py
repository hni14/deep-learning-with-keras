from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np 

# 訓練データ
x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([0,0,0,1])

# テストデータ
x_test = x_train
y_test = y_train

# ニューラルネットワークの定義
model = Sequential()
model.add(Dense(units=1, 
                input_dim=2, 
                activation='sigmoid'))

# 定義したニューラルネットワークに、学習のための機構を組み合わせる。
# 損失関数: 訓練データでのネットワークの性能を評価し、ネットワークの重みをどの方向に更新するか決める。
# オプティマイザ：与えられたデータと損失関数に基づいてネットワークが自身の重みを更新する方法
# メトリックス： 訓練とテストを監視するための評価指標
#model.compile(loss='binary_crossentropy',
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.1),
              metrics=['binary_accuracy'])

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

# 重みとバイアスを表示する。
for layer in model.layers:
    print('Weights:', np.array(layer.get_weights()[0]).flatten())
    print('Biases:', layer.get_weights()[1])

