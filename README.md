# MNISTから画像データをロードする。
Kerasライブラリでは、組み込みでサンプルデータ・セットが用意されています。  
このMNISTというデータ・セットは、手書き数字の画像セットです。  
簡単な実験から論文で用いられる等よく利用されるデータ・セットです。  
次のように読み込むことができます。  

```python
from keras.datasets import mnist
 
# データ・セットを訓練データ、テストデータに分けて読み込みできる。
(train_images), train_labels), (test_images, test_labels) = mnist.load_data()
```

train_imagesは訓練用の画像データであり、train_labelsは訓練用の画像に対するラベル(1が描かれてる画像に対して、1というラベルがつけれている)。

データ・セットの形状を確認します。  
訓練データ・セットには６万のサンプル画像があり、それぞれ28ピクセル×28ピクセルです。  

```python
print(X_train.shape)
# (60000, 28, 28)
```

これを確認するために、matplotlibで可視化してみます。  

```python
from matplotlib import pyplot as plt
plt.imshow(X_train[0])
```



# Kerasの入力データを前処理する。

各ピクセル


# reference
https://elitedatascience.com/keras-tutorial-deep-learning-in-python#step-5
