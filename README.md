# Deep learning with Keras
Kerasを中心に例題の解釈、ハンズオンを通じてディープラーニングに対する理解を深める。  

## システム要件

### オペレーティングシステム
- Ubuntu 16.04 or later (64-bit)
- macOS 10.12.6 (Sierra) or later (64-bit)
- Windows 7 or later (64-bit) (Python 3 only)

### python
- Python 2.7-3.6

## 例題

### パーセプトロンに関する例題
- [perceptron](perceptron) パーセプトロンで論理回路を作成する例

### ニューラルネットワークの学習に関する例題
- [neural_network](neural_network) 論理回路の訓練を行う例。
- [工事中] [optimizer](optimizer) 最適化手法を実装する例。

### コンピュータビジョンの関する例題
- [mnist_mlp](mnist_mlp) MNISTデータ・セットで多層パーセプトロンの訓練を行う例。
- [工事中] [mnist_cnn](mnist_cnn) MNISTデータ・セットで畳み込みニューラルネットワークの訓練を行う例。

### シーケンスに関する例題
TOBD

### おすすめの読み方
1. [perceptron](perceptron) パーセプトロンで論理回路を作成する例
2. [neural_network](neural_network) 論理回路の訓練を行う例。
3. [mnist_mlp](mnist_mlp) MNISTデータ・セットで多層パーセプトロンの訓練を行う例。
4. [mnist_cnn](mnist_cnn) MNISTデータ・セットで畳み込みニューラルネットワークの訓練を行う例。

## 出典
- [keras/examples](https://github.com/keras-team/keras/blob/master/examples/README.md)

## 参考文献
- Keras
  - [Keras公式ドキュメント](https://keras.io/ja/)
  - Francois Chollet (著): 「PythonとKerasによるディープラーニング」 マイナビ出版 刊
- ディープラーニング
  - 斎藤 康毅 (著): 「ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装」 オライリージャパン 刊
  - 斎藤 康毅 (著): 「ゼロから作るDeep Learning ❷ ―自然言語処理編」 オライリージャパン 刊
  - (参考予定) Charu C. Aggarwal (著)：「Neural Networks and Deep Learning: A Textbook」 Springer 刊
- 損失関数
  - [損失関数の利用方法](https://keras.io/ja/losses/)
  - [Loss functions in neural networks](https://isaacchanghau.github.io/post/loss_functions/)
- 評価関数
  - [評価関数の利用方法](https://keras.io/ja/metrics/)
- 最適化
  - [オプティマイザ（最適化アルゴリズム）の利用方法](https://keras.io/ja/optimizers/)
  - [勾配降下法の最適化アルゴリズムを概観する](https://postd.cc/optimizing-gradient-descent/)
- 活性化関数
  - [活性化関数の使い方](https://keras.io/ja/activations/)
  - [An overview of activation functions used in neural networks](https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html)
- 正則化
  - [正則化の利用方法](https://keras.io/ja/regularizers/)
