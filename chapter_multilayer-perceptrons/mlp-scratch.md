# Implementierung von Multilayer Perceptrons von Grund auf
:label:`sec_mlp_scratch`

Nun, da wir mehrschichtige Wahrnehmungen (MLPs) mathematisch charakterisiert haben, lassen Sie uns versuchen, einen selbst zu implementieren. Um mit den bisherigen Ergebnissen der softmax-Regression (:numref:`sec_softmax_scratch`) zu vergleichen, werden wir weiterhin mit dem Fashion-MNIST-Bildklassifizierungsdatensatz (:numref:`sec_fashion_mnist`) arbeiten.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initialisieren von Modellparametern

Daran erinnern, dass Fashion-MNIST 10 Klassen enthält, und dass jedes Bild aus einem $28 \times 28 = 784$ Raster von Graustufenpixelwerten besteht. Auch hier werden wir die räumliche Struktur zwischen den Pixeln vorerst ignorieren, so dass wir uns dies als ein Klassifizierungsdataset mit 784 Eingabe-Features und 10 Klassen vorstellen können. Zunächst werden wir ein MLP mit einer versteckten Schicht und 256 versteckten Einheiten implementieren. Beachten Sie, dass wir diese beiden Mengen als Hyperparameter betrachten können. In der Regel wählen wir Layer-Breiten mit Potenzen von 2, die aufgrund der Zuweisung und Adressierung von Speicher in der Hardware eher rechnerisch effizient sind.

Auch hier werden wir unsere Parameter mit mehreren Tensoren darstellen. Beachten Sie, dass wir *für jede Ebenen* eine Gewichtsmatrix und einen Bias-Vektor verfolgen müssen. Wie immer weisen wir Speicher für die Gradienten des Verlustes in Bezug auf diese Parameter zu.

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## Aktivierungsfunktion

Um sicherzustellen, dass wir wissen, wie alles funktioniert, implementieren wir die ReLU-Aktivierung selbst mit der maximalen Funktion, anstatt die eingebaute `relu` Funktion direkt aufzurufen.

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## Modell

Da wir nicht berücksichtigen räumliche Struktur, wir `reshape` jedes zweidimensionale Bild in einen flachen Vektor der Länge `num_inputs`. Schließlich implementieren wir unser Modell mit nur wenigen Codezeilen.

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## Verlust-Funktion

Um numerische Stabilität zu gewährleisten und da wir die softmax-Funktion bereits von Grund auf neu implementiert haben (:numref:`sec_softmax_scratch`), nutzen wir die integrierte Funktion von High-Level-APIs zur Berechnung des softmax-und cross-entropie-Verlusts. Erinnern Sie sich an unsere frühere Diskussion dieser Feinheiten in :numref:`subsec_softmax-implementation-revisited`. Wir ermutigen den interessierten Leser, den Quellcode für die Verlustfunktion zu untersuchen, um ihr Wissen über Implementierungsdetails zu vertiefen.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## Ausbildung

Glücklicherweise ist die Trainingsschleife für MLPs genau die gleiche wie für die softmax-Regression. Wenn wir das Paket `d2l` wieder nutzen, rufen wir die Funktion `train_ch3` auf (siehe :numref:`sec_softmax_scratch`), wobei die Anzahl der Epochen auf 10 und die Lernrate auf 0,5 gesetzt wird.

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

Um das erlernte Modell zu bewerten, wenden wir es auf einige Testdaten an.

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## Zusammenfassung

* Wir haben gesehen, dass die Implementierung eines einfachen MLP einfach ist, selbst wenn es manuell gemacht wird.
* Bei einer großen Anzahl von Ebenen kann die Implementierung von MLPs von Grund auf immer noch chaotisch werden (z. B. Benennen und Verfolgen der Parameter unseres Modells).

## Übungen

1. Ändern Sie den Wert des Hyperparameters `num_hiddens`, und sehen Sie, wie sich dieser Hyperparameter auf die Ergebnisse auswirkt. Bestimmen Sie den besten Wert dieses Hyperparameters, wobei alle anderen konstant bleiben.
1. Versuchen Sie, einen zusätzlichen ausgeblendeten Layer hinzuzufügen, um zu sehen, wie er sich auf die Ergebnisse auswirkt.
1. Wie ändert sich das Ändern der Lernrate Ihre Ergebnisse? Wenn Sie die Modellarchitektur und andere Hyperparameter (einschließlich der Anzahl der Epochen) beheben, welche Lernrate liefert Ihnen die besten Ergebnisse?
1. Was ist das beste Ergebnis, das Sie erzielen können, wenn Sie alle Hyperparameter (Lernrate, Anzahl der Epochen, Anzahl der versteckten Ebenen, Anzahl der versteckten Einheiten pro Schicht) gemeinsam optimieren?
1. Beschreiben Sie, warum es viel schwieriger ist, mit mehreren Hyperparametern umzugehen.
1. Was ist die intelligenteste Strategie, die Sie sich vorstellen können, um eine Suche über mehrere Hyperparameter zu strukturieren?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
