# Prägnante Implementierung von mehrschichtigen Perzeptrons
:label:`sec_mlp_concise`

Wie Sie vielleicht erwarten, können wir mithilfe der High-Level-APIs MLPs noch prägnanter implementieren.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
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

## Modell

Im Vergleich zu unserer prägnanten Implementierung der softmax-Regressionsimplementierung (:numref:`sec_softmax_concise`) besteht der einzige Unterschied darin, dass wir
*zwei* voll vernetzte Schichten
(zuvor haben wir *eine* hinzugefügt). Die erste ist unsere versteckte Schicht, die 256 versteckte Einheiten enthält und die RelU-Aktivierungsfunktion anwendet. Die zweite ist unsere Ausgabeschicht.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

Die Trainingsschleife ist genau die gleiche wie bei der Implementierung der softmax-Regression. Diese Modularität ermöglicht es uns, Fragen der Modellarchitektur von orthogonalen Überlegungen zu trennen.

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## Zusammenfassung

* Mit High-Level-APIs können wir MLPs viel prägnanter implementieren.
* Für das gleiche Klassifizierungsproblem ist die Implementierung eines MLP die gleiche wie die der Softmax-Regression mit Ausnahme zusätzlicher versteckter Layer mit Aktivierungsfunktionen.

## Übungen

1. Versuchen Sie, verschiedene Anzahl ausgeblendeter Ebenen hinzuzufügen (Sie können auch die Lernrate ändern). Welche Einstellung funktioniert am besten?
1. Probieren Sie verschiedene Aktivierungsfunktionen aus. Welcher funktioniert am besten?
1. Probieren Sie verschiedene Schemata aus, um die Gewichtungen zu initialisieren. Welche Methode funktioniert am besten?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/262)
:end_tab:
