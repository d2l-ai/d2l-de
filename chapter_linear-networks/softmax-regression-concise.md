# Prägnante Implementierung der Softmax-Regression
:label:`sec_softmax_concise`

Genauso wie High-Level-APIs von Deep Learning-Frameworks es viel einfacher gemacht haben, lineare Regression in :numref:`sec_linear_concise` zu implementieren, werden wir es für die Implementierung von Klassifizierungsmodellen ähnlich (oder möglicherweise mehr) finden. Bleiben wir mit dem Fashion-MNIST-Datensatz und halten Sie die Chargengröße bei 256 wie in :numref:`sec_softmax_scratch`.

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

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initialisieren von Modellparametern

Wie in :numref:`sec_softmax` erwähnt, ist die Ausgabeschicht der softmax-Regression eine vollständig verbundene Schicht. Um unser Modell zu implementieren, müssen wir daher nur eine vollständig verbundene Schicht mit 10 Ausgängen zu unserem `Sequential` hinzufügen. Auch hier ist der `Sequential` nicht wirklich notwendig, aber wir könnten genauso gut die Gewohnheit bilden, da er bei der Implementierung tiefer Modelle allgegenwärtig sein wird. Wieder initialisieren wir die Gewichte zufällig mit Null Mittelwert und Standardabweichung 0,01.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch does not implicitly reshape the inputs. Thus we define a layer to
# reshape the inputs in our network
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,784)

net = nn.Sequential(Reshape(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## Softmax-Implementierung neu besucht
:label:`subsec_softmax-implementation-revisited`

Im vorherigen Beispiel von :numref:`sec_softmax_scratch` haben wir die Ausgabe unseres Modells berechnet und diese Ausgabe dann durch den Kreuzentropieverlust ausgeführt. Mathematisch ist das eine absolut vernünftige Sache zu tun. Aus rechnerischer Sicht kann die Exponentiierung jedoch eine Quelle für numerische Stabilitätsprobleme sein.

Daran erinnern, dass die softmax-Funktion $\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$ berechnet, wobei $\hat y_j$ das $j^\mathrm{th}$ Element der prognostizierten Wahrscheinlichkeitsverteilung $\hat{\mathbf{y}}$ und $o_j$ das $j^\mathrm{th}$ Element der Logits $\mathbf{o}$ ist. Wenn einige der $o_k$ sehr groß sind (dh sehr positiv), dann ist $\exp(o_k)$ möglicherweise größer als die größte Anzahl, die wir für bestimmte Datentypen haben können (dh *Überlauf*). Dies würde den Nenner (und/oder Zähler) `inf` (unendlich) machen und wir landen entweder 0, `inf`, oder `nan` (keine Zahl) für $\hat y_j$. In diesen Situationen erhalten wir keinen genau definierten Rückgabewert für die Kreuzentropie.

Ein Trick, um dies zu umgehen, besteht darin, zuerst $\max(o_k)$ von allen $o_k$ zu subtrahieren, bevor Sie mit der Softmax-Berechnung fortfahren. Sie können überprüfen, ob diese Verschiebung von jedem $o_k$ um konstanten Faktor den Rückgabewert von softmax nicht ändert. Nach dem Subtraktions- und Normalisierungsschritt kann es möglich sein, dass einige $o_j$ große negative Werte haben und somit der entsprechende $\exp(o_j)$ Werte annimmt, die nahe Null liegen. Diese könnten aufgrund endlicher Präzision auf Null gerundet werden (d. h., *underflow*), was $\hat y_j$ Null macht und uns `-inf` für $\log(\hat y_j)$ geben. Ein paar Schritte den Weg hinunter in der Rückverbreitung, könnten wir uns mit einem Screenful der gefürchteten `nan` Ergebnisse konfrontiert sehen.

Glücklicherweise werden wir durch die Tatsache gerettet, dass wir, obwohl wir exponentielle Funktionen berechnen, letztendlich beabsichtigen, ihr Protokoll zu nehmen (bei der Berechnung des Kreuzentropieverlustes). Durch die Kombination dieser beiden Operatoren softmax und Cross-Entropie können wir den numerischen Stabilitätsproblemen entkommen, die uns sonst während der Backpropagation plagen könnten. Wie in der folgenden Gleichung gezeigt, vermeiden wir die Berechnung $\exp(o_j)$ und können stattdessen $o_j$ direkt aufgrund der Stornierung in $\log(\exp(\cdot))$ verwenden.

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j)}{\sum_k \exp(o_k)}\right) \\
& = \log{(\exp(o_j))}-\log{\left( \sum_k \exp(o_k) \right)} \\
& = o_j -\log{\left( \sum_k \exp(o_k) \right)}.
\end{aligned}
$$

Wir wollen die herkömmliche softmax-Funktion griffbereit halten, falls wir jemals die Ausgabewahrscheinlichkeiten durch unser Modell auswerten wollen. Aber anstatt softmax-Wahrscheinlichkeiten in unsere neue Verlustfunktion zu übergeben, werden wir einfach die Logits übergeben und den softmax und sein Protokoll auf einmal innerhalb der Kreuzentropie-Verlustfunktion berechnen, die intelligente Dinge wie den ["LogsumExp trick"](https://en.wikipedia.org/wiki/LogSumExp) tut.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## Optimierungs-Algorithmus

Hier verwenden wir den Minibatch-stochastischen Gradientenabstieg mit einer Lernrate von 0,1 als Optimierungsalgorithmus. Beachten Sie, dass dies die gleiche ist wie im Beispiel für lineare Regression angewendet wird, und es veranschaulicht die allgemeine Anwendbarkeit der Optimierer.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## Ausbildung

Als nächstes rufen wir die in :numref:`sec_softmax_scratch` definierte Trainingsfunktion auf, um das Modell zu trainieren.

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

Wie zuvor konvergiert dieser Algorithmus zu einer Lösung, die eine anständige Genauigkeit erreicht, wenn auch diesmal weniger Codezeilen als zuvor.

## Zusammenfassung

* Mit High-Level-APIs können wir softmax-Regression viel prägnanter implementieren.
* Aus rechnerischer Sicht hat die Implementierung von Softmax-Regression Feinheiten. Beachten Sie, dass in vielen Fällen ein Deep Learning-Framework zusätzliche Vorsichtsmaßnahmen trifft, die über diese bekanntesten Tricks hinausgehen, um numerische Stabilität zu gewährleisten, was uns vor noch mehr Fallstricken bewahrt, die wir treffen würden, wenn wir versuchen, alle unsere Modelle in der Praxis von Grund auf neu zu codieren.

## Übungen

1. Versuchen Sie, die Hyperparameter wie Stapelgröße, Anzahl der Epochen und Lernrate anzupassen, um zu sehen, wie die Ergebnisse sind.
1. Erhöhen Sie die Anzahl der Epochen für das Training. Warum kann die Prüfgenauigkeit nach einiger Zeit abnehmen? Wie konnten wir das in Ordnung bringen?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
