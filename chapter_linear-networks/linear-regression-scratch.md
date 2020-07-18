# Lineare Regressionsimplementierung von Grund auf
:label:`sec_linear_scratch`

Jetzt, da Sie die Schlüsselideen hinter der linearen Regression verstehen, können wir beginnen, durch eine praktische Implementierung im Code zu arbeiten. In diesem Abschnitt werden wir die gesamte Methode von Grund auf neu implementieren, einschließlich der Datenpipeline, dem Modell, der Verlustfunktion und dem Minibatch-stochastischen Gradientenabstiegs-Optimierer. Während moderne Deep Learning-Frameworks fast all diese Arbeit automatisieren können, ist die Implementierung von Dingen der einzige Weg, um sicherzustellen, dass Sie wirklich wissen, was Sie tun. Wenn es an der Zeit ist, Modelle anzupassen, unsere eigenen Schichten oder Verlustfunktionen zu definieren, wird sich außerdem als nützlich erweisen, wie die Dinge unter der Haube funktionieren. In diesem Abschnitt werden wir uns nur auf Tensoren und automatische Differenzierung verlassen. Anschließend werden wir eine prägnantere Implementierung einführen und dabei Schnickschnack von Deep Learning-Frameworks nutzen.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Generieren des Datensatzes

Um die Dinge einfach zu halten, erstellen wir einen künstlichen Datensatz nach einem linearen Modell mit additivem Rauschen. Unsere Aufgabe wird es sein, die Parameter dieses Modells mithilfe der endlichen Reihe von Beispielen, die in unserem Datensatz enthalten sind, wiederherzustellen. Wir halten die Daten niedrig dimensioniert, damit wir sie leicht visualisieren können. Im folgenden Codeausschnitt generieren wir ein Dataset mit 1000 Beispielen, die jeweils aus zwei Features bestehen, die aus einer Standardnormalverteilung abgetastet wurden. Somit wird unser synthetischer Datensatz eine Matrix $\mathbf{X}\in \mathbb{R}^{1000 \times 2}$ sein.

Die wahren Parameter, die unseren Datensatz erzeugen, werden $\mathbf{w} = [2, -3.4]^\top$ und $b = 4.2$ sein, und unsere synthetischen Beschriftungen werden gemäß dem folgenden linearen Modell mit dem Rauschbegriff $\epsilon$ zugewiesen:

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$

Sie könnten sich $\epsilon$ vorstellen, um potenzielle Messfehler auf den Features und Beschriftungen zu erfassen. Wir gehen davon aus, dass die Standardannahmen halten und damit $\epsilon$ einer Normalverteilung mit dem Mittelwert 0 gehorcht. Um unser Problem einfach zu machen, setzen wir seine Standardabweichung auf 0,01. Der folgende Code generiert unseren synthetischen Datensatz.

```{.python .input}
#@tab mxnet, pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

Beachten Sie, dass jede Zeile in `features` aus einem zweidimensionalen Datenpunkt besteht und dass jede Zeile in `labels` aus einem eindimensionalen Beschriftungswert (einem Skalar) besteht.

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

Durch das Generieren eines Streudiagramms mit dem zweiten Feature `features[:, 1]` und `labels` können wir die lineare Korrelation zwischen den beiden deutlich beobachten.

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## Lesen des Datensatzes

Daran erinnern, dass Schulungsmodelle darin bestehen, mehrere Durchläufe über das Dataset zu erstellen, jeweils eine Minibatch von Beispielen zu erstellen und sie zum Aktualisieren unseres Modells zu verwenden. Da dieser Prozess so grundlegend für das Training von Algorithmen für maschinelles Lernen ist, lohnt es sich, eine Hilfsfunktion zu definieren, um das Dataset zu mischen und in Minibatches darauf zuzugreifen.

Im folgenden Code definieren wir die Funktion `data_iter`, um eine mögliche Implementierung dieser Funktionalität zu demonstrieren. Die Funktion nimmt eine Chargengröße, eine Matrix von Features und einen Vektor von Beschriftungen, was Minibatches der Größe `batch_size` ergibt. Jede Minibatch besteht aus einem Tupel von Features und Beschriftungen.

```{.python .input}
#@tab mxnet, pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```

Beachten Sie im Allgemeinen, dass wir Minibatches in vernünftiger Größe verwenden möchten, um die GPU-Hardware zu nutzen, die sich bei Parallelisierungsoperationen auszeichnet. Da jedes Beispiel parallel durch unsere Modelle gespeist werden kann und der Gradient der Verlustfunktion für jedes Beispiel auch parallel genommen werden kann, ermöglichen GPUs es uns, Hunderte von Beispielen in kaum mehr Zeit zu verarbeiten, als nur ein einziges Beispiel benötigt wird.

Um einige Intuition zu bauen, lassen Sie uns lesen und drucken Sie die erste kleine Charge von Datenbeispielen. Die Form der Features in jedem Minibatch gibt uns sowohl die Minibatch-Größe als auch die Anzahl der Eingabe-Features an. Ebenso wird unser Minibatch von Etiketten eine Form haben, die von `batch_size` gegeben wird.

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

Während wir die Iteration ausführen, erhalten wir verschiedene Minibatches nacheinander, bis der gesamte Datensatz erschöpft ist (versuchen Sie dies). Während die oben implementierte Iteration für didaktische Zwecke gut ist, ist sie ineffizient in einer Weise, die uns in Schwierigkeiten mit echten Problemen bringen könnte. Zum Beispiel erfordert es, dass wir alle Daten in den Speicher laden und dass wir viele zufällige Speicherzugriffe ausführen. Die in einem Deep Learning-Framework implementierten integrierten Iteratoren sind wesentlich effizienter und können sowohl in Dateien gespeicherte Daten als auch über Datenströme eingespeist werden.

## Initialisieren von Modellparametern

Bevor wir mit der Optimierung der Parameter unseres Modells durch Minibatch stochastischen Gradientenabstieg beginnen können, müssen wir an erster Stelle einige Parameter haben. Im folgenden Code initialisieren wir Gewichtungen, indem wir Zufallszahlen aus einer Normalverteilung mit dem Mittelwert 0 und einer Standardabweichung von 0,01 abtasten und die Verzerrung auf 0 setzen.

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```

Nach der Initialisierung unserer Parameter besteht unsere nächste Aufgabe darin, sie zu aktualisieren, bis sie unsere Daten ausreichend gut passen. Jedes Update erfordert die Gradient unserer Verlustfunktion in Bezug auf die Parameter. Angesichts dieses Gradienten können wir jeden Parameter in die Richtung aktualisieren, die den Verlust reduzieren kann.

Da niemand Gradienten explizit berechnen möchte (dies ist mühsam und fehleranfällig), verwenden wir die automatische Differenzierung, wie in :numref:`sec_autograd` eingeführt, um den Gradienten zu berechnen.

## Definieren des Modells

Als nächstes müssen wir unser Modell definieren und seine Eingänge und Parameter mit seinen Ausgängen in Beziehung setzen. Daran erinnern, dass wir zur Berechnung der Ausgabe des linearen Modells einfach das Matrix-Vektorpunkt-Produkt der Eingabe-Features $\mathbf{X}$ und die Modellgewichte $\mathbf{w}$ nehmen und jedem Beispiel den Offset $b$ hinzufügen. Beachten Sie, dass unter $\mathbf{Xw}$ ein Vektor und $b$ ein Skalar ist. Erinnern Sie sich an den Rundfunkmechanismus, wie in :numref:`subsec_broadcasting` beschrieben. Wenn wir einen Vektor und einen Skalar hinzufügen, wird der Skalar zu jeder Komponente des Vektors hinzugefügt.

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## Definieren der Verlustfunktion

Da die Aktualisierung unseres Modells den Gradienten unserer Verlustfunktion erfordert, sollten wir zuerst die Verlustfunktion definieren. Hier werden wir die quadrierte Verlustfunktion verwenden, wie in :numref:`sec_linear_regression` beschrieben. In der Implementierung müssen wir den wahren Wert `y` in die Form des vorhergesagten Wertes `y_hat` umwandeln. Das Ergebnis, das von der folgenden Funktion zurückgegeben wird, hat auch die gleiche Form wie `y_hat`.

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## Definieren des Optimierungsalgorithmus

Wie wir in :numref:`sec_linear_regression` besprochen haben, hat die lineare Regression eine geschlossene Lösung. Dies ist jedoch kein Buch über lineare Regression: Es ist ein Buch über Deep Learning. Da keines der anderen Modelle, die in diesem Buch vorgestellt werden, analytisch gelöst werden kann, werden wir diese Gelegenheit nutzen, um Ihr erstes Arbeitsbeispiel der Minibatch-stochastischen Gradientenabstieg vorzustellen.

Bei jedem Schritt werden wir mit einem Minibatch, das zufällig aus unserem Datensatz gezogen wird, den Gradienten des Verlustes in Bezug auf unsere Parameter schätzen. Als nächstes werden wir unsere Parameter in die Richtung aktualisieren, die den Verlust reduzieren kann. Der folgende Code wendet die Minibatch-Aktualisierung des stochastischen Gradientenabstiegs unter Berücksichtigung einer Reihe von Parametern, einer Lernrate und einer Chargengröße an. Die Größe des Aktualisierungsschritts wird durch die Lernrate `lr` bestimmt. Da unser Verlust als Summe über die Minibatch von Beispielen berechnet wird, normalisieren wir unsere Schrittgröße um die Chargengröße (`batch_size`), so dass die Größe einer typischen Schrittgröße nicht stark von unserer Wahl der Chargengröße abhängt.

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param.data.sub_(lr*param.grad/batch_size)
        param.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

## Ausbildung

Jetzt, da wir alle Teile an Ort und Stelle haben, sind wir bereit, die Haupttrainingsschleife zu implementieren. Es ist entscheidend, dass Sie diesen Code verstehen, da Sie während Ihrer Karriere im Deep Learning immer wieder nahezu identische Trainingsschleifen sehen werden.

In jeder Iteration werden wir eine Minibatch von Trainingsbeispielen greifen und sie durch unser Modell weiterleiten, um eine Reihe von Vorhersagen zu erhalten. Nach der Berechnung des Verlustes initiieren wir den Rückwärtslauf durch das Netzwerk und speichern die Gradienten in Bezug auf jeden Parameter. Schließlich werden wir den Optimierungsalgorithmus `sgd` aufrufen, um die Modellparameter zu aktualisieren.

Zusammenfassend werden wir die folgende Schleife ausführen:

* Parameter initialisieren $(\mathbf{w}, b)$
* Wiederholen, bis Sie fertig sind
    * Gradienten berechnen $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Update-Parameter $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

In jeder *Epoch* durchlaufen wir den gesamten Datensatz (mit der Funktion `data_iter`), sobald wir jedes Beispiel im Trainingsdatensatz durchlaufen (vorausgesetzt, die Anzahl der Beispiele ist durch die Stapelgröße teilbar). Die Anzahl der Epochen `num_epochs` und die Lernrate `lr` sind beide Hyperparameter, die wir hier auf 3 bzw. 0,03 setzen. Leider ist das Setzen von Hyperparametern schwierig und erfordert einige Anpassungen durch Versuch und Irrtum. Wir entziehen diese Details vorerst, überarbeiten sie aber später in :numref:`chap_optimization`.

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

In diesem Fall, weil wir den Datensatz selbst synthetisiert haben, wissen wir genau, was die wahren Parameter sind. So können wir unseren Erfolg im Training bewerten, indem wir die wahren Parameter mit denen vergleichen, die wir durch unsere Trainingsschleife gelernt haben. In der Tat erweisen sie sich als sehr nahe beieinander.

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

Beachten Sie, dass wir es nicht als selbstverständlich betrachten sollten, dass wir in der Lage sind, die Parameter perfekt wiederherzustellen. Im maschinellen Lernen beschäftigen wir uns jedoch in der Regel weniger mit der Wiederherstellung wahrer zugrunde liegender Parameter und mehr mit Parametern, die zu einer hochgenauen Vorhersage führen. Glücklicherweise kann der stochastische Gradientenabstieg selbst bei schwierigen Optimierungsproblemen oft bemerkenswert gute Lösungen finden, teilweise aufgrund der Tatsache, dass es für tiefe Netzwerke viele Konfigurationen der Parameter gibt, die zu einer hochgenauen Vorhersage führen.

## Zusammenfassung

* Wir haben gesehen, wie ein tiefes Netzwerk von Grund auf neu implementiert und optimiert werden kann, indem nur Tensoren und automatische Differenzierung verwendet werden, ohne dass Ebenen oder ausgefallene Optimierer definiert werden müssen.
* Dieser Abschnitt kratzt nur die Oberfläche dessen, was möglich ist. In den folgenden Abschnitten werden wir zusätzliche Modelle basierend auf den Konzepten, die wir gerade eingeführt haben, beschreiben und lernen, wie man sie prägnanter umsetzt.

## Übungen

1. Was würde passieren, wenn wir die Gewichte auf Null initialisieren würden. Würde der Algorithmus noch funktionieren?
1. Angenommen Sie, Sie sind [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) versuchen, ein Modell zwischen Spannung und Strom zu finden. Können Sie die automatische Differenzierung verwenden, um die Parameter Ihres Modells zu lernen?
1. Können Sie [Planck's Law](https://en.wikipedia.org/wiki/Planck%27s_law) verwenden, um die Temperatur eines Objekts mittels spektraler Energiedichte zu bestimmen?
1. Welche Probleme könnten auftreten, wenn Sie die zweiten Derivate berechnen möchten? Wie würdest du sie reparieren?
1.  Warum wird die Funktion `reshape` in der Funktion `squared_loss` benötigt?
1. Experimentieren Sie mit unterschiedlichen Lernraten, um herauszufinden, wie schnell der Wert der Verlustfunktion sinkt.
1. Wenn die Anzahl der Beispiele nicht durch die Stapelgröße geteilt werden kann, was passiert mit dem Verhalten der Funktion `data_iter`?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
