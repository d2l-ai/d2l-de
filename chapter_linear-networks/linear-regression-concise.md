# Prägnante Umsetzung der linearen Regression
:label:`sec_linear_concise`

Ein breites und intensives Interesse an Deep Learning in den letzten Jahren hat Unternehmen, Wissenschaftler und Hobbyisten dazu inspiriert, eine Vielzahl ausgereifter Open-Source-Frameworks zu entwickeln, um die sich wiederholende Arbeit der Implementierung gradientenbasierter Lernalgorithmen zu automatisieren. In :numref:`sec_linear_scratch` haben wir uns nur auf (i) Tensoren für die Datenspeicherung und lineare Algebra verlassen; und (ii) automatische Differenzierung für die Berechnung von Gradienten. Da Dateniteratoren, Verlustfunktionen, Optimierer und neuronale Netzwerkschichten in der Praxis so häufig sind, implementieren moderne Bibliotheken diese Komponenten auch für uns.

In diesem Abschnitt zeigen wir Ihnen, wie Sie das lineare Regressionsmodell von :numref:`sec_linear_scratch` mithilfe von High-Level-APIs von Deep Learning-Frameworks prägnant implementieren.

## Generieren des Datensatzes

Um zu beginnen, werden wir den gleichen Datensatz wie in :numref:`sec_linear_scratch` generieren.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## Lesen des Datensatzes

Anstatt unseren eigenen Iterator zu rollen, können wir die vorhandene API in einem Framework aufrufen, um Daten zu lesen. Wir übergeben `features` und `labels` als Argumente und geben `batch_size` an, wenn ein Dateniteratorobjekt instanziiert wird. Außerdem gibt der boolesche Wert `is_train` an, ob das Dateniteratorobjekt die Daten auf jeder Epoche mischt (Durchlaufen des Datasets).

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Jetzt können wir `data_iter` in der gleichen Weise verwenden, wie wir die `data_iter`-Funktion in :numref:`sec_linear_scratch` genannt haben. Um zu überprüfen, ob es funktioniert, können wir die erste Minibatch von Beispielen lesen und drucken. Im Vergleich mit :numref:`sec_linear_scratch` verwenden wir hier `iter`, um einen Python-Iterator zu konstruieren und `next` zu verwenden, um das erste Element aus dem Iterator zu erhalten.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## Definieren des Modells

Als wir in :numref:`sec_linear_scratch` lineare Regression von Grund auf neu implementiert haben, haben wir unsere Modellparameter explizit definiert und die Berechnungen codiert, um Ausgaben mit einfachen linearen Algebra-Operationen zu erzeugen. Du sollte* wissen, wie das geht. Aber sobald Ihre Modelle komplexer werden und wenn Sie dies fast jeden Tag tun müssen, werden Sie sich für die Hilfe freuen. Die Situation ist ähnlich wie die Programmierung Ihres eigenen Blogs von Grund auf neu. Es ein- oder zweimal zu tun ist lohnend und lehrreich, aber Sie wären ein lausiger Web-Entwickler, wenn Sie jedes Mal, wenn Sie einen Blog brauchten, Sie einen Monat damit verbracht haben, das Rad neu zu erfinden.

Für Standardoperationen können wir die vordefinierten Layer eines Frameworks verwenden, die es uns ermöglichen, uns besonders auf die Layer zu konzentrieren, die verwendet werden, um das Modell zu konstruieren, anstatt sich auf die Implementierung zu konzentrieren. Wir definieren zunächst eine Modellvariable `net`, die sich auf eine Instanz der `Sequential`-Klasse bezieht. Die Klasse `Sequential` definiert einen Container für mehrere Layer, die miteinander verkettet werden. Bei den Eingabedaten übergibt eine `Sequential`-Instanz sie durch die erste Ebene, die wiederum die Ausgabe als Eingabe der zweiten Ebene übergibt und so weiter. Im folgenden Beispiel besteht unser Modell nur aus einer Schicht, so dass wir `Sequential` nicht wirklich brauchen. Da aber fast alle unsere zukünftigen Modelle mehrere Ebenen umfassen werden, werden wir es trotzdem verwenden, um Sie mit dem gängigsten Workflow vertraut zu machen.

Erinnern Sie sich an die Architektur eines Single-Layer-Netzwerks, wie in :numref:`fig_single_neuron` gezeigt. Der Layer wird als *voll verbunden* bezeichnet, da jeder seiner Eingänge mittels einer Matrix-Vektormultiplikation mit jedem seiner Ausgänge verbunden ist.

:begin_tab:`mxnet`
In Gluon ist die vollständig vernetzte Schicht in der Klasse `Dense` definiert. Da wir nur eine einzelne skalare Ausgabe erzeugen wollen, setzen wir diese Zahl auf 1.

Es ist erwähnenswert, dass Gluon aus Gründen der Bequemlichkeit nicht verlangt, dass wir die Eingabeform für jede Schicht angeben müssen. Hier müssen wir Gluon nicht sagen, wie viele Eingaben in diese lineare Schicht gehen. Wenn wir zum ersten Mal versuchen, Daten durch unser Modell zu übergeben, z. B. wenn wir `net(X)` später ausführen, wird Gluon automatisch die Anzahl der Eingaben für jede Schicht ableiten. Wie das funktioniert, werden wir später näher beschreiben.
:end_tab:

:begin_tab:`pytorch`
In PyTorch ist die vollständig verbundene Schicht in der Klasse `Linear` definiert. Beachten Sie, dass wir zwei Argumente in `nn.Linear` übergeben. Die erste gibt die Eingabe-Feature-Bemaßung an, die 2 ist, und die zweite ist die Ausgabe-Feature-Dimension, bei der es sich um einen einzelnen Skalar und daher um 1 handelt.
:end_tab:

:begin_tab:`tensorflow`
In Keras ist die vollständig verbundene Schicht in der Klasse `Dense` definiert. Da wir nur eine einzelne skalare Ausgabe erzeugen wollen, setzen wir diese Zahl auf 1.

Es ist erwähnenswert, dass Keras aus Gründen der Bequemlichkeit nicht die Eingabeform für jede Ebene angeben muss. Hier müssen wir Keras also nicht sagen, wie viele Eingaben in diese lineare Schicht gehen. Wenn wir zum ersten Mal versuchen, Daten durch unser Modell zu übergeben, z. B. wenn wir `net(X)` später ausführen, wird Keras automatisch die Anzahl der Eingaben für jede Schicht ableiten. Wie das funktioniert, werden wir später näher beschreiben.
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## Initialisieren von Modellparametern

Bevor Sie `net` verwenden, müssen Sie die Modellparameter initialisieren, z. B. die Gewichtungen und die Verzerrung im linearen Regressionsmodell. Deep Learning-Frameworks haben oft eine vordefinierte Möglichkeit, die Parameter zu initialisieren. Hier geben wir an, dass jeder Gewichtsparameter zufällig aus einer Normalverteilung mit dem Mittelwert 0 und der Standardabweichung 0,01 entnommen werden soll. Der Parameter „Bias“ wird auf Null initialisiert.

:begin_tab:`mxnet`
Wir importieren das Modul `initializer` aus MXNet. Dieses Modul bietet verschiedene Methoden zur Initialisierung von Modellparametern. Gluon stellt `init` als Abkürzung (Abkürzung) zur Verfügung, um auf das Paket `initializer` zuzugreifen. Wir geben nur an, wie das Gewicht initialisiert wird, indem Sie `init.Normal(sigma=0.01)` aufrufen. Vorgabeparameter werden standardmäßig auf Null initialisiert.
:end_tab:

:begin_tab:`pytorch`
Wie wir bei der Konstruktion `nn.Linear` die Eingabe- und Ausgabeabmessungen angegeben haben. Jetzt greifen wir direkt auf die Parameter zu, um dort Anfangswerte anzugeben. Zuerst suchen wir den Layer nach `net[0]`, der die erste Schicht im Netzwerk ist, und verwenden dann die Methoden `weight.data` und `bias.data`, um auf die Parameter zuzugreifen. Als nächstes verwenden wir die Ersetzungsmethoden `normal_` und `fill_`, um Parameterwerte zu überschreiben.
:end_tab:

:begin_tab:`tensorflow`
Das Modul `initializers` in TensorFlow bietet verschiedene Methoden zur Initialisierung von Modellparametern. Die einfachste Möglichkeit, die Initialisierungsmethode in Keras anzugeben, ist beim Erstellen des Layers durch Angabe von `kernel_initializer`. Hier erstellen wir `net` wieder.
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
Der obige Code mag unkompliziert aussehen, aber Sie sollten beachten, dass hier etwas Seltsames passiert. Wir initialisieren Parameter für ein Netzwerk, obwohl Gluon noch nicht weiß, wie viele Dimensionen die Eingabe haben wird! Es könnte 2 sein, wie in unserem Beispiel, oder es könnte 2000 sein. Gluon lässt uns damit davonkommen, denn hinter der Szene ist die Initialisierung tatsächlich *verzögert*. Die eigentliche Initialisierung findet nur statt, wenn wir zum ersten Mal versuchen, Daten über das Netzwerk zu übergeben. Achten Sie darauf, dass wir, da die Parameter noch nicht initialisiert wurden, nicht auf sie zugreifen oder manipulieren können.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
Der obige Code mag unkompliziert aussehen, aber Sie sollten beachten, dass hier etwas Seltsames passiert. Wir initialisieren Parameter für ein Netzwerk, obwohl Keras noch nicht weiß, wie viele Dimensionen die Eingabe haben wird! Es könnte 2 sein, wie in unserem Beispiel, oder es könnte 2000 sein. Keras lässt uns damit davonkommen, denn hinter den Kulissen ist die Initialisierung tatsächlich *verzögert*. Die eigentliche Initialisierung findet nur statt, wenn wir zum ersten Mal versuchen, Daten über das Netzwerk zu übergeben. Achten Sie darauf, dass wir, da die Parameter noch nicht initialisiert wurden, nicht auf sie zugreifen oder manipulieren können.
:end_tab:

## Definieren der Verlustfunktion

:begin_tab:`mxnet`
In Gluon definiert das Modul `loss` verschiedene Verlustfunktionen. In diesem Beispiel werden wir die Gluon-Implementierung von quadratischem Verlust (`L2Loss`) verwenden.
:end_tab:

:begin_tab:`pytorch`
Die Klasse `MSELoss` berechnet den mittleren quadrierten Fehler, auch bekannt als quadrierte $L_2$ Norm. Standardmäßig gibt es den durchschnittlichen Verlust über Beispiele zurück.
:end_tab:

:begin_tab:`tensorflow`
Die Klasse `MeanSquaredError` berechnet den mittleren quadrierten Fehler, auch bekannt als quadrierte $L_2$ Norm. Standardmäßig gibt es den durchschnittlichen Verlust über Beispiele zurück.
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## Definieren des Optimierungsalgorithmus

:begin_tab:`mxnet`
Minibatch stochastischer Gradientenabstieg ist ein Standardwerkzeug zur Optimierung neuronaler Netzwerke und somit unterstützt Gluon es neben einer Reihe von Variationen dieses Algorithmus durch seine Klasse `Trainer`. Wenn wir `Trainer` instanziieren, werden wir die zu optimierenden Parameter angeben (erhältlich von unserem Modell `net` über `net.collect_params()`), den Optimierungsalgorithmus, den wir verwenden möchten (`sgd`) und ein Wörterbuch mit Hyperparametern, die von unserem Optimierungsalgorithmus benötigt werden. Minibatch stochastischen Gradientenabstieg erfordert nur, dass wir den Wert `learning_rate` setzen, der hier auf 0,03 gesetzt ist.
:end_tab:

:begin_tab:`pytorch`
Minibatch stochastischer Gradientenabstieg ist ein Standardwerkzeug zur Optimierung neuronaler Netze und unterstützt damit PyTorch es neben einer Reihe von Variationen dieses Algorithmus im Modul `optim`. Wenn wir eine `SGD`-Instanz instanziieren, werden wir die Parameter angeben, die optimiert werden sollen (erreichbar von unserem Netz über `net.parameters()`), mit einem Wörterbuch von Hyperparametern, die von unserem Optimierungsalgorithmus benötigt werden. Minibatch stochastischen Gradientenabstieg erfordert nur, dass wir den Wert `lr` setzen, der hier auf 0,03 gesetzt ist.
:end_tab:

:begin_tab:`tensorflow`
Minibatch stochastischer Gradientenabstieg ist ein Standardwerkzeug zur Optimierung neuronaler Netze und somit unterstützt Keras es neben einer Reihe von Variationen dieses Algorithmus im Modul `optimizers`. Minibatch stochastischen Gradientenabstieg erfordert nur, dass wir den Wert `learning_rate` setzen, der hier auf 0,03 gesetzt ist.
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## Ausbildung

Sie haben vielleicht bemerkt, dass das Ausdrücken unseres Modells durch High-Level-APIs eines Deep Learning-Frameworks vergleichsweise wenige Codezeilen erfordert. Wir mussten keine Parameter einzeln zuweisen, unsere Verlustfunktion definieren oder minibatch stochastischen Gradientenabstieg implementieren. Sobald wir beginnen, mit viel komplexeren Modellen zu arbeiten, werden die Vorteile von High-Level-APIs erheblich zunehmen. Sobald wir jedoch alle Grundstücke an Ort und Stelle haben, ist die Trainingsschleife selbst auffallend ähnlich dem, was wir getan haben, wenn wir alles von Grund auf neu umsetzen.

Um Ihr Gedächtnis zu aktualisieren: Für eine gewisse Anzahl von Epochen werden wir einen vollständigen Durchgang über den Datensatz (`train_data`) machen und iterativ eine Minibatch von Eingaben und die entsprechenden Boden-Wahrheitsbeschriftungen abrufen. Für jeden Minibatch durchlaufen wir folgendes Ritual:

* Generieren Sie Vorhersagen, indem Sie `net(X)` aufrufen und berechnen Sie den Verlust `l` (die Forward Propagation).
* Berechnen Sie Verläufe, indem Sie die Backpropagation ausführen.
* Aktualisieren Sie die Modellparameter, indem Sie unseren Optimierer aufrufen.

Für eine gute Maßnahme berechnen wir den Verlust nach jeder Epoche und drucken ihn, um den Fortschritt zu überwachen.

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

Im Folgenden vergleichen wir die Modellparameter, die durch Schulungen zu endlichen Daten gelernt wurden, mit den tatsächlichen Parametern, die unseren Datensatz generiert haben. Um auf Parameter zuzugreifen, greifen wir zuerst auf den Layer zu, den wir von `net` benötigen, und greifen dann auf die Gewichtungen und Verzerrungen dieser Ebene zu. Wie bei unserer von-scratch Implementierung, beachten Sie, dass unsere geschätzten Parameter nahe an ihren Grund-Wahrheits-Gegenstücken liegen.

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## Zusammenfassung

:begin_tab:`mxnet`
* Mit Gluon können wir Modelle viel prägnanter implementieren.
* In Gluon bietet das Modul `data` Werkzeuge für die Datenverarbeitung, das Modul `nn` definiert eine große Anzahl neuronaler Netzwerkschichten und das Modul `loss` definiert viele gemeinsame Verlustfunktionen.
* Das Modul `initializer` von MXNet bietet verschiedene Methoden zur Initialisierung von Modellparametern.
* Dimensionalität und Speicherung werden automatisch abgeleitet. Achten Sie jedoch darauf, dass Sie nicht versuchen, auf Parameter zuzugreifen, bevor sie initialisiert wurden.
:end_tab:

:begin_tab:`pytorch`
* Mit den High-Level-APIs von PyTorch können wir Modelle viel prägnanter implementieren.
* In PyTorch bietet das Modul `data` Werkzeuge für die Datenverarbeitung, das Modul `nn` definiert eine große Anzahl von neuronalen Netzwerkschichten und gemeinsamen Verlustfunktionen.
* Wir können die Parameter initialisieren, indem sie ihre Werte durch Methoden ersetzen, die mit `_` enden.
:end_tab:

:begin_tab:`tensorflow`
* Mit den High-Level-APIs von TensorFlow können wir Modelle viel prägnanter implementieren.
* In TensorFlow bietet das Modul `data` Werkzeuge für die Datenverarbeitung, das Modul `keras` definiert eine große Anzahl von neuronalen Netzwerkschichten und gemeinsamen Verlustfunktionen.
* Das Modul `initializers` von TensorFlow bietet verschiedene Methoden zur Initialisierung von Modellparametern.
* Dimensionalität und Speicherung werden automatisch abgeleitet (achten Sie jedoch darauf, dass Sie nicht versuchen, auf Parameter zuzugreifen, bevor sie initialisiert wurden).
:end_tab:

## Übungen

:begin_tab:`mxnet`
1. Wenn wir `l = loss(output, y)` durch `l = loss(output, y).mean()` ersetzen, müssen wir `trainer.step(batch_size)` auf `trainer.step(1)` ändern, damit der Code sich identisch verhält. Warum?
1. In der MXNet-Dokumentation erfahren Sie, welche Verlustfunktionen und Initialisierungsmethoden in den Modulen `gluon.loss` und `init` bereitgestellt werden. Ersetzen Sie den Verlust durch Hubers Verlust.
1. Wie greifen Sie auf den Farbverlauf von `dense.weight` zu?

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. Wenn wir `nn.MSELoss(reduction='sum')` durch `nn.MSELoss()` ersetzen, wie können wir die Lernrate für den Code ändern, um sich identisch zu verhalten. Warum?
1. Lesen Sie die PyTorch-Dokumentation, um zu sehen, welche Verlustfunktionen und Initialisierungsmethoden zur Verfügung gestellt werden. Ersetzen Sie den Verlust durch Hubers Verlust.
1. Wie greifen Sie auf den Farbverlauf von `net[0].weight` zu?

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. In der TensorFlow-Dokumentation erfahren Sie, welche Verlustfunktionen und Initialisierungsmethoden bereitgestellt werden. Ersetzen Sie den Verlust durch Hubers Verlust.

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
