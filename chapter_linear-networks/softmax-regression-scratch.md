# Implementierung von Softmax-Regression von Grund auf
:label:`sec_softmax_scratch`

Genau wie wir lineare Regression von Grund auf neu implementiert haben, glauben wir, dass die softmax-Regression ähnlich grundlegend ist und Sie sollten die blöden Details wissen, wie Sie sie selbst implementieren können. Wir werden mit dem Fashion-MNIST-Datensatz arbeiten, der gerade in :numref:`sec_fashion_mnist` eingeführt wurde und einen Dateniterator mit Batchgröße 256 einrichten.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Initialisieren von Modellparametern

Wie in unserem Beispiel für lineare Regression wird jedes Beispiel hier durch einen Vektor mit fester Länge dargestellt. Jedes Beispiel im Raw-Dataset ist ein $28 \times 28$-Bild. In diesem Abschnitt werden wir jedes Bild glätten und sie als Vektoren der Länge 784 behandeln. In Zukunft werden wir über ausgefeiltere Strategien sprechen, um die räumliche Struktur in Bildern zu nutzen, aber vorerst behandeln wir jede Pixelposition als ein weiteres Feature.

Daran erinnern, dass wir in der softmax-Regression so viele Ausgänge haben, wie es Klassen gibt. Da unser Dataset 10 Klassen hat, hat unser Netzwerk eine Ausgabedimension von 10. Folglich bilden unsere Gewichte eine $784 \times 10$-Matrix und die Verzerrungen bilden einen $1 \times 10$-Zeilenvektor. Wie bei der linearen Regression werden wir unsere Gewichte `W` mit Gaußschen Rauschen und unsere Vorspannung initialisieren, um den Anfangswert 0 zu nehmen.

```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## Softmax-Operation definieren

Bevor wir das softmax-Regressionsmodell implementieren, lassen Sie uns kurz prüfen, wie der Summenoperator entlang bestimmter Dimensionen in einem Tensor arbeitet, wie in :numref:`subseq_lin-alg-reduction` und :numref:`subseq_lin-alg-non-reduction` diskutiert. Bei einer Matrix `X` können wir über alle Elemente (standardmäßig) oder nur über Elemente in der gleichen Achse summieren, d.h. die gleiche Spalte (Achse 0) oder die gleiche Zeile (Achse 1). Beachten Sie, dass, wenn `X` ein Tensor mit Form (2, 3) ist und wir die Spalten summieren, das Ergebnis ein Vektor mit Form (3,) sein wird. Wenn Sie den Summenoperator aufrufen, können wir angeben, dass die Anzahl der Achsen im ursprünglichen Tensor beibehalten wird, anstatt die Dimension, die wir summiert haben, zu reduzieren. Dies führt zu einem zweidimensionalen Tensor mit Form (1, 3).

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Wir sind nun bereit, den softmax-Betrieb zu implementieren. Daran erinnern, dass softmax aus drei Schritten besteht: i) wir exponentiieren jeden Term (mit `exp`); ii) wir summieren über jede Zeile (wir haben eine Zeile pro Beispiel im Batch), um die Normalisierungskonstante für jedes Beispiel zu erhalten; iii) wir teilen jede Zeile durch ihre Normalisierungskonstante, um sicherzustellen, dass das Ergebnis auf 1 summiert. Bevor wir uns den Code ansehen, lassen Sie uns daran erinnern, wie dies als Gleichung ausgedrückt aussieht:

$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$

Der Nenner, oder Normalisierungskonstante, wird manchmal auch *Partitionsfunktion* genannt (und sein Logarithmus wird als log-partition Funktion bezeichnet). Die Ursprünge dieses Namens liegen in [Statistische Physik](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)), wo eine verwandte Gleichung die Verteilung über ein Ensemble von Teilchen modelliert.

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

Wie Sie sehen können, verwandeln wir für jede zufällige Eingabe jedes Element in eine nicht-negative Zahl. Darüber hinaus summiert jede Zeile bis zu 1, wie es für eine Wahrscheinlichkeit erforderlich ist.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

Beachten Sie, dass dies mathematisch korrekt aussieht, wir waren zwar ein bisschen schlampig in unserer Implementierung, da wir aufgrund großer oder sehr kleiner Elemente der Matrix keine Vorsichtsmaßnahmen gegen numerischen Überlauf oder Unterlauf getroffen haben.

## Definieren des Modells

Jetzt, da wir die softmax-Operation definiert haben, können wir das softmax-Regressionsmodell implementieren. Der folgende Code definiert, wie die Eingabe auf die Ausgabe über das Netzwerk zugeordnet wird. Beachten Sie, dass wir jedes Originalbild im Batch mithilfe der Funktion `reshape` in einen Vektor abflachen, bevor die Daten durch unser Modell übergeben werden.

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## Definieren der Verlustfunktion

Als nächstes müssen wir die Cross-Entropie-Verlustfunktion implementieren, wie in :numref:`sec_softmax` eingeführt. Dies kann die häufigste Verlustfunktion in allen Deep Learning sein, da Klassifizierungsprobleme derzeit weit überwiegen Regressionsprobleme.

Daran erinnern, dass die Kreuzentropie die negative Log-Likelihood der vorhergesagten Wahrscheinlichkeit annimmt, die dem wahren Label zugewiesen ist. Anstatt die Vorhersagen mit einer Python for-Schleife zu iterieren (die tendenziell ineffizient ist), können wir alle Elemente mit einem einzigen Operator auswählen. Im Folgenden erstellen wir eine Spielzeugdaten `y_hat` mit 2 Beispielen für prognostizierte Wahrscheinlichkeiten über 3 Klassen. Dann wählen wir die Wahrscheinlichkeit der ersten Klasse im ersten Beispiel und die Wahrscheinlichkeit der dritten Klasse im zweiten Beispiel.

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

Jetzt können wir die Cross-Entropie-Verlustfunktion effizient mit nur einer Codezeile implementieren.

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## Klassifizierungsgenauigkeit

Angesichts der prognostizierten Wahrscheinlichkeitsverteilung `y_hat` wählen wir in der Regel die Klasse mit der höchsten prognostizierten Wahrscheinlichkeit, wenn wir eine harte Vorhersage ausgeben müssen. In der Tat erfordern viele Anwendungen, dass wir eine Wahl treffen. Gmail muss eine E-Mail in „Primary“, „Social“, „Updates“ oder „Foren“ kategorisieren. Es könnte Wahrscheinlichkeiten intern schätzen, aber am Ende des Tages muss es eine unter den Klassen wählen.

Wenn Prognosen mit der Labelklasse `y` übereinstimmen, sind sie korrekt. Die Klassifizierungsgenauigkeit ist der Bruchteil aller korrekten Vorhersagen. Obwohl es schwierig sein kann, Genauigkeit direkt zu optimieren (es ist nicht differenzierbar), ist es oft die Leistungsmaßnahme, die uns am meisten wichtig ist, und wir werden es fast immer beim Training von Klassifikatoren melden.

Um die Genauigkeit zu berechnen, tun wir folgendes. Erstens, wenn `y_hat` eine Matrix ist, gehen wir davon aus, dass die zweite Dimension Prognosewerte für jede Klasse speichert. Wir verwenden `argmax`, um die prognostizierte Klasse durch den Index für den größten Eintrag in jeder Zeile zu erhalten. Dann vergleichen wir die vorhergesagte Klasse mit der Grundwahrheit `y` elementweise. Da der Gleichheitsoperator `==` für Datentypen sensibel ist, konvertieren wir den Datentyp `y_hat` in Übereinstimmung mit dem Datentyp `y`. Das Ergebnis ist ein Tensor, der Einträge 0 (false) und 1 (true) enthält. Die Summe ergibt die Anzahl der korrekten Vorhersagen.

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)        
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

Wir werden weiterhin die zuvor definierten Variablen `y_hat` und `y` als vorhergesagte Wahrscheinlichkeitsverteilungen bzw. Beschriftungen verwenden. Wir können sehen, dass die Vorhersageklasse des ersten Beispiels 2 ist (das größte Element der Zeile ist 0,6 mit dem Index 2), was inkonsistent mit der tatsächlichen Bezeichnung 0 ist. Die Vorhersageklasse des zweiten Beispiels ist 2 (das größte Element der Zeile ist 0,5 mit dem Index 2), was mit der tatsächlichen Beschriftung 2 konsistent ist. Daher beträgt die Klassifizierungsgenauigkeitsrate für diese beiden Beispiele 0,5.

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

Ebenso können wir die Genauigkeit für jedes Modell `net` auf einem Dataset bewerten, auf das über den Dateniterator `data_iter` zugegriffen wird.

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for _, (X, y) in enumerate(data_iter):
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

Hier `Accumulator` ist eine Utility-Klasse Summen über mehrere Variablen zu akkumulieren. In der obigen Funktion `evaluate_accuracy` erstellen wir 2 Variablen in der Instanz `Accumulator`, um sowohl die Anzahl der korrekten Vorhersagen als auch die Anzahl der Vorhersagen zu speichern. Beide werden im Laufe der Zeit akkumuliert, wenn wir über den Datensatz iterieren.

```{.python .input}
#@tab all
class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

Da wir das `net`-Modell mit zufälligen Gewichten initialisiert haben, sollte die Genauigkeit dieses Modells nahe dem Zufallsraten liegen, d.h. 0,1 für 10 Klassen.

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## Ausbildung

Die Trainingsschleife für die Softmax-Regression sollte auffallend vertraut aussehen, wenn Sie unsere Implementierung der linearen Regression in :numref:`sec_linear_scratch` durchlesen. Hier umgestalten wir die Implementierung, um sie wiederverwendbar zu machen. Zuerst definieren wir eine Funktion, die für eine Epoche trainiert werden soll. Beachten Sie, dass `updater` eine allgemeine Funktion ist, um die Modellparameter zu aktualisieren, die die Stapelgröße als Argument akzeptiert. Es kann entweder ein Wrapper der `d2l.sgd` Funktion oder die integrierte Optimierungsfunktion eines Frameworks sein.

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Train a model within one epoch (defined in Chapter 3)."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

Bevor wir die Implementierung der Trainingsfunktion zeigen, definieren wir eine Utility-Klasse, die Daten in Animation plotten. Auch hier zielt es darauf ab, Code im Rest des Buches zu vereinfachen.

```{.python .input}
#@tab all
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

Die folgende Trainingsfunktion trainiert dann ein Modell `net` auf einem Trainingsdatensatz, auf den über `train_iter` für mehrere Epochen zugegriffen wird, der durch `num_epochs` spezifiziert wird. Am Ende jeder Epoche wird das Modell auf einem Testdatensatz ausgewertet, auf den über `test_iter` zugegriffen wird. Wir werden die Klasse `Animator` nutzen, um den Trainingsfortschritt zu visualisieren.

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

Als Implementierung von Grund auf verwenden wir den in :numref:`sec_linear_scratch` definierten Minibatch-stochastischen Gradientenabstieg, um die Verlustfunktion des Modells mit einer Lernrate 0.1 zu optimieren.

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """For updating parameters using minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

Jetzt trainieren wir das Modell mit 10 Epochen. Beachten Sie, dass sowohl die Anzahl der Epochen (`num_epochs`) als auch die Lernrate (`lr`) anpassbare Hyperparameter sind. Indem wir ihre Werte ändern, können wir die Klassifizierungsgenauigkeit des Modells erhöhen.

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## Vorhersage

Jetzt, da das Training abgeschlossen ist, ist unser Modell bereit, einige Bilder zu klassifizieren. Angesichts einer Reihe von Bildern werden wir ihre tatsächlichen Beschriftungen (erste Zeile der Textausgabe) und die Vorhersagen aus dem Modell (zweite Zeile der Textausgabe) vergleichen.

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## Zusammenfassung

* Mit der softmax-Regression können wir Modelle für die Multiklass-Klassifizierung trainieren.
* Die Trainingsschleife der softmax-Regression ähnelt der linearen Regression: Daten abrufen und lesen, Modelle und Verlustfunktionen definieren, dann Modelle mit Optimierungsalgorithmen trainieren. Wie Sie bald herausfinden werden, haben die gängigsten Deep Learning-Modelle ähnliche Trainingsverfahren.

## Übungen

1. In diesem Abschnitt haben wir direkt die softmax-Funktion basierend auf der mathematischen Definition der softmax-Operation implementiert. Welche Probleme könnte das verursachen? Tipp: Versuchen Sie, die Größe von $\exp(50)$ zu berechnen.
1. Die Funktion `cross_entropy` in diesem Abschnitt wurde gemäß der Definition der Querentropie-Verlustfunktion implementiert. Was könnte das Problem mit dieser Implementierung sein? Hinweis: Betrachten Sie die Domäne des Logarithmus.
1. Welche Lösungen können Sie sich vorstellen, um die beiden oben genannten Probleme zu beheben?
1. Ist es immer eine gute Idee, das wahrscheinlichste Etikett zurückzugeben? Würden Sie dies zum Beispiel für die medizinische Diagnose tun?
1. Angenommen wir möchten softmax-Regression verwenden, um das nächste Wort basierend auf einigen Features vorherzusagen. Was sind einige Probleme, die durch ein großes Vokabular entstehen könnten?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
