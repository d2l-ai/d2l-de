# Mehrschichtige Perzeptrons
:label:`sec_mlp`

In :numref:`chap_linear` haben wir softmax-Regression (:numref:`sec_softmax`) eingeführt, den Algorithmus von Grund auf neu implementiert (:numref:`sec_softmax_scratch`) und High-Level-APIs (:numref:`sec_softmax_concise`) und Schulungsklassifikatoren, um 10 Kategorien von Kleidung aus Bildern mit niedriger Auflösung zu erkennen. Auf dem Weg lernten wir, Daten zu streiten, unsere Ausgaben in eine gültige Wahrscheinlichkeitsverteilung zu zwingen, eine geeignete Verlustfunktion anzuwenden und sie in Bezug auf die Parameter unseres Modells zu minimieren. Jetzt, da wir diese Mechanik im Kontext einfacher linearer Modelle beherrschen, können wir unsere Erforschung tiefer neuronaler Netze starten, der vergleichsweise reichen Modellklasse, mit der sich dieses Buch in erster Linie beschäftigt.

## Ausgeblendete Layer

Wir haben die affine Transformation in :numref:`subsec_linear_model` beschrieben, eine lineare Transformation, die durch eine Verzerrung hinzugefügt wird. Erinnern Sie sich zunächst an die Modellarchitektur, die unserem softmax-Regressionsbeispiel entspricht, das in :numref:`fig_softmaxreg` dargestellt ist. Dieses Modell mappte unsere Eingänge direkt auf unsere Ausgänge über eine einzelne affine Transformation, gefolgt von einer softmax-Operation. Wenn unsere Labels durch eine affine Transformation wirklich mit unseren Eingabedaten in Zusammenhang stehen, wäre dieser Ansatz ausreichend. Aber Linearität in affinen Transformationen ist eine *stark* Annahme.

### Lineare Modelle können schief gehen

Zum Beispiel impliziert die Linearität die *schwächer* Annahme von *Monotonität*: Jede Erhöhung unserer Funktion muss entweder immer zu einer Erhöhung der Ausgabe unseres Modells führen (wenn das entsprechende Gewicht positiv ist) oder immer zu einer Abnahme der Ausgabe unseres Modells führen (wenn das entsprechende Gewicht negativ ist). Manchmal ergibt das Sinn. Wenn wir zum Beispiel versuchen, vorherzusagen, ob eine Person einen Kredit zurückzahlen wird, könnten wir uns vernünftigerweise vorstellen, dass ein Bewerber mit höherem Einkommen immer häufiger zurückzahlen würde als ein Bewerber mit niedrigerem Einkommen. Obwohl monotone, ist diese Beziehung wahrscheinlich nicht linear mit der Wahrscheinlichkeit der Rückzahlung verbunden. Ein Anstieg der Erträge von 0 auf 50 Tausend entspricht wahrscheinlich einem größeren Anstieg der Rückzahlungswahrscheinlichkeit als einem Anstieg von 1 Million auf 1,05 Millionen. Eine Möglichkeit, dies zu handhaben, könnte darin bestehen, unsere Daten so zu verarbeiten, dass die Linearität beispielsweise durch die Verwendung des Logarithmus des Einkommens als unsere Funktion plausibler wird.

Beachten Sie, dass wir leicht Beispiele finden können, die Monotonie verletzen. Sagen Sie zum Beispiel, dass wir die Wahrscheinlichkeit des Todes basierend auf der Körpertemperatur vorhersagen wollen. Bei Personen mit einer Körpertemperatur über 37°C weisen höhere Temperaturen auf ein höheres Risiko hin. Bei Personen mit Körpertemperaturen unter 37° C weisen höhere Temperaturen jedoch auf ein geringeres Risiko hin! Auch in diesem Fall können wir das Problem mit einer cleveren Vorverarbeitung lösen. Nämlich, wir könnten die Entfernung von 37°C als unser Merkmal verwenden.

Aber was ist mit der Klassifizierung von Bildern von Katzen und Hunden? Sollte die Erhöhung der Intensität des Pixels am Standort (13, 17) immer die Wahrscheinlichkeit erhöhen (oder immer verringern), dass das Bild einen Hund darstellt? Die Abhängigkeit von einem linearen Modell entspricht der impliziten Annahme, dass die einzige Voraussetzung für die Differenzierung von Katzen und Hunden darin besteht, die Helligkeit einzelner Pixel zu beurteilen. Dieser Ansatz ist zum Scheitern verurteilt in einer Welt, in der das Umkehren eines Bildes die Kategorie bewahrt.

Und trotz der scheinbaren Absurdität der Linearität hier, im Vergleich zu unseren vorherigen Beispielen, ist es weniger offensichtlich, dass wir das Problem mit einer einfachen Vorverarbeitung beheben könnten. Das liegt daran, dass die Signifikanz eines Pixels auf komplexe Weise von seinem Kontext abhängt (den Werten der umgebenden Pixel). Zwar könnte es eine Darstellung unserer Daten geben, die die relevanten Wechselwirkungen zwischen unseren Merkmalen berücksichtigen würde. Darüber hinaus ist ein lineares Modell geeignet, aber wir wissen einfach nicht, wie wir es von Hand berechnen können. Bei tiefen neuronalen Netzwerken nutzten wir Beobachtungsdaten, um gemeinsam sowohl eine Darstellung über versteckte Schichten als auch einen linearen Prädiktor zu erlernen, der auf diese Darstellung einwirkt.

### Einbinden von ausgeblendeten Layern

Wir können diese Einschränkungen linearer Modelle überwinden und eine allgemeinere Klasse von Funktionen verarbeiten, indem wir eine oder mehrere versteckte Ebenen integrieren. Der einfachste Weg, dies zu tun, besteht darin, viele vollständig verbundene Layer übereinander zu stapeln. Jede Schicht wird in die darüber liegende Ebene eingespeist, bis wir Ausgänge generieren. Wir können uns die ersten $L-1$ Ebenen als unsere Repräsentation und die letzte Schicht als unseren linearen Prädiktor vorstellen. Diese Architektur wird üblicherweise als mehrschichtiges Perzeptron* bezeichnet, oft abgekürzt als *MLP*. Im Folgenden zeigen wir ein MLP diagrammatisch (:numref:`fig_mlp`).

![An MLP with a hidden layer of 5 hidden units. ](../img/mlp.svg)
:label:`fig_mlp`

Dieser MLP hat 4 Eingänge, 3 Ausgänge, und seine versteckte Schicht enthält 5 versteckte Einheiten. Da der Eingabe-Layer keine Berechnungen beinhaltet, erfordert das Erzeugen von Ausgaben mit diesem Netzwerk die Implementierung der Berechnungen sowohl für die ausgeblendeten als auch für die Ausgabe-Layer. Daher ist die Anzahl der Layer in diesem MLP 2. Beachten Sie, dass diese Layer beide vollständig verbunden sind. Jeder Eingang beeinflusst jedes Neuron in der versteckten Schicht, und jeder von ihnen wiederum beeinflusst jedes Neuron in der Ausgabeschicht.

### Von linear zu nichtlinear

Nach wie vor bezeichnen wir durch die Matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ einen Minibatch von $n$ Beispielen, bei denen jedes Beispiel $d$ Eingänge (Features) hat. Bezeichnen Sie bei einem MLP mit einer verborgenen Schicht, dessen versteckte Schicht $h$ versteckte Einheiten enthält, mit $\mathbf{H} \in \mathbb{R}^{n \times h}$ die Ausgaben des ausgeblendeten Layers. Hier wird $\mathbf{H}$ auch als *versteckte Variable* oder eine *versteckte Variable* bezeichnet. Da die versteckten und Ausgabe-Layer beide vollständig miteinander verbunden sind, haben wir Hidden-Layer-Gewichte $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ und Verzerrungen $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$ und Ausgabe-Layer-Gewichte $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ und Verzerrungen $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$. Formal berechnen wir die Ausgänge $\mathbf{O} \in \mathbb{R}^{n \times q}$ des ein-versteckten MLP wie folgt:

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

Beachten Sie, dass wir nach dem Hinzufügen der ausgeblendeten Ebene jetzt zusätzliche Parametersätze verfolgen und aktualisieren müssen. Was haben wir im Gegenzug gewonnen? Sie könnten überrascht sein, herauszufinden, dass — im oben definierten Modell — wir nichts für unsere Probleme* gewinnen! Der Grund ist klar. Die versteckten Einheiten oben werden durch eine affine Funktion der Eingänge gegeben, und die Ausgänge (pre-softmax) sind nur eine affine Funktion der versteckten Einheiten. Eine affine Funktion einer affinen Funktion ist selbst eine affine Funktion. Darüber hinaus war unser lineares Modell bereits in der Lage, jede affine Funktion darzustellen.

Wir können die Äquivalenz formal sehen, indem wir beweisen, dass wir für alle Werte der Gewichte einfach die versteckte Schicht ausblenden können, was ein äquivalentes Single-Layer-Modell mit den Parametern $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ und $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ ergibt:

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

Um das Potenzial von Multilayer-Architekturen zu realisieren, benötigen wir einen weiteren wichtigen Bestandteil: eine nichtlineare *Aktivierungsfunktion* $\sigma$, die nach der affinen Transformation auf jede versteckte Einheit angewendet wird. Die Ausgänge der Aktivierungsfunktionen (z.B. $\sigma(\cdot)$) heißen *Aktivierungen*. Im Allgemeinen ist es mit Aktivierungsfunktionen nicht mehr möglich, unser MLP in ein lineares Modell zu kollabieren:

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

Da jede Zeile in $\mathbf{X}$ einem Beispiel im Minibatch entspricht, definieren wir mit einem gewissen Missbrauch der Notation die Nichtlinearität $\sigma$, die auf ihre Eingaben in einer Zeilenweise angewendet wird, d.h. ein Beispiel nach dem anderen. Beachten Sie, dass wir die Notation für softmax auf die gleiche Weise verwendet haben, um eine rowwise Operation in :numref:`subsec_softmax_vectorization` zu bezeichnen. Oft sind, wie in diesem Abschnitt, die Aktivierungsfunktionen, die wir auf versteckte Ebenen anwenden, nicht nur rowwise, sondern elementweise. Das bedeutet, dass wir nach der Berechnung des linearen Teils der Ebene jede Aktivierung berechnen können, ohne die Werte der anderen versteckten Einheiten zu betrachten. Dies gilt für die meisten Aktivierungsfunktionen.

Um allgemeinere MLPs zu erstellen, können wir weiterhin solche versteckten Schichten stapeln, z. B. $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ und $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$, übereinander, was immer ausdrucksstärkere Modelle liefert.

### Universal-Näherungsgeräte

MLPs können komplexe Interaktionen zwischen unseren Eingängen über ihre versteckten Neuronen erfassen, die von den Werten der einzelnen Eingänge abhängen. Wir können leicht versteckte Knoten entwerfen, um willkürliche Berechnungen durchzuführen, zum Beispiel grundlegende Logikoperationen an einem Paar von Eingaben. Darüber hinaus ist es für bestimmte Optionen der Aktivierungsfunktion allgemein bekannt, dass MLPs universelle Approximatoren sind. Selbst mit einem Single-Hidden-Layer-Netzwerk, das genügend Knoten (möglicherweise absurd viele) und dem richtigen Satz von Gewichten gegeben ist, können wir jede Funktion modellieren, obwohl das Lernen dieser Funktion der schwierige Teil ist. Sie könnten sich Ihr neuronales Netzwerk als ein bisschen wie die C-Programmiersprache vorstellen. Die Sprache ist, wie jede andere moderne Sprache, in der Lage, jedes berechenbare Programm auszudrücken. Aber tatsächlich ein Programm zu finden, das Ihren Vorgaben entspricht, ist der schwierige Teil.

Darüber hinaus, nur weil ein Single-Hidden-Layer-Netzwerk
*kann* jede Funktion lernen
bedeutet nicht, dass Sie versuchen sollten, alle Ihre Probleme mit Single-Layer-Netzwerken zu lösen. In der Tat können wir viele Funktionen viel kompakter annähern, indem wir tiefere (im Vergleich zu breiteren) Netzwerke verwenden. In den folgenden Kapiteln werden wir strengere Argumente ansprechen.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Aktivierungsfunktionen

Aktivierungsfunktionen entscheiden, ob ein Neuron aktiviert werden soll oder nicht, indem die gewichtete Summe berechnet und weitere Bias hinzugefügt wird. Sie sind differenzierbare Operatoren, um Eingangssignale in Ausgänge zu transformieren, während die meisten von ihnen Nicht-Linearität hinzufügen. Da Aktivierungsfunktionen grundlegend für Deep Learning sind, lassen Sie uns kurz einige allgemeine Aktivierungsfunktionen untersuchen.

### ReLu (Funktion)

Die beliebteste Wahl, sowohl aufgrund der Einfachheit der Implementierung als auch seiner guten Leistung bei einer Vielzahl von prädiktiven Aufgaben, ist die *rektifizierte lineare Einheit* (*RelU*). ReLu bietet eine sehr einfache nichtlineare Transformation. Bei einem Element $x$ wird die Funktion als das Maximum dieses Elements und $0$ definiert:

$$\operatorname{ReLU}(x) = \max(x, 0).$$

Informell behält die RELU-Funktion nur positive Elemente bei und verwirft alle negativen Elemente, indem die entsprechenden Aktivierungen auf 0 gesetzt werden. Um etwas Intuition zu gewinnen, können wir die Funktion plotten. Wie Sie sehen können, ist die Aktivierungsfunktion stückweise linear.

```{.python .input}
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

Wenn die Eingabe negativ ist, ist die Ableitung der RelU-Funktion 0, und wenn die Eingabe positiv ist, ist die Ableitung der RelU-Funktion 1. Beachten Sie, dass die ReLU-Funktion nicht differenzierbar ist, wenn die Eingabe genau gleich 0 annimmt. In diesen Fällen verwenden wir standardmäßig die linke Ableitung und sagen, dass die Ableitung 0 ist, wenn die Eingabe 0 ist. Wir können damit davonkommen, weil die Eingabe niemals tatsächlich Null sein kann. Es gibt ein altes Sprichwort, dass, wenn subtile Randbedingungen wichtig sind, wahrscheinlich (*real*) Mathematik, nicht Ingenieurwesen. Diese konventionelle Weisheit kann hier gelten. Wir plotten die Ableitung der reLU-Funktion unten dargestellt.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

Der Grund für die Verwendung von ReLu ist, dass seine Derivate besonders gut benommen sind: Entweder sie verschwinden oder sie lassen einfach das Argument durch. Dies macht die Optimierung besser benommen und es mildert das gut dokumentierte Problem der verschwundenen Gradienten, die frühere Versionen neuronaler Netzwerke geplagt haben (mehr dazu später).

Beachten Sie, dass es viele Varianten für die RelU-Funktion gibt, einschließlich der Funktion *parametrisierter RelU* (*PreLU*) :cite:`He.Zhang.Ren.ea.2015`. Diese Variation fügt RelU einen linearen Term hinzu, sodass einige Informationen immer noch durchkommen, selbst wenn das Argument negativ ist:

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Sigmoid-Funktion

Die *sigmoid-Funktion* wandelt ihre Eingänge, für die Werte in der Domäne $\mathbb{R}$ liegen, in Ausgänge um, die auf dem Intervall liegen (0, 1). Aus diesem Grund wird das Sigmoid oft als *squashing-Funktion* bezeichnet: es quetscht jede Eingabe im Bereich (-inf, inf) zu einem Wert im Bereich (0, 1):

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

In den frühesten neuronalen Netzen interessierten sich die Wissenschaftler für die Modellierung von biologischen Neuronen, die entweder *Feuer* oder *nicht feuern*. So konzentrierten sich die Pioniere dieses Feldes, die den ganzen Weg zurück zu McCulloch und Pitts, den Erfindern des künstlichen Neurons, auf Schwellenwerte. Eine Schwellenwertaktivierung nimmt den Wert 0 an, wenn ihre Eingabe unter einem Schwellenwert liegt, und Wert 1, wenn die Eingabe den Schwellenwert überschreitet.

Wenn die Aufmerksamkeit auf das Gradientenbasierte Lernen verschoben wurde, war die Sigmoidfunktion eine natürliche Wahl, da sie eine glatte, differenzierbare Annäherung an eine Schwellenwerteinheit darstellt. Sigmoide werden immer noch weit verbreitet als Aktivierungsfunktionen auf den Ausgabeeinheiten verwendet, wenn wir die Ausgaben als Wahrscheinlichkeiten für binäre Klassifizierungsprobleme interpretieren wollen (man kann sich das Sigmoid als Sonderfall des Softmax vorstellen). Das Sigmoid wurde jedoch meist durch das einfachere und leichter trainierbare ReLu für die meisten Anwendungen in versteckten Schichten ersetzt. In späteren Kapiteln zu wiederkehrenden neuronalen Netzwerken werden wir Architekturen beschreiben, die Sigmoid-Einheiten verwenden, um den Informationsfluss über die Zeit zu steuern.

Im Folgenden plotten wir die Sigmoid-Funktion. Beachten Sie, dass sich die Sigmoidfunktion einer linearen Transformation nähert, wenn die Eingabe nahe 0 liegt.

```{.python .input}
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

Die Ableitung der Sigmoidfunktion wird durch folgende Gleichung gegeben:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

Die Ableitung der Sigmoidfunktion ist unten dargestellt. Beachten Sie, dass, wenn die Eingabe 0 ist, die Ableitung der Sigmoid-Funktion ein Maximum von 0,25 erreicht. Da der Eingang von 0 in beide Richtungen abweicht, nähert sich die Ableitung 0.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

### Tanh-Funktion

Wie die Sigmoidfunktion zerquetscht auch die tanh (hyperbolische Tangente) Funktion ihre Eingaben und wandelt sie in Elemente im Intervall zwischen -1 und 1 um:

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

Wir plotten die tanh-Funktion unten. Beachten Sie, dass sich die tanh-Funktion einer linearen Transformation nähert, wenn die Eingabe 0 nähert. Obwohl die Form der Funktion der Sigmoidfunktion ähnlich ist, weist die tanh-Funktion Punktsymmetrie über den Ursprung des Koordinatensystems auf.

```{.python .input}
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

Die Ableitung der tanh-Funktion lautet:

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

Die Ableitung der tanh-Funktion ist unten dargestellt. Da sich der Eingang 0 nähert, nähert sich die Ableitung der tanh-Funktion einem Maximum von 1. Und wie wir mit der Sigmoidfunktion gesehen haben, nähert sich die Ableitung der tanh-Funktion 0, wenn sich der Eingang von 0 in beide Richtungen weg bewegt.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

Zusammenfassend wissen wir nun, wie man Nichtlinearitäten einbaut, um ausdrucksstarke, mehrschichtige neuronale Netzwerkarchitekturen zu erstellen. Als Randnotiz, Ihr Wissen bringt Sie bereits das Kommando über ein ähnliches Toolkit wie ein Praktiker um 1990. In gewisser Weise haben Sie einen Vorteil gegenüber allen, die in den 1990er Jahren arbeiten, da Sie leistungsstarke Open-Source-Deep-Learning-Frameworks nutzen können, um Modelle schnell mit nur wenigen Codezeilen zu erstellen. Zuvor musste die Ausbildung dieser Netzwerke Tausende von Zeilen von C und Fortran codieren.

## Zusammenfassung

* MLP fügt eine oder mehrere vollständig verbundene versteckte Layer zwischen den Ausgabe- und Eingabe-Layern hinzu und transformiert die Ausgabe der ausgeblendeten Ebene über eine Aktivierungsfunktion.
* Zu den häufig verwendeten Aktivierungsfunktionen gehören die RelU-Funktion, die Sigmoid-Funktion und die tanh-Funktion.

## Übungen

1. Berechnen Sie die Ableitung der Aktivierungsfunktion PreLU.
1. Zeigen Sie, dass ein MLP, der nur RelU (oder PreLu) verwendet, eine kontinuierliche stückweise lineare Funktion konstruiert.
1. Zeigen Sie, dass $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
1. Angenommen, wir haben eine Nichtlinearität, die für jeweils eine Minibatch gilt. Welche Arten von Problemen erwarten Sie, dass dies verursacht?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:
