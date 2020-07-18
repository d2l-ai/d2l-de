# Numerische Stabilität und Initialisierung
:label:`sec_numerical_stability`

Bisher erforderte jedes Modell, das wir implementiert haben, dass wir seine Parameter entsprechend einer vorgegebenen Verteilung initialisieren. Bis jetzt haben wir das Initialisierungsschema als selbstverständlich angesehen und die Details darüber geschliffenen, wie diese Entscheidungen getroffen werden. Vielleicht haben Sie sogar den Eindruck bekommen, dass diese Entscheidungen nicht besonders wichtig sind. Im Gegenteil, die Wahl des Initialisierungsschemas spielt eine wichtige Rolle beim Lernen neuronaler Netzwerke, und es kann entscheidend für die Aufrechterhaltung der numerischen Stabilität sein. Darüber hinaus können diese Optionen auf interessante Weise mit der Wahl der nichtlinearen Aktivierungsfunktion gebunden werden. Welche Funktion wir wählen und wie wir Parameter initialisieren, kann bestimmen, wie schnell unser Optimierungsalgorithmus konvergiert. Schlechte Entscheidungen hier können dazu führen, dass wir während des Trainings explodierende oder verschwindende Steigungen begegnen. In diesem Abschnitt vertiefen wir uns ausführlicher mit diesen Themen und diskutieren einige nützliche Heuristiken, die Sie während Ihrer gesamten Karriere im Deep Learning nützlich finden werden.

## Verschwindende und explodierende Verläufe

Betrachten Sie ein tiefes Netzwerk mit $L$ Schichten, Eingang $\mathbf{x}$ und Ausgang $\mathbf{o}$. Mit jeder Schicht $l$ definiert durch eine Transformation $f_l$ parametrisiert durch Gewichte $\mathbf{W}^{(l)}$, deren versteckte Variable $\mathbf{h}^{(l)}$ ist (lassen Sie $\mathbf{h}^{(0)} = \mathbf{x}$), unser Netzwerk kann ausgedrückt werden als:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ and thus } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

Wenn alle versteckten Variablen und die Eingabe Vektoren sind, können wir den Gradient von $\mathbf{o}$ in Bezug auf einen beliebigen Satz von Parametern $\mathbf{W}^{(l)}$ wie folgt schreiben:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$

Mit anderen Worten, dieser Gradient ist das Produkt von $L-l$ Matrizen $\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$ und dem Gradientenvektor $\mathbf{v}^{(l)}$. So sind wir anfällig für die gleichen Probleme des numerischen Unterflusses, die oft auftreten, wenn zu viele Wahrscheinlichkeiten miteinander multipliziert werden. Beim Umgang mit Wahrscheinlichkeiten besteht ein gemeinsamer Trick darin, in den Log-Space zu wechseln, d.h. den Druck von der Mantisse auf den Exponenten der numerischen Darstellung zu verschieben. Leider ist unser obige Problem ernster: Zunächst können die Matrizen $\mathbf{M}^{(l)}$ eine Vielzahl von Eigenwerten haben. Sie können klein oder groß sein, und ihr Produkt könnte *sehr groß* oder *sehr klein* sein.

Die Risiken, die durch instabile Gradienten entstehen, gehen über die numerische Darstellung hinaus. Unvorhersehbare Gradienten gefährden auch die Stabilität unserer Optimierungsalgorithmen. Möglicherweise stehen wir mit Parameteraktualisierungen konfrontiert, die entweder (i) übermäßig groß sind und unser Modell zerstören (das Problem mit dem Explodieren des Gradient*) oder (ii) übermäßig klein (das Problem mit dem Verschwundenen Gradient*), was das Lernen unmöglich macht, da sich die Parameter kaum bei jedem Update bewegen.

### Verschwindende Farbverläufe

Ein häufiger Täter, der das Problem des Verschwindenden Gradienten verursacht, ist die Wahl der Aktivierungsfunktion $\sigma$, die nach den linearen Operationen jeder Ebene angehängt wird. Historisch gesehen war die Sigmoidfunktion $1/(1 + \exp(-x))$ (eingeführt in :numref:`sec_mlp`) beliebt, weil sie einer Schwellenwertfunktion ähnelt. Da frühe künstliche neuronale Netze von biologischen neuronalen Netzen inspiriert wurden, schien die Idee von Neuronen, die entweder *voll* oder gar nicht* (wie biologische Neuronen) feuern, attraktiv zu sein. Werfen wir einen genaueren Blick auf das Sigma, um zu sehen, warum es zu verschwindenden Steigungen führen kann.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Wie Sie sehen können, verschwindet der Gradient des Sigmas sowohl, wenn seine Eingänge groß sind als auch wenn sie klein sind. Darüber hinaus können die Steigungen des Gesamtprodukts verschwinden, wenn wir nicht in der Goldilocks-Zone sind, wo die Eingänge für viele der Sigmoide nahe Null liegen. Wenn unser Netzwerk viele Ebenen aufweist, wenn wir nicht vorsichtig sind, wird der Farbverlauf wahrscheinlich auf einer Ebene abgeschnitten. In der Tat, dieses Problem verwendet, um tiefe Netzwerktraining plagen. Folglich haben sich RELus, die stabiler (aber weniger neural plausibel) sind, als Standardwahl für Praktiker erwiesen.

### Explodieren von Verläufen

Das gegenteilige Problem, wenn Gradienten explodieren, kann ähnlich belästigend sein. Um dies etwas besser zu veranschaulichen, zeichnen wir 100 Gaußsche Zufallsmatrizen und multiplizieren sie mit einer anfänglichen Matrix. Für die Skala, die wir ausgewählt haben (die Wahl der Varianz $\sigma^2=1$), explodiert das Matrixprodukt. Wenn dies aufgrund der Initialisierung eines tiefen Netzwerks geschieht, haben wir keine Chance, einen Gradientenabstiegs-Optimierer zu konvergieren.

```{.python .input}
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('after multiplying 100 matrices', M)
```

```{.python .input}
#@tab pytorch
M = torch.normal(0, 1, size=(4,4))
print('a single matrix \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('after multiplying 100 matrices\n', M)
```

```{.python .input}
#@tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('after multiplying 100 matrices\n', M.numpy())
```

### Die Symmetrie durchbrechen

Ein weiteres Problem im neuronalen Netzwerkdesign ist die Symmetrie, die ihrer Parametrierung innewohnt. Angenommen, wir haben eine einfache MLP mit einem versteckten Layer und zwei Einheiten. In diesem Fall könnten wir die Gewichte $\mathbf{W}^{(1)}$ der ersten Schicht permutieren und ebenfalls die Gewichte der Ausgabeschicht permutieren, um die gleiche Funktion zu erhalten. Es gibt nichts Besonderes, das die erste versteckte Einheit gegenüber der zweiten versteckten Einheit unterscheidet. Mit anderen Worten, wir haben Permutationssymmetrie zwischen den versteckten Einheiten jeder Schicht.

Das ist mehr als nur ein theoretisches Ärgernis. Betrachten Sie die oben erwähnte ein-versteckte MLP mit zwei versteckten Einheiten. Angenommen, der Ausgabe-Layer wandelt die beiden ausgeblendeten Einheiten in nur eine Ausgabeeinheit um. Stellen Sie sich vor, was passieren würde, wenn wir alle Parameter der versteckten Schicht als $\mathbf{W}^{(1)} = c$ für eine konstante $c$ initialisieren würden. In diesem Fall nimmt jede versteckte Einheit während der Vorwärtsausbreitung die gleichen Eingänge und Parameter und erzeugt die gleiche Aktivierung, die der Ausgangseinheit zugeführt wird. Während der Backpropagation ergibt die Differenzierung der Ausgabeeinheit hinsichtlich der Parameter $\mathbf{W}^{(1)}$ einen Gradienten, dessen Elemente alle den gleichen Wert annehmen. Daher haben alle Elemente von $\mathbf{W}^{(1)}$ nach der Gradienten-basierten Iteration (z. B. Minibatch stochastischer Gradientenabstieg) immer noch den gleichen Wert. Solche Iterationen würden niemals die Symmetrie* alleine durchbrechen und wir könnten die Ausdruckskraft des Netzwerks niemals realisieren. Die ausgeblendete Ebene würde sich so verhalten, als hätte sie nur eine einzige Einheit. Beachten Sie, dass während minibatch stochastischen Gradienten Abstieg diese Symmetrie nicht brechen würde, Dropout Regularisierung würde!

## Parameter-Initialisierung

Eine Möglichkeit, die oben aufgeworfenen Probleme zu beheben — oder zumindest zu mindern, besteht in einer sorgfältigen Initialisierung. Zusätzliche Sorgfalt bei der Optimierung und geeignete Regularisierung können die Stabilität weiter verbessern.

### Standardinitialisierung

In den vorangegangenen Abschnitten, z.B. in :numref:`sec_linear_concise`, haben wir eine Normalverteilung verwendet, um die Werte unserer Gewichte zu initialisieren. Wenn wir die Initialisierungsmethode nicht angeben, verwendet das Framework eine standardmäßige zufällige Initialisierungsmethode, die oft gut in der Praxis für moderate Problemgrößen funktioniert.

### Xavier-Initialisierung

Lassen Sie uns die Skalenverteilung einer Ausgabe (zB eine versteckte Variable) $o_{i}$ für eine voll verbundene Schicht betrachten
*ohne Nichtlinearien*.
Mit $n_\mathrm{in}$ Eingängen $x_j$ und ihren zugehörigen Gewichten $w_{ij}$ für diesen Layer wird eine Ausgabe von

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

Die Gewichte $w_{ij}$ werden unabhängig von der gleichen Verteilung gezogen. Lassen Sie uns außerdem davon ausgehen, dass diese Verteilung Null Mittelwert und Varianz $\sigma^2$ hat. Beachten Sie, dass dies nicht bedeutet, dass die Verteilung Gaußsch sein muss, nur dass der Mittelwert und die Varianz existieren müssen. Lassen Sie uns vorerst davon ausgehen, dass die Eingaben für die Schicht $x_j$ auch Null Mittelwert und Varianz $\gamma^2$ haben und dass sie unabhängig von $w_{ij}$ und unabhängig voneinander sind. In diesem Fall können wir den Mittelwert und die Varianz von $o_i$ wie folgt berechnen:

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Eine Möglichkeit, die Varianz fest zu halten, besteht darin, $n_\mathrm{in} \sigma^2 = 1$ einzustellen. Betrachten Sie nun die Rückverbreitung. Dort stehen wir vor einem ähnlichen Problem, wenn auch mit Gradienten, die von den Schichten näher an der Ausgabe propagiert werden. Unter Verwendung der gleichen Argumentation wie für die Vorwärtspropagation sehen wir, dass die Varianz der Gradienten sprengen kann, es sei denn, $n_\mathrm{out} \sigma^2 = 1$, wobei $n_\mathrm{out}$ die Anzahl der Ausgänge dieser Schicht ist. Das lässt uns in ein Dilemma: Wir können beide Bedingungen möglicherweise nicht gleichzeitig erfüllen. Stattdessen versuchen wir einfach zu erfüllen:

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

Dies ist die Argumentation, die der jetzigen Standard- und praktisch vorteilhaften *Xavier-Initialisierung* zugrunde liegt, benannt nach dem ersten Autor seiner Schöpfer :cite:`Glorot.Bengio.2010`. Typischerweise Samples bei der Xavier-Initialisierung Gewichtungen aus einer Gaußschen Verteilung mit dem Mittelwert Null und der Varianz $\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$. Wir können Xavers Intuition auch anpassen, um die Varianz beim Abtasten von Gewichten aus einer einheitlichen Verteilung zu wählen. Beachten Sie, dass die gleichmäßige Verteilung $U(-a, a)$ die Varianz $\frac{a^2}{3}$ aufweist. Das Einstecken von $\frac{a^2}{3}$ in unseren Zustand auf $\sigma^2$ ergibt den Vorschlag, gemäß

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

Obwohl die Annahme, dass Nichtvorhandensein von Nichtlinearitäten in der obigen mathematischen Argumentation in neuronalen Netzwerken leicht verletzt werden kann, erweist sich die Xavier-Initialisierungsmethode als gut in der Praxis.

### Jenseits

Die obige Argumentation zerkratzt kaum die Oberfläche moderner Ansätze zur Parameterinitialisierung. Ein Deep Learning-Framework implementiert oft mehr als ein Dutzend verschiedene Heuristiken. Darüber hinaus ist die Parameter-Initialisierung weiterhin ein wichtiger Bereich der Grundlagenforschung im Deep Learning. Dazu gehören Heuristiken, die auf gebundene (gemeinsam genutzte) Parameter, Superauflösung, Sequenzmodelle und andere Situationen spezialisiert sind. Beispielsweise demonstrierten Xiao et al. die Möglichkeit, 10000-Layer-neuronale Netze ohne architektonische Tricks zu trainieren, indem sie eine sorgfältig gestaltete Initialisierungsmethode :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018` verwenden.

Wenn das Thema Sie interessiert, empfehlen wir einen tiefen Einblick in die Angebote dieses Moduls, lesen Sie die Beiträge, die jede Heuristik vorgeschlagen und analysiert haben, und erkunden Sie dann die neuesten Publikationen zu diesem Thema. Vielleicht werden Sie über eine clevere Idee stolpern oder sogar erfinden und eine Implementierung zu Deep Learning-Frameworks beitragen.

## Zusammenfassung

* Verschwindende und explodierende Gradienten sind häufige Probleme in tiefen Netzwerken. Bei der Parameterinitialisierung ist große Sorgfalt erforderlich, um sicherzustellen, dass Gradienten und Parameter gut kontrolliert bleiben.
* Initialisierungsheuristiken sind erforderlich, um sicherzustellen, dass die anfänglichen Verläufe weder zu groß noch zu klein sind.
* ReLU-Aktivierungsfunktionen mildern das Problem des Verschwindenden Gradienten. Dies kann die Konvergenz beschleunigen.
* Zufällige Initialisierung ist der Schlüssel, um sicherzustellen, dass die Symmetrie vor der Optimierung unterbrochen wird.
* Die Xaver-Initialisierung legt nahe, dass für jede Ebene die Varianz einer Ausgabe nicht von der Anzahl der Eingaben beeinflusst wird und die Varianz eines Gradienten nicht von der Anzahl der Ausgänge beeinflusst wird.

## Übungen

1. Können Sie andere Fälle entwerfen, in denen ein neuronales Netzwerk Symmetrie aufweist, die neben der Permutationssymmetrie in den Schichten eines MLP brechen muss?
1. Können wir alle Gewichtsparameter in linearer Regression oder in Softmax-Regression auf den gleichen Wert initialisieren?
1. Nachschlagen analytischer Grenzen auf die Eigenwerte des Produkts von zwei Matrizen. Was sagt Ihnen das, um sicherzustellen, dass Steigungen gut konditioniert sind?
1. Wenn wir wissen, dass einige Begriffe abweichen, können wir das nach der Tatsache beheben? Schauen Sie sich das Papier auf layer-weise adaptive Rate Skalierung für Inspiration :cite:`You.Gitman.Ginsburg.2017`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:
