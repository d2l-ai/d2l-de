# Lineare Algebra
:label:`sec_linear-algebra`

Nun, da Sie Daten speichern und manipulieren können, lassen Sie uns kurz die Teilmenge der grundlegenden linearen Algebra überprüfen, die Sie benötigen, um die meisten Modelle in diesem Buch zu verstehen und zu implementieren. Im Folgenden stellen wir die grundlegenden mathematischen Objekte, Arithmetik und Operationen in der linearen Algebra vor, wobei jeder von ihnen durch mathematische Notation und die entsprechende Implementierung im Code ausgedrückt wird.

## Skalare

Wenn Sie nie lineare Algebra oder maschinelles Lernen studiert haben, dann bestand Ihre bisherige Erfahrung mit Mathematik wahrscheinlich darin, über eine Nummer nachzudenken. Und wenn Sie jemals ein Scheckbuch ausgeglichen oder sogar für das Abendessen in einem Restaurant bezahlt haben, dann wissen Sie bereits, wie man grundlegende Dinge wie das Hinzufügen und Multiplizieren von Zahlenpaaren macht. Zum Beispiel beträgt die Temperatur in Palo Alto $52$ Grad Fahrenheit. Formal nennen wir Werte, die aus nur einer numerischen Menge*Skalaren* bestehen. Wenn Sie diesen Wert in Celsius (die sinnvollere Temperaturskala des metrischen Systems) konvertieren möchten, würden Sie den Ausdruck $c = \frac{5}{9}(f - 32)$ auswerten und $f$ auf $52$ einstellen. In dieser Gleichung sind jeder der Begriffs—$5$, $9$ und $32$ — skalare Werte. Die Platzhalter $c$ und $f$ heißen *Variablen* und stellen unbekannte Skalarwerte dar.

In diesem Buch übernehmen wir die mathematische Notation, bei der skalare Variablen durch gewöhnliche Kleinbuchstaben bezeichnet werden (z. B. $x$, $y$ und $z$). Wir bezeichnen den Raum aller (kontinuierlichen) *real werte* Skalare mit $\mathbb{R}$. Zur Zweckmäßigkeit werden wir auf strenge Definitionen dessen achten, was genau *Raum* ist, aber denken Sie daran, dass der Ausdruck $x \in \mathbb{R}$ ein formaler Weg ist, um zu sagen, dass $x$ ein real bewerteter Skalar ist. Das Symbol $\in$ kann „in“ ausgesprochen werden und bezeichnet einfach die Mitgliedschaft in einem Satz. Analog könnten wir $x, y \in \{0, 1\}$ schreiben, um zu sagen, dass $x$ und $y$ Zahlen sind, deren Wert nur $0$ oder $1$ sein kann.

Ein Skalar wird durch einen Tensor mit nur einem Element dargestellt. Im nächsten Snippet instanziieren wir zwei Skalare und führen einige vertraute arithmetische Operationen mit ihnen durch, nämlich Addition, Multiplikation, Division und Exponentiation.

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant([3.0])
y = tf.constant([2.0])

x + y, x * y, x / y, x**y
```

## Vektoren

Sie können sich einen Vektor als einfach eine Liste skalarer Werte vorstellen. Wir nennen diese Werte die *Elements* (*Einträge* oder *Komponenten*) des Vektors. Wenn unsere Vektoren Beispiele aus unserem Datensatz darstellen, haben ihre Werte eine echte Bedeutung. Wenn wir zum Beispiel ein Modell ausbilden, um das Risiko vorherzusagen, dass ein Darlehen ausfallen, könnten wir jedem Antragsteller einen Vektor zuordnen, dessen Komponenten seinem Einkommen, seiner Beschäftigungsdauer, der Anzahl der früheren Ausfälle und anderen Faktoren entsprechen. Wenn wir das Risiko von Herzinfarkten untersuchen, die Krankenhauspatienten potenziell konfrontiert sind, könnten wir jeden Patienten durch einen Vektor darstellen, dessen Komponenten ihre neuesten Vitalzeichen, Cholesterinspiegel, Minuten der Übung pro Tag usw. erfassen. Buchstaben (z. B. $\mathbf{x}$, $\mathbf{y}$ und $\mathbf{z})$.

Wir arbeiten mit Vektoren über eindimensionale Tensoren. Im Allgemeinen können Tensoren beliebige Längen haben, abhängig von den Speichergrenzen Ihrer Maschine.

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

Wir können auf jedes Element eines Vektors verweisen, indem wir einen Tiefpunkt verwenden. Zum Beispiel können wir auf das $i^\mathrm{th}$ Element von $\mathbf{x}$ von $x_i$ verweisen. Beachten Sie, dass das Element $x_i$ ein Skalar ist, so dass wir die Schriftart nicht fett stellen, wenn wir darauf verweisen. Umfangreiche Literatur betrachtet Spaltenvektoren als Standardausrichtung von Vektoren, so auch dieses Buch. In Mathematik kann ein Vektor $\mathbf{x}$ als

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

wobei $x_1, \ldots, x_n$ Elemente des Vektors sind. Im Code greifen wir auf jedes Element zu, indem wir in den Tensor indizieren.

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

### Länge, Dimensionalität und Form

Lassen Sie uns einige Konzepte von :numref:`sec_ndarray` überarbeiten. Ein Vektor ist nur ein Array von Zahlen. Und so wie jedes Array eine Länge hat, so auch jeder Vektor. Wenn wir in der mathematischen Notation sagen wollen, dass ein Vektor $\mathbf{x}$ aus $n$ realen Skalaren besteht, können wir dies als $\mathbf{x} \in \mathbb{R}^n$ ausdrücken. Die Länge eines Vektors wird üblicherweise *Dimension* des Vektors genannt.

Wie bei einem gewöhnlichen Python-Array können wir auf die Länge eines Tensors zugreifen, indem wir Pythons eingebaute `len()`-Funktion aufrufen.

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

Wenn ein Tensor einen Vektor darstellt (mit genau einer Achse), können wir auch über das Attribut `.shape` auf seine Länge zugreifen. Die Form ist ein Tupel, das die Länge (Dimensionalität) entlang jeder Achse des Tensors auflistet. Bei Tensoren mit nur einer Achse hat die Form nur ein Element.

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

Beachten Sie, dass das Wort „Dimension“ in diesen Kontexten überlastet wird und dies dazu neigt, Menschen zu verwirren. Um zu verdeutlichen, verwenden wir die Dimensionalität eines *Vektor* oder einer *Achse*, um auf seine Länge zu verweisen, d.h. die Anzahl der Elemente eines Vektors oder einer Achse. Wir verwenden jedoch die Dimensionalität eines Tensors, um sich auf die Anzahl der Achsen zu beziehen, die ein Tensor hat. In diesem Sinne ist die Dimensionalität einer Achse eines Tensors die Länge dieser Achse.

## Matrizen

Genauso wie Vektoren Skalare von Ordnung Null zu Ordnung eins verallgemeinern, verallgemeinern Matrizen Vektoren von Ordnung eins zu Ordnung zwei. Matrizen, die wir typischerweise mit Großbuchstaben (z. B. $\mathbf{X}$, $\mathbf{Y}$ und $\mathbf{Z}$) bezeichnen, werden im Code als Tensoren mit zwei Achsen dargestellt.

In der mathematischen Notation verwenden wir $\mathbf{A} \in \mathbb{R}^{m \times n}$, um auszudrücken, dass die Matrix $\mathbf{A}$ aus $m$ Zeilen und $n$ Spalten mit realen Skalaren besteht. Visuell können wir jede Matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ als Tabelle illustrieren, wobei jedes Element $a_{ij}$ zur $i^{\mathrm{th}}$ Zeile und $j^{\mathrm{th}}$ Spalte gehört:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

Für jeden $\mathbf{A} \in \mathbb{R}^{m \times n}$ ist die Form von $\mathbf{A}$ ($m$, $n$) oder $m \times n$. Insbesondere, wenn eine Matrix die gleiche Anzahl von Zeilen und Spalten hat, wird ihre Form zu einem Quadrat; daher wird sie als *quadratische Matrix* bezeichnet.

Wir können eine $m \times n$ Matrix erstellen, indem wir eine Form mit zwei Komponenten $m$ und $n$ angeben, wenn eine unserer Lieblingsfunktionen zum Instanziieren eines Tensors aufgerufen wird.

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

Wir können auf das skalare Element $a_{ij}$ einer Matrix $\mathbf{A}$ in :eqref:`eq_matrix_def` zugreifen, indem Sie die Indizes für die Zeile ($i$) und Spalte ($j$), wie $[\mathbf{A}]_{ij}$ angeben. Wenn die Skalarelemente einer Matrix $\mathbf{A}$, wie in :eqref:`eq_matrix_def`, nicht angegeben sind, können wir einfach den Kleinbuchstaben der Matrix $\mathbf{A}$ mit dem Index Index Index, $a_{ij}$, verwenden, um sich auf $[\mathbf{A}]_{ij}$ zu beziehen. Um die Schreibweise einfach zu halten, werden Kommas nur dann eingefügt, wenn nötig, um Indizes zu trennen, z. B. $a_{2, 3j}$ und $[\mathbf{A}]_{2i-1, 3}$.

Manchmal wollen wir die Achsen umdrehen. Wenn wir die Zeilen und Spalten einer Matrix austauschen, wird das Ergebnis die *Transpose* der Matrix genannt. Formal bedeuten wir eine Matrix $\mathbf{A}$ Transponieren durch $\mathbf{A}^\top$ und wenn $\mathbf{B} = \mathbf{A}^\top$, dann $b_{ij} = a_{ji}$ für alle $i$ und $j$. Somit ist die Transponie von $\mathbf{A}$ in :eqref:`eq_matrix_def` eine $n \times m$ Matrix:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Jetzt greifen wir auf die Transponie einer Matrix im Code zu.

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

Als besondere Art der quadratischen Matrix ist eine *symmetrische Matrix* $\mathbf{A}$ gleich ihrer Transponie: $\mathbf{A} = \mathbf{A}^\top$. Hier definieren wir eine symmetrische Matrix `B`.

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

Jetzt vergleichen wir `B` mit seiner Transponie.

```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

Matrizen sind nützliche Datenstrukturen: Sie ermöglichen es uns, Daten zu organisieren, die unterschiedliche Variationsmodalitäten haben. Beispielsweise können Zeilen in unserer Matrix verschiedenen Häusern (Datenpunkten) entsprechen, während Spalten unterschiedlichen Attributen entsprechen können. Dies sollte vertraut klingen, wenn Sie jemals Tabellenkalkulationssoftware verwendet haben oder :numref:`sec_pandas` gelesen haben. Obwohl die Standardausrichtung eines einzelnen Vektors ein Spaltenvektor ist, ist es in einer Matrix, die ein tabellarisches Dataset darstellt, herkömmlicher, jeden Datenpunkt als Zeilenvektor in der Matrix zu behandeln. Und wie wir in späteren Kapiteln sehen werden, wird diese Konvention gemeinsame Deep Learning-Praktiken ermöglichen. Zum Beispiel können wir entlang der äußersten Achse eines Tensors auf Minibatches von Datenpunkten zugreifen oder aufzählen, oder nur Datenpunkte, wenn keine Minibatch vorhanden ist.

## Tensoren

So wie Vektoren Skalare verallgemeinern und Matrizen Vektoren verallgemeinern, können wir Datenstrukturen mit noch mehr Achsen erstellen. Tensoren („Tensoren“ in diesem Unterabschnitt beziehen sich auf algebraische Objekte) geben uns eine generische Möglichkeit, $n$-dimensionale Arrays mit einer beliebigen Anzahl von Achsen zu beschreiben. Vektoren sind beispielsweise Tensoren erster Ordnung, und Matrizen sind Tensoren zweiter Ordnung. Tensoren werden mit Großbuchstaben einer speziellen Schriftart bezeichnet (z. B. $\mathsf{X}$, $\mathsf{Y}$ und $\mathsf{Z}$) und ihr Indizierungsmechanismus (z. B. $x_{ijk}$ und $[\mathsf{X}]_{1, 2i-1, 3}$) ähnelt dem von Matrizen.

Tensoren werden wichtiger, wenn wir mit Bildern arbeiten, die als $n$-dimensionale Arrays mit 3 Achsen ankommen, die der Höhe, Breite und einer *Kanal*-Achse zum Stapeln der Farbkanäle (rot, grün und blau) entsprechen. Im Moment werden wir Tensoren höherer Ordnung überspringen und uns auf die Grundlagen konzentrieren.

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

## Grundlegende Eigenschaften der Tensorarithmetik

Skalare, Vektoren, Matrizen und Tensoren („Tensoren“ in diesem Unterabschnitt beziehen sich auf algebraische Objekte) einer beliebigen Anzahl von Achsen haben einige schöne Eigenschaften, die oft nützlich sind. Angenommen, Sie haben vielleicht aus der Definition einer elementweisen Operation bemerkt, dass jede elementweise unäre Operation die Form ihres Operanden nicht ändert. In ähnlicher Weise wird bei zwei Tensoren mit derselben Form das Ergebnis einer binären elementweisen Operation ein Tensor derselben Form sein. Beispielsweise führt das Hinzufügen von zwei Matrizen derselben Form eine elementweise Addition über diese beiden Matrizen durch.

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

Konkret wird die elementweise Multiplikation zweier Matrizen ihr *Hadamard-Produkt* (mathematische Notation $\odot$) genannt. Betrachten Sie Matrix $\mathbf{B} \in \mathbb{R}^{m \times n}$, deren Element der Zeile $i$ und Spalte $j$ $b_{ij}$ ist. Das Hadamard-Produkt der Matrizen $\mathbf{A}$ (definiert in :eqref:`eq_matrix_def`) und $\mathbf{B}$

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

Das Multiplizieren oder Hinzufügen eines Tensors mit einem Skalar ändert auch nicht die Form des Tensors, wobei jedes Element des Operandentensors hinzugefügt oder mit dem Skalar multipliziert wird.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## Reduktion
:label:`subseq_lin-alg-reduction`

Eine nützliche Operation, die wir mit beliebigen Tensoren durchführen können, ist die Berechnung der Summe ihrer Elemente. In mathematischer Notation drücken wir Summen mit dem $\sum$-Symbol aus. Um die Summe der Elemente in einem Vektor $\mathbf{x}$ der Länge $d$ auszudrücken, schreiben wir $\sum_{i=1}^d x_i$. Im Code können wir einfach die Funktion zur Berechnung der Summe aufrufen.

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

Wir können Summen über die Elemente von Tensoren beliebiger Form ausdrücken. Zum Beispiel könnte die Summe der Elemente einer $m \times n$ Matrix $\mathbf{A}$ geschrieben werden $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

Standardmäßig wird die Funktion zum Berechnen der Summe aufgerufen
*senkt einen Tensor entlang aller Achsen auf einen Skalar.
Wir können auch die Achsen angeben, entlang denen der Tensor durch Summierung reduziert wird. Nehmen Sie Matrizen als Beispiel. Um die Zeilendimension (Achse 0) zu reduzieren, indem Elemente aller Zeilen zusammengefasst werden, geben wir `axis=0` beim Aufruf der Funktion an. Da die Eingabematrix entlang der Achse 0 reduziert wird, um den Ausgabevektor zu generieren, geht die Dimension der Achse 0 der Eingabe im Ausgabe-Shape verloren.

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

Durch die Angabe von `axis=1` wird die Spaltenbemaßung (Achse 1) reduziert, indem Elemente aller Spalten zusammengefasst werden. Somit geht die Dimension der Achse 1 der Eingabe in der Ausgabeform verloren.

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

Das Reduzieren einer Matrix entlang beiden Zeilen und Spalten durch Summierung entspricht dem Zusammenfassen aller Elemente der Matrix.

```{.python .input}
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # Same as `tf.reduce_sum(A)`
```

Eine zugehörige Menge ist die *Mittel*, die auch *Mittel* genannt wird. Wir berechnen den Mittelwert, indem wir die Summe durch die Gesamtzahl der Elemente dividieren. Im Code könnten wir einfach die Funktion aufrufen, um den Mittelwert auf Tensoren beliebiger Form zu berechnen.

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Ebenso kann die Funktion zur Berechnung des Mittelwerts auch einen Tensor entlang der angegebenen Achsen reduzieren.

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

### Summe der nicht reduzierten Summe
:label:`subseq_lin-alg-non-reduction`

Manchmal kann es jedoch nützlich sein, die Anzahl der Achsen unverändert zu halten, wenn die Funktion zum Berechnen der Summe oder des Mittelwerts aufgerufen wird.

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

Zum Beispiel, da `sum_A` immer noch seine beiden Achsen behält, nachdem jede Reihe summiert wurde, können wir `A` durch `sum_A` mit Rundfunk teilen.

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

Wenn wir die kumulative Summe der Elemente von `A` entlang einer Achse berechnen möchten, sagen wir `axis=0` (Zeile für Zeile), können wir die Funktion `cumsum` aufrufen. Diese Funktion reduziert den Eingangstensor entlang einer Achse nicht.

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## Dot Produkte

Bisher haben wir nur elementweise Operationen, Summen und Durchschnittswerte durchgeführt. Und wenn das alles wäre, was wir tun könnten, würde die lineare Algebra wahrscheinlich keinen eigenen Abschnitt verdienen. Eine der grundlegendsten Operationen ist jedoch das Punktprodukt. Bei zwei Vektoren $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ ist ihr *dot product* $\mathbf{x}^\top \mathbf{y}$ (oder $\langle \mathbf{x}, \mathbf{y}  \rangle$) eine Summe über die Produkte der Elemente an derselben Position: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

Beachten Sie, dass wir das Punktprodukt zweier Vektoren gleichwertig ausdrücken können, indem wir eine elementweise Multiplikation und dann eine Summe durchführen:

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

Dot Produkte sind in einer Vielzahl von Kontexten nützlich. Bei einigen Werten, die mit einem Vektor $\mathbf{x}  \in \mathbb{R}^d$ und einem Satz von Gewichten bezeichnet werden, die mit $\mathbf{w} \in \mathbb{R}^d$ bezeichnet werden, könnte die gewichtete Summe der Werte in $\mathbf{x}$ entsprechend den Gewichten $\mathbf{w}$ als Punktprodukt $\mathbf{x}^\top \mathbf{w}$ ausgedrückt werden. Wenn die Gewichtungen nicht negativ sind und auf eins (d. h. $\left(\sum_{i=1}^{d} {w_i} = 1\right)$) summieren, drückt das Punktprodukt einen *gewichteten Durchschnitt* aus. Nachdem zwei Vektoren normalisiert wurden, um die Einheitenlänge zu haben, drücken die Punktprodukte den Kosinus des Winkels zwischen ihnen aus. Wir werden diesen Begriff der *Länge* später in diesem Abschnitt formell einführen.

## Matrix-Vector Produkte

Jetzt, da wir wissen, wie man Punktprodukte berechnet, können wir beginnen, *Matrix-Vektorprodukte* zu verstehen. Erinnern Sie sich an die Matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ und den Vektor $\mathbf{x} \in \mathbb{R}^n$ definiert und visualisiert in :eqref:`eq_matrix_def` und :eqref:`eq_vec_def` bzw.. Beginnen wir mit der Visualisierung der Matrix $\mathbf{A}$ in Bezug auf ihre Zeilenvektoren

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

wobei jeder $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ ein Zeilenvektor ist, der die $i^\mathrm{th}$ Zeile der Matrix $\mathbf{A}$ darstellt. Das Matrix-Vektor-Produkt $\mathbf{A}\mathbf{x}$ ist einfach ein Spaltenvektor der Länge $m$, dessen $i^\mathrm{th}$ Element das Punktprodukt $\mathbf{a}^\top_i \mathbf{x}$ ist:

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Wir können die Multiplikation mit einer Matrix $\mathbf{A}\in \mathbb{R}^{m \times n}$ als Transformation vorstellen, die Vektoren von $\mathbb{R}^{n}$ bis $\mathbb{R}^{m}$ projiziert. Diese Transformationen erweisen sich als bemerkenswert nützlich. Zum Beispiel können wir Rotationen als Multiplikationen mit einer quadratischen Matrix darstellen. Wie wir in den nachfolgenden Kapiteln sehen werden, können wir auch Matrix-Vektorprodukte verwenden, um die intensivsten Berechnungen zu beschreiben, die erforderlich sind, wenn jede Schicht in einem neuronalen Netzwerk unter Berücksichtigung der Werte der vorherigen Schicht berechnet wird.

Wenn wir Matrix-Vektorprodukte im Code mit Tensoren ausdrücken, verwenden wir die gleiche `dot` Funktion wie für Punktprodukte. Wenn wir `np.dot(A, x)` mit einer Matrix `A` und einem Vektor `x` aufrufen, wird das Matrix-Vektor-Produkt durchgeführt. Beachten Sie, dass die Spaltenbemaßung von `A` (Länge entlang Achse 1) mit der Bemaßung von `x` (Länge) übereinstimmen muss.

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## Matrix-Matrix-Multiplikation

Wenn Sie den Hang von Punktprodukten und Matrix-Vektor-Produkten erhalten haben, sollte *Matrix-Matrix-Multiplikation* einfach sein.

Sagen wir, dass wir zwei Matrizen $\mathbf{A} \in \mathbb{R}^{n \times k}$ und $\mathbf{B} \in \mathbb{R}^{k \times m}$ haben:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

Bezeichnen Sie mit $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ den Zeilenvektor, der die $i^\mathrm{th}$ Zeile der Matrix $\mathbf{A}$ darstellt, und lassen Sie $\mathbf{b}_{j} \in \mathbb{R}^k$ der Spaltenvektor aus der $j^\mathrm{th}$ Spalte der Matrix $\mathbf{B}$ sein. Um das Matrixprodukt $\mathbf{C} = \mathbf{A}\mathbf{B}$ herzustellen, ist es am einfachsten, an $\mathbf{A}$ in Bezug auf seine Zeilenvektoren und $\mathbf{B}$ in Bezug auf seine Spaltenvektoren zu denken:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

Dann wird das Matrixprodukt $\mathbf{C} \in \mathbb{R}^{n \times m}$ produziert, da wir einfach jedes Element $c_{ij}$ als Punktprodukt $\mathbf{a}^\top_i \mathbf{b}_j$ berechnen:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

Wir können uns vorstellen, dass die Matrix-Matrix-Multiplikation $\mathbf{AB}$ einfach $m$ Matrix-Vektor-Produkte durchführt und die Ergebnisse zusammenfügt, um eine $n \times m$ Matrix zu bilden. Im folgenden Snippet führen wir Matrixmultiplikation auf `A` und `B` durch. Hier ist `A` eine Matrix mit 5 Zeilen und 4 Spalten, und `B` ist eine Matrix mit 4 Zeilen und 3 Spalten. Nach der Multiplikation erhalten wir eine Matrix mit 5 Zeilen und 3 Spalten.

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

Matrix-Matrix-Multiplikation kann einfach *Matrix-Multiplikation* genannt werden und sollte nicht mit dem Hadamard-Produkt verwechselt werden.

## Normen
:label:`subsec_lin-algebra-norms`

Einige der nützlichsten Operatoren in der linearen Algebra sind *Normen*. Informell sagt uns die Norm eines Vektors, wie *big* ein Vektor ist. Der hier betrachtete Begriff der *Größe* betrifft nicht die Dimensionalität, sondern die Größe der Komponenten.

In der linearen Algebra ist eine Vektornorm eine Funktion $f$, die einen Vektor einem Skalar zuordnet und eine Handvoll Eigenschaften erfüllt. Bei jedem Vektor $\mathbf{x}$ sagt die erste Eigenschaft, dass, wenn wir alle Elemente eines Vektors um einen konstanten Faktor $\alpha$ skalieren, seine Norm auch um den *absoluten Werte* desselben konstanten Faktors skaliert:

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

Die zweite Eigenschaft ist die bekannte Dreiecksungleichheit:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

Die dritte Eigenschaft sagt einfach, dass die Norm nicht negativ sein muss:

$$f(\mathbf{x}) \geq 0.$$

Das macht Sinn, da in den meisten Kontexten die kleinste *Größe* für irgendetwas 0 ist. Die endgültige Eigenschaft erfordert, dass die kleinste Norm erreicht und nur durch einen Vektor erreicht wird, der aus allen Nullen besteht.

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

Sie können feststellen, dass Normen sehr ähnlich wie Entfernungsmaße klingen. Und wenn Sie sich an euklidische Entfernungen (denken Sie an den Satz von Pythagoras) von der Grundschule erinnern, dann könnten die Konzepte der Nicht-Negativität und der Dreiecksungleichheit eine Glocke klingeln. Tatsächlich ist die euklidische Entfernung eine Norm: speziell ist sie die Norm $L_2$. Angenommen, die Elemente im $n$-dimensionalen Vektor $\mathbf{x}$ sind $x_1, \ldots, x_n$. Die $L_2$ *Norm* von $\mathbf{x}$ ist die Quadratwurzel der Summe der Quadrate der Vektorelemente:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$

wobei der Index $2$ häufig in $L_2$ Normen weggelassen wird, d.h. $\|\mathbf{x}\|$ entspricht $\|\mathbf{x}\|_2$. Im Code können wir die Norm $L_2$ eines Vektors wie folgt berechnen.

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

Im Deep Learning arbeiten wir öfter mit der quadratischen Norm $L_2$. Sie werden auch häufig auf die $L_1$ *norm* stoßen, die als Summe der absoluten Werte der Vektorelemente ausgedrückt wird:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

Im Vergleich zur Norm $L_2$ wird sie weniger von Ausreißern beeinflusst. Um die Norm $L_1$ zu berechnen, komponieren wir die Absolutwertfunktion mit einer Summe über die Elemente.

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

Sowohl die Norm $L_2$ als auch die Norm $L_1$ sind Sonderfälle der allgemeineren $L_p$ *Norm*:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Analog zu $L_2$ Normen von Vektoren ist die *Frobenius-Norm* einer Matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ die Quadratwurzel der Summe der Quadrate der Matrixelemente:

$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

Die Frobenius-Norm erfüllt alle Eigenschaften von Vektornormen. Es verhält sich wie eine $L_2$ Norm eines matrixförmigen Vektors. Durch den Aufruf der folgenden Funktion wird die Frobenius-Norm einer Matrix berechnet.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

### Normen und Ziele
:label:`subsec_norms_and_objectives`

Obwohl wir uns nicht allzu weit voraus sein wollen, können wir schon einige Intuition darüber setzen, warum diese Konzepte nützlich sind. Im Deep Learning versuchen wir oft Optimierungsprobleme zu lösen:
*maximieren* die Wahrscheinlichkeit, die den beobachteten Daten zugeordnet ist;
*Minimieren* den Abstand zwischen Vorhersagen
und die Bodenwahrheitsbeobachtungen. Weisen Sie Elementen Vektordarstellungen zu (wie Wörtern, Produkten oder Nachrichtenartikeln) so zu, dass der Abstand zwischen ähnlichen Elementen minimiert wird und der Abstand zwischen unterschiedlichen Elementen maximiert wird. Oft werden die Ziele, vielleicht die wichtigsten Komponenten von Deep Learning-Algorithmen (neben den Daten), als Normen ausgedrückt.

## Mehr zu Linear Algebra

In diesem Abschnitt haben wir Ihnen alle linearen Algebra beigebracht, die Sie benötigen, um einen bemerkenswerten Teil des modernen Deep Learning zu verstehen. Es gibt viel mehr zu linearer Algebra und viel davon ist Mathematik nützlich für maschinelles Lernen. Beispielsweise können Matrizen in Faktoren zerlegt werden, und diese Zersetzungen können eine niedrigdimensionale Struktur in realen Datasets offenbaren. Es gibt ganze Teilfelder des maschinellen Lernens, die sich auf die Verwendung von Matrixzersetzungen und deren Verallgemeinerungen für Tensoren hoher Ordnung konzentrieren, um Struktur in Datensätzen zu erkennen und Vorhersageprobleme zu lösen. Aber dieses Buch konzentriert sich auf Deep Learning. Und wir glauben, dass Sie viel mehr dazu neigen werden, mehr Mathematik zu lernen, sobald Sie Ihre Hände schmutzig gemacht haben, indem Sie nützliche Machine Learning-Modelle auf echten Datensätzen bereitstellen. Während wir uns also das Recht vorbehalten, viel später mehr Mathematik einzuführen, werden wir diesen Abschnitt hier einschließen.

Wenn Sie begierig sind, mehr über lineare Algebra zu erfahren, können Sie entweder auf die [online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html) oder andere ausgezeichnete Ressourcen :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008` beziehen.

## Zusammenfassung

* Skalare, Vektoren, Matrizen und Tensoren sind grundlegende mathematische Objekte in der linearen Algebra.
* Vektoren verallgemeinern Skalare und Matrizen verallgemeinern Vektoren.
* Skalare, Vektoren, Matrizen und Tensoren haben Null, eins, zwei und eine beliebige Anzahl von Achsen.
* Ein Tensor kann entlang der angegebenen Achsen um `sum` und `mean` reduziert werden.
* Elementweise Multiplikation von zwei Matrizen wird ihr Hadamard-Produkt genannt. Es unterscheidet sich von der Matrixmultiplikation.
* Im Deep Learning arbeiten wir oft mit Normen wie der Norm $L_1$, der Norm $L_2$ und der Norm Frobenius.
* Wir können eine Vielzahl von Operationen über Skalare, Vektoren, Matrizen und Tensoren durchführen.

## Übungen

1. Beweisen Sie, dass die Transponie einer Matrix $\mathbf{A}$ $\mathbf{A}$:$(\mathbf{A}^\top)^\top = \mathbf{A}$ ist.
1. Da zwei Matrizen $\mathbf{A}$ und $\mathbf{B}$, zeigen, dass die Summe der Transponieren gleich der Transponieren einer Summe ist: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Angesichts jeder quadratischen Matrix $\mathbf{A}$, ist $\mathbf{A} + \mathbf{A}^\top$ immer symmetrisch? Warum?
1. In diesem Abschnitt haben wir den Tensor `X` der Form (2, 3, 4) definiert. Was ist die Ausgabe von `len(X)`?
1. Entspricht für einen Tensor `X` beliebiger Form `len(X)` immer der Länge einer bestimmten Achse von `X`? Was ist das für eine Achse?
1. Führen Sie `A / A.sum(axis=1)` aus und sehen Sie, was passiert. Können Sie den Grund analysieren?
1. Wenn Sie zwischen zwei Punkten in Manhattan reisen, was ist die Entfernung, die Sie in Bezug auf die Koordinaten abdecken müssen, d. h. in Bezug auf Alleen und Straßen? Kannst du schräg reisen?
1. Betrachten Sie einen Tensor mit Form (2, 3, 4). Was sind die Formen der Summenausgaben entlang der Achse 0, 1 und 2?
1. Führen Sie einen Tensor mit 3 oder mehr Achsen in die Funktion `linalg.norm` ein und beobachten Sie dessen Ausgang. Was berechnet diese Funktion für Tensoren beliebiger Form?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:
