# Manipulation von Daten
:label:`sec_ndarray`

Um etwas zu erledigen, brauchen wir eine Möglichkeit, Daten zu speichern und zu manipulieren. Im Allgemeinen gibt es zwei wichtige Dinge, die wir mit Daten tun müssen: (i) sie erfassen; und (ii) verarbeiten sie, sobald sie sich im Computer befinden. Es hat keinen Sinn, Daten ohne irgendeine Möglichkeit zu speichern, also lassen Sie uns zuerst die Hände schmutzig machen, indem wir mit synthetischen Daten spielen. Zunächst stellen wir das $n$-dimensionale Array vor, das auch *tensor* genannt wird.

Wenn Sie mit NumPy gearbeitet haben, dem am weitesten verbreiteten wissenschaftlichen Computing-Paket in Python, dann werden Sie diesen Abschnitt vertraut finden. Egal welches Framework Sie verwenden, seine *tensor-Klasse* (`ndarray` in MxNet, `Tensor` in PyTorch und TensorFlow) ähnelt NumPys `ndarray` mit einigen Killer-Funktionen. Erstens wird GPU gut unterstützt, um die Berechnung zu beschleunigen, während NumPy nur CPU-Berechnung unterstützt. Zweitens unterstützt die Tensor-Klasse die automatische Differenzierung. Diese Eigenschaften machen die Tensor-Klasse für Deep Learning geeignet. Wenn wir im gesamten Buch Tensoren sagen, beziehen wir uns auf Instanzen der Tensor-Klasse, sofern nicht anders angegeben.

## Erste Schritte

In diesem Abschnitt möchten wir Sie zum Laufen bringen und Sie mit den grundlegenden mathematischen und numerischen Rechenwerkzeugen ausstatten, auf denen Sie aufbauen werden, während Sie durch das Buch fortschreiten. Mach dir keine Sorgen, wenn du Probleme hast, einige der mathematischen Konzepte oder Bibliotheksfunktionen zu grok zu machen. In den folgenden Abschnitten wird dieses Material im Kontext von praktischen Beispielen erneut behandelt und es wird sinken. Auf der anderen Seite, wenn Sie bereits einen Hintergrund haben und tiefer in den mathematischen Inhalt gehen wollen, überspringen Sie einfach diesen Abschnitt.

:begin_tab:`mxnet`
Zunächst importieren wir die Module `np` (`numpy`) und `npx` (`numpy_extension`) aus MXNet. Hier enthält das Modul `np` Funktionen, die von NumPy unterstützt werden, während das Modul `npx` eine Reihe von Erweiterungen enthält, die entwickelt wurden, um Deep Learning in einer numPy-ähnlichen Umgebung zu ermöglichen. Bei der Verwendung von Tensoren rufen wir fast immer die Funktion `set_np` auf: Dies dient der Kompatibilität der Tensor-Verarbeitung durch andere Komponenten von MxNet.
:end_tab:

:begin_tab:`pytorch`
Um zu beginnen, importieren wir `torch`. Beachten Sie, dass, obwohl es PyTorch genannt wird, wir `torch` anstelle von `pytorch` importieren sollten.
:end_tab:

:begin_tab:`tensorflow`
Um zu beginnen, importieren wir `tensorflow`. Da der Name ein wenig lang ist, importieren wir ihn oft mit einem kurzen Alias `tf`.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

Ein Tensor stellt ein (möglicherweise mehrdimensionales) Array von numerischen Werten dar. Bei einer Achse entspricht ein Tensor (in Mathematik) einem *Vektor*. Bei zwei Achsen entspricht ein Tensor einer *Matrix*. Tensoren mit mehr als zwei Achsen haben keine speziellen mathematischen Namen.

Um zu beginnen, können wir `arange` verwenden, um einen Zeilenvektor `x` zu erstellen, der die ersten 12 Ganzzahlen beginnend mit 0 enthält, obwohl sie standardmäßig als Floats erstellt werden. Jeder der Werte in einem Tensor wird ein *Element* des Tensors genannt. Zum Beispiel gibt es 12 Elemente im Tensor `x`. Sofern nicht anders angegeben, wird ein neuer Tensor im Hauptspeicher gespeichert und für die CPU-basierte Berechnung bestimmt.

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

Wir können auf die *Form* eines Tensors zugreifen (die Länge entlang jeder Achse), indem wir seine `shape` Eigenschaft inspizieren.

```{.python .input}
#@tab all
x.shape
```

Wenn wir nur die Gesamtzahl der Elemente in einem Tensor wissen wollen, das heißt, das Produkt aller Formelemente, können wir seine Größe überprüfen. Da wir es hier mit einem Vektor zu tun haben, ist das einzelne Element seines `shape` identisch mit seiner Größe.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

Um die Form eines Tensors zu ändern, ohne entweder die Anzahl der Elemente oder deren Werte zu verändern, können wir die Funktion `reshape` aufrufen. Zum Beispiel können wir unseren Tensor, `x`, von einem Zeilenvektor mit Form (12,) in eine Matrix mit Form (3, 4) transformieren. Dieser neue Tensor enthält die exakt gleichen Werte, sieht sie aber als Matrix aus, die als 3 Zeilen und 4 Spalten organisiert ist. Um zu wiederholen, obwohl sich die Form geändert hat, haben die Elemente in `x` nicht. Beachten Sie, dass die Größe durch Umformen unverändert bleibt.

```{.python .input}
#@tab mxnet, pytorch
x = x.reshape(3, 4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.reshape(x, (3, 4))
x
```

Eine Umformung durch manuelles Angeben jeder Dimension ist nicht erforderlich. Wenn unsere Zielform eine Matrix mit Form (Höhe, Breite) ist, dann wird die Höhe implizit angegeben, nachdem wir die Breite kennen. Warum sollten wir die Division selbst durchführen müssen? Im obigen Beispiel, um eine Matrix mit 3 Zeilen zu erhalten, haben wir beide angegeben, dass es 3 Zeilen und 4 Spalten haben sollte. Glücklicherweise können Tensoren bei der restlichen Bemaßung automatisch eine Dimension ausarbeiten. Wir rufen diese Fähigkeit auf, indem wir `-1` für die Dimension platzieren, die Tensoren automatisch ableiten sollen. In unserem Fall hätten wir anstelle von `x.reshape(3, 4)` gleichwertig `x.reshape(-1, 4)` oder `x.reshape(3, -1)` genannt werden können.

In der Regel möchten wir, dass unsere Matrizen entweder mit Nullen, Einsen, einigen anderen Konstanten oder Zahlen initialisiert werden, die zufällig aus einer bestimmten Verteilung abgetastet werden. Wir können einen Tensor, der einen Tensor darstellt, mit allen Elementen auf 0 und eine Form von (2, 3, 4) wie folgt erstellen:

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros(2, 3, 4)
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

In ähnlicher Weise können wir Tensoren mit jedem Element auf 1 wie folgt erstellen:

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

Oft möchten wir die Werte für jedes Element in einem Tensor nach dem Zufallsprinzip anhand einer Wahrscheinlichkeitsverteilung abtasten. Wenn wir beispielsweise Arrays konstruieren, um als Parameter in einem neuronalen Netzwerk zu dienen, werden wir normalerweise ihre Werte zufällig initialisieren. Das folgende Snippet erstellt einen Tensor mit Form (3, 4). Jedes seiner Elemente wird nach dem Zufallsprinzip aus einer standardmäßigen Gaußschen (Normalverteilung) Verteilung mit einem Mittelwert von 0 und einer Standardabweichung von 1.

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

Wir können auch die genauen Werte für jedes Element im gewünschten Tensor angeben, indem wir eine Python-Liste (oder Liste von Listen) mit den numerischen Werten liefern. Hier entspricht die äußerste Liste der Achse 0 und die innere Liste der Achse 1.

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Operationen

In diesem Buch geht es nicht um Software-Engineering. Unsere Interessen beschränken sich nicht darauf, einfach Daten von/zu Arrays zu lesen und zu schreiben. Wir wollen mathematische Operationen auf diesen Arrays durchführen. Einige der einfachsten und nützlichsten Operationen sind die *elementwise* Operationen. Diese wenden eine standardmäßige Skalaroperation auf jedes Element eines Arrays an. Für Funktionen, die zwei Arrays als Eingaben verwenden, wenden elementweise Operationen einen standardmäßigen binären Operator auf jedes Paar von entsprechenden Elementen aus den beiden Arrays an. Wir können eine elementweise Funktion aus jeder Funktion erstellen, die von einem Skalar zu einem Skalar abbildet.

In der mathematischen Notation würden wir einen solchen *unary* skalaren Operator (wobei eine Eingabe erfolgt) durch die Signatur $f: \mathbb{R} \rightarrow \mathbb{R}$ bezeichnen. Dies bedeutet nur, dass die Funktion von jeder reellen Zahl ($\mathbb{R}$) auf eine andere abbildet. Ebenso bezeichnen wir einen *binär* skalaren Operator (wobei zwei reale Eingaben verwendet werden und eine Ausgabe ergibt) durch die Signatur $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$. Angesichts von zwei Vektoren $\mathbf{u}$ und $\mathbf{v}$ *gleicher Form* und einem binären Operator $f$ können wir einen Vektor $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ erzeugen, indem wir $c_i \gets f(u_i, v_i)$ für alle $i$ einstellen, wobei $c_i, u_i$ und $v_i$ die $i^\mathrm{th}$ Elemente der Vektoren $\mathbf{c}, \mathbf{u}$ und $\mathbf{v}$ sind. Hier haben wir den Vektorwert $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ erzeugt, indem wir die Skalarfunktion zu einer elementweisen Vektoroperation heben*.

Die üblichen Standard-arithmetischen Operatoren (`+`, `-`, `*`, `/` und `**`) wurden alle zu elementweisen Operationen für identisch geformte Tensoren beliebiger Form angehoben. Wir können elementweise Operationen an zwei zwei Tensoren derselben Form aufrufen. Im folgenden Beispiel verwenden wir Kommas, um ein 5-Element-Tupel zu formulieren, wobei jedes Element das Ergebnis einer elementweisen Operation ist.

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

Viele weitere Operationen können elementweise angewendet werden, einschließlich unärer Operatoren wie Exponentiation.

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

Neben elementweisen Berechnungen können wir auch lineare Algebraoperationen durchführen, einschließlich Vektorpunktprodukte und Matrixmultiplikation. Wir werden die entscheidenden Bits der linearen Algebra (ohne angenommene Vorkenntnisse) in :numref:`sec_linear-algebra` erklären.

Wir können auch mehrere Tensoren miteinander verketten, indem wir sie Ende-zu-Ende stapeln, um einen größeren Tensor zu bilden. Wir müssen nur eine Liste von Tensoren bereitstellen und dem System mitteilen, entlang welcher Achse verkettet werden soll. Das folgende Beispiel zeigt, was passiert, wenn wir zwei Matrizen entlang der Zeilen (Achse 0, das erste Element der Form) im Vergleich zu Spalten (Achse 1, das zweite Element der Form) verketten. Wir können sehen, dass die Achsen-0-Länge des ersten Ausgangstensors ($6$) die Summe der Achsen-0-Längen der beiden Eingangstensoren ist ($3 + 3$); während die Achse-1-Länge des zweiten Ausgangstensors ($8$) die Summe der Achsen-1-Längen der beiden Eingangstensoren ist ($4 + 4$).

```{.python .input}
x = np.arange(12).reshape(3, 4)
y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([x, y], axis=0), np.concatenate([x, y], axis=1)
```

```{.python .input}
#@tab pytorch
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)
```

```{.python .input}
#@tab tensorflow
x = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([x, y], axis=0), tf.concat([x, y], axis=1)
```

Manchmal wollen wir einen binären Tensor über *logische Anweisungen* konstruieren. Nehmen Sie `x == y` als Beispiel. Wenn für jede Position `x` und `y` an dieser Position gleich sind, nimmt der entsprechende Eintrag im neuen Tensor den Wert 1 an, was bedeutet, dass die logische Aussage `x == y` an dieser Position wahr ist; andernfalls nimmt diese Position 0 an.

```{.python .input}
#@tab all
x == y
```

Die Summe aller Elemente im Tensor ergibt einen Tensor mit nur einem Element.

```{.python .input}
#@tab mxnet, pytorch
x.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x)
```

## Rundfunkmechanismus
:label:`subsec_broadcasting`

Im obigen Abschnitt haben wir gesehen, wie man elementweise Operationen an zwei Tensoren derselben Form durchführt. Unter bestimmten Bedingungen, auch wenn Formen unterschiedlich sind, können wir immer noch elementweise Operationen durchführen, indem wir den *Rundfunkmechanismus* aufrufen. Dieser Mechanismus funktioniert wie folgt: Erweitern Sie zunächst ein oder beide Arrays, indem Sie Elemente entsprechend kopieren, so dass die beiden Tensoren nach dieser Transformation die gleiche Form haben. Zweitens führen Sie die elementweisen Operationen auf die resultierenden Arrays aus.

In den meisten Fällen senden wir entlang einer Achse, auf der ein Array zunächst nur die Länge 1 hat, wie im folgenden Beispiel:

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

Da `a` und `b` $3\times1$ bzw. $1\times2$ Matrizen sind, stimmen ihre Formen nicht überein, wenn wir sie hinzufügen möchten. Wir *übertragen* die Einträge beider Matrizen in eine größere $3\times2$ Matrix wie folgt: Für die Matrix `a` repliziert sie die Spalten und für die Matrix `b` repliziert sie die Zeilen, bevor sie beide elementweise addieren.

```{.python .input}
#@tab all
a + b
```

## Indizierung und Slicing

Wie in jedem anderen Python-Array kann auf Elemente in einem Tensor per Index zugegriffen werden. Wie in jedem Python-Array hat das erste Element den Index 0 und Bereiche werden angegeben, um das erste, aber *vor* das letzte Element einzuschließen. Wie in Standard-Python-Listen, können wir Elemente entsprechend ihrer relativen Position zum Ende der Liste zugreifen, indem wir negative Indizes verwenden.

Daher wählt `[-1]` das letzte Element aus und `[1:3]` wählt das zweite und das dritte Element wie folgt aus:

```{.python .input}
#@tab all
x[-1], x[1:3]
```

:begin_tab:`mxnet, pytorch`
Über das Lesen hinaus können wir auch Elemente einer Matrix schreiben, indem wir Indizes angeben.
:end_tab:

:begin_tab:`tensorflow`
`Tensors` in TensorFlow sind unveränderlich und können nicht zugewiesen werden. `Variables` in TensorFlow sind veränderbare Zustandscontainer, die Zuweisungen unterstützen. Beachten Sie, dass Verläufe in TensorFlow nicht rückwärts durch `Variable` Zuweisungen fließen.

Über die Zuweisung eines Wertes auf die gesamte `Variable` hinaus können wir Elemente eines `Variable` durch Angabe von Indizes schreiben.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
x[1, 2] = 9
x
```

```{.python .input}
#@tab tensorflow
x_var = tf.Variable(x)
x_var[1, 2].assign(9)
x_var
```

Wenn wir mehreren Elementen den gleichen Wert zuweisen möchten, indizieren wir einfach alle von ihnen und weisen ihnen dann den Wert zu. Zum Beispiel greift `[0:2, :]` auf die erste und zweite Zeile zu, wobei `:` alle Elemente entlang Achse 1 (Spalte) nimmt. Während wir die Indizierung für Matrizen besprochen haben, funktioniert dies natürlich auch für Vektoren und für Tensoren von mehr als 2 Dimensionen.

```{.python .input}
#@tab mxnet, pytorch
x[0:2, :] = 12
x
```

```{.python .input}
#@tab tensorflow
x_var = tf.Variable(x)
x_var[0:2,:].assign(tf.ones(x_var[0:2,:].shape, dtype = tf.float32)*12)
x_var
```

## Speicherkapazität speichern

Ausführen von Vorgängen kann dazu führen, dass neuen Speicher den Host-Ergebnissen zugewiesen wird. Wenn wir beispielsweise `y = x + y` schreiben, werden wir den Tensor dereferenzieren, den `y` verwendet hat, um auf den neu zugewiesenen Speicher zu verweisen und stattdessen `y` zu verweisen. Im folgenden Beispiel demonstrieren wir dies mit Pythons `id()`-Funktion, die uns die genaue Adresse des referenzierten Objekts im Speicher gibt. Nach dem Ausführen von `y = y + x` werden wir feststellen, dass `id(y)` auf einen anderen Standort zeigt. Das liegt daran, dass Python zuerst `y + x` auswertet, neuen Speicher für das Ergebnis zuweist und dann `y` auf diesen neuen Speicherort im Speicher verweist.

```{.python .input}
#@tab all
before = id(y)
y = y + x
id(y) == before
```

Dies kann aus zwei Gründen unerwünscht sein. Erstens wollen wir nicht ständig Speicher unnötig zuweisen. Beim maschinellen Lernen haben wir möglicherweise Hunderte von Megabyte an Parametern und aktualisieren sie alle mehrmals pro Sekunde. In der Regel möchten wir diese Aktualisierungen*an Ort und Stelle durchführen*. Zweitens könnten wir auf die gleichen Parameter aus mehreren Variablen zeigen. Wenn wir nicht an Ort und Stelle aktualisieren, zeigen andere Referenzen immer noch auf den alten Speicherort, was es ermöglicht, dass Teile unseres Codes versehentlich veraltete Parameter referenzieren.

:begin_tab:`mxnet, pytorch`
Glücklicherweise ist die Durchführung von In-Place-Operationen einfach. Wir können das Ergebnis einer Operation einem zuvor zugewiesenen Array mit Slice-Notation zuweisen, z. B. `y[:] = <expression>`. Um dieses Konzept zu veranschaulichen, erstellen wir zunächst eine neue Matrix `z` mit der gleichen Form wie ein anderer `y`, mit `zeros_like`, um einen Block von $0$ Einträgen zuzuweisen.
:end_tab:

:begin_tab:`tensorflow`
`Variables` sind veränderbare Container des Zustands in TensorFlow. Sie bieten eine Möglichkeit, Ihre Modellparameter zu speichern. Wir können das Ergebnis einer Operation einem `Variable` mit `assign` zuweisen. Um dieses Konzept zu veranschaulichen, erstellen wir einen `Variable` `z` mit der gleichen Form wie ein anderer Tensor `y`, wobei `zeros_like` verwendet wird, um einen Block von $0$ Einträgen zuzuweisen.
:end_tab:

```{.python .input}
z = np.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab pytorch
z = torch.zeros_like(y)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

```{.python .input}
#@tab tensorflow
z = tf.Variable(tf.zeros_like(y))
print('id(z):', id(z))
z.assign(x + y)
print('id(z):', id(z))
```

:begin_tab:`mxnet, pytorch`
Wenn der Wert von `x` bei nachfolgenden Berechnungen nicht wiederverwendet wird, können wir auch `x[:] = x + y` oder `x += y` verwenden, um den Speicheraufwand des Vorgangs zu reduzieren.
:end_tab:

:begin_tab:`tensorflow`
Selbst wenn Sie den Status dauerhaft in einem `Variable` speichern, können Sie Ihre Speicherauslastung weiter reduzieren, indem Sie überschüssige Zuweisungen für Tensoren vermeiden, die nicht Ihre Modellparameter sind.

Da TensorFlow `Tensors` unveränderlich sind und Verläufe nicht durch `Variable`-Zuweisungen fließen, bietet TensorFlow keine explizite Möglichkeit, einen einzelnen Vorgang direkt auszuführen.

TensorFlow stellt jedoch den `tf.function` Dekorator bereit, um die Berechnung innerhalb eines TensorFlow-Graphen zu umbrechen, das vor der Ausführung kompiliert und optimiert wird. Auf diese Weise kann TensorFlow ungenutzte Werte beschneiden und vorherige Zuordnungen wiederverwenden, die nicht mehr benötigt werden. Dadurch wird der Speicheraufwand von TensorFlow-Berechnungen minimiert.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(x)
x += y
id(x) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(x, y):
  z = tf.zeros_like(y)  # This unused value will be pruned out.
  a = x + y  # Allocations will be re-used when no longer needed.
  b = a + y
  c = b + y
  return c + y

computation(x, y)
```

## Konvertierung in andere Python-Objekte

Die Umwandlung in einen NumPy Tensor oder umgekehrt ist einfach. Das konvertierte Ergebnis teilt keinen Speicher. Diese kleine Unannehmlichkeit ist eigentlich ziemlich wichtig: Wenn Sie Operationen auf der CPU oder auf GPUs ausführen, möchten Sie die Berechnung nicht stoppen und warten, um zu sehen, ob das NumPy-Paket von Python etwas anderes mit dem gleichen Speicherblock tun möchte.

```{.python .input}
a = x.asnumpy()
b = np.array(a)
type(a), type(b)
```

```{.python .input}
#@tab pytorch
a = x.numpy()
b = torch.tensor(a)
type(a), type(b)
```

```{.python .input}
#@tab tensorflow
a = x.numpy()
b = tf.constant(a)
type(a), type(b)
```

Um einen Tensor der Größe 1 in einen Python-Skalar zu konvertieren, können wir die Funktion `item` oder die integrierten Funktionen von Python aufrufen.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## Zusammenfassung

* Die Hauptschnittstelle zum Speichern und Bearbeiten von Daten für Deep Learning ist der Tensor ($n$-dimensionales Array). Es bietet eine Vielzahl von Funktionalitäten, einschließlich grundlegende mathematische Operationen, Rundfunk, Indizierung, Slicing, Speicherspeicherung und Konvertierung in andere Python-Objekte.

## Übungen

1. Führen Sie den Code in diesem Abschnitt aus. Ändern Sie die bedingte Anweisung `x == y` in diesem Abschnitt auf `x < y` oder `x > y`, und sehen Sie dann, welche Art von Tensor Sie erhalten können.
1. Ersetzen Sie die beiden Tensoren, die nach Element im Rundfunkmechanismus arbeiten, durch andere Formen, z.B. dreidimensionale Tensoren. Ist das Ergebnis das gleiche wie erwartet?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
