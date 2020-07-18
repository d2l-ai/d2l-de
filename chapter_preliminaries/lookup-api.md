# Dokumentation

Aufgrund von Einschränkungen in der Länge dieses Buches können wir möglicherweise nicht jede einzelne MxNet-Funktion und -Klasse einführen (und Sie würden es wahrscheinlich nicht wünschen). Die API-Dokumentation und zusätzliche Tutorials und Beispiele bieten viel Dokumentation über das Buch hinaus. In diesem Abschnitt geben wir Ihnen einige Hinweise zur Erkundung der MXNet-API.

## Suchen aller Funktionen und Klassen in einem Modul

Um zu wissen, welche Funktionen und Klassen in einem Modul aufgerufen werden können, rufen wir die Funktion `dir` auf. Zum Beispiel können wir alle Eigenschaften im Modul abfragen, um Zufallszahlen zu generieren:

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

Im Allgemeinen können wir Funktionen ignorieren, die mit `__` beginnen und enden (spezielle Objekte in Python) oder Funktionen, die mit einem einzigen `_` beginnen (normalerweise interne Funktionen). Basierend auf den verbleibenden Funktions- oder Attributnamen könnten wir eine Vermutung riskieren, dass dieses Modul verschiedene Methoden zur Generierung von Zufallszahlen bietet, einschließlich Stichproben aus der einheitlichen Verteilung (`uniform`), der Normalverteilung (`normal`) und der multinomialen Verteilung (`multinomial`).

## Suche nach der Verwendung bestimmter Funktionen und Klassen

Für genauere Anweisungen zur Verwendung einer bestimmten Funktion oder Klasse, können wir die `help` Funktion aufrufen. Als Beispiel, lassen Sie uns die Gebrauchsanweisungen für Tensors `ones` Funktion untersuchen.

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

Aus der Dokumentation können wir sehen, dass die Funktion `ones` einen neuen Tensor mit der angegebenen Form erstellt und alle Elemente auf den Wert 1 setzt. Wann immer möglich, sollten Sie einen Schnelltest durchführen, um Ihre Interpretation zu bestätigen:

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

Im Jupyter Notizbuch können wir `verwenden?`, um das Dokument in einem anderen Fenster anzuzeigen. Zum Beispiel, `list?`erstellt Inhalte, die fast identisch mit `help(list)` sind und in einem neuen Browserfenster angezeigt werden. Darüber hinaus, wenn wir zwei Fragezeichen verwenden, wie `list??`, wird auch der Python-Code angezeigt, der die Funktion implementiert.

## Zusammenfassung

* Die offizielle Dokumentation enthält viele Beschreibungen und Beispiele, die über dieses Buch hinausgehen.
* Wir können die Dokumentation für die Verwendung einer API nachschlagen, indem wir die Funktionen `dir` und `help` aufrufen, oder `?` and `??`in Jupyter Notizbüchern.

## Übungen

1. Suchen Sie nach der Dokumentation für jede Funktion oder Klasse im Deep Learning-Framework. Können Sie die Dokumentation auch auf der offiziellen Website des Frameworks finden?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:
