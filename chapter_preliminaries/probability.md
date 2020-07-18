# Wahrscheinlichkeit
:label:`sec_prob`

In irgendeiner Form geht es beim maschinellen Lernen darum, Vorhersagen zu machen. Vielleicht möchten wir die Wahrscheinlichkeit eines Patienten voraussagen, der im nächsten Jahr einen Herzinfarkt erlitten hat, angesichts seiner klinischen Anamnese. Bei der Anomalieerkennung möchten wir vielleicht beurteilen, wie *wahrscheinlich* eine Reihe von Messwerten aus dem Jetmotor eines Flugzeugs wäre, wenn es normal funktioniert. Im verstärkten Lernen wollen wir, dass ein Agent in einer Umgebung intelligent agiert. Dies bedeutet, dass wir über die Wahrscheinlichkeit nachdenken müssen, dass bei jeder der verfügbaren Aktionen eine hohe Belohnung erhalten. Und wenn wir Empfehlungssysteme bauen, müssen wir auch über die Wahrscheinlichkeit nachdenken. Sagen Sie zum Beispiel *hypothetisch*, dass wir für einen großen Online-Buchhändler gearbeitet haben. Vielleicht möchten wir die Wahrscheinlichkeit schätzen, dass ein bestimmter Benutzer ein bestimmtes Buch kaufen würde. Dazu müssen wir die Sprache der Wahrscheinlichkeit verwenden. Ganze Kurse, Majors, Abschlussarbeiten, Karrieren und sogar Abteilungen sind der Wahrscheinlichkeit gewidmet. Unser Ziel in diesem Abschnitt ist also natürlich nicht, das ganze Thema zu unterrichten. Stattdessen hoffen wir, Sie vom Boden zu bekommen, Ihnen gerade genug beizubringen, dass Sie mit dem Aufbau Ihrer ersten Deep Learning-Modelle beginnen können, und Ihnen genug Geschmack für das Thema zu geben, dass Sie beginnen können, es auf eigene Faust zu erforschen, wenn Sie möchten.

Wir haben bereits in früheren Abschnitten Wahrscheinlichkeiten angeführt, ohne zu artikulieren, was genau sie sind oder ein konkretes Beispiel geben zu müssen. Lassen Sie uns jetzt ernster werden, indem wir den ersten Fall betrachten: Katzen und Hunde anhand von Fotografien unterscheiden. Das mag einfach klingen, aber es ist eigentlich eine gewaltige Herausforderung. Zunächst kann die Schwierigkeit des Problems von der Auflösung des Bildes abhängen.

![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat_dog_pixels.png)
:width:`300px`
:label:`fig_cat_dog`

Wie in :numref:`fig_cat_dog` gezeigt, ist es für Menschen leicht, Katzen und Hunde mit einer Auflösung von $160 \times 160$ Pixeln zu erkennen, es wird mit $40 \times 40$ Pixeln eine Herausforderung und mit $10 \times 10$ Pixeln fast unmöglich. Mit anderen Worten, unsere Fähigkeit, Katzen und Hunde in großer Entfernung (und damit in geringer Auflösung) auseinander zu sagen, könnte sich uninformiertes Erraten nähern. Die Wahrscheinlichkeit gibt uns einen formalen Weg, über unser Maß an Sicherheit zu denken. Wenn wir völlig sicher sind, dass das Bild eine Katze darstellt, sagen wir, dass die *Wahrscheinlichkeit*, dass das entsprechende Etikett $y$ „Katze“ ist, bezeichnet $P(y=$ „Katze“ $)$ entspricht $1$. Wenn wir keine Beweise dafür hätten, dass $y =$ „Katze“ oder dass $y =$ „Hund“, dann könnten wir sagen, dass die beiden Möglichkeiten gleich waren
*wahrscheinlich* ausgedrückt dies als $P(y=$ „cat"$) = P(y=$ „dog"$) = 0.5$. Wenn wir vernünftigerweise
zuversichtlich, aber nicht sicher, dass das Bild dargestellt eine Katze, könnten wir eine Wahrscheinlichkeit zuweisen $0.5  < P(y=$ „cat"$) < 1$.

Betrachten Sie nun den zweiten Fall: Angesichts einiger Wetterüberwachungsdaten wollen wir die Wahrscheinlichkeit vorhersagen, dass es morgen in Taipeh regnen wird. Wenn es im Sommer ist, könnte der Regen mit der Wahrscheinlichkeit 0,5 kommen.

In beiden Fällen haben wir einen gewissen Wert von Interesse. Und in beiden Fällen sind wir unsicher über das Ergebnis. Aber es gibt einen entscheidenden Unterschied zwischen den beiden Fällen. In diesem ersten Fall ist das Bild in der Tat entweder ein Hund oder eine Katze, und wir wissen einfach nicht, welche. Im zweiten Fall kann das Ergebnis tatsächlich ein zufälliges Ereignis sein, wenn Sie an solche Dinge glauben (und die meisten Physiker tun). Daher ist die Wahrscheinlichkeit eine flexible Sprache, um über unser Maß an Sicherheit nachzudenken, und sie kann effektiv in einem breiten Spektrum von Kontexten angewendet werden.

## Grundlegende Wahrscheinlichkeitstheorie

Sagen Sie, wir werfen einen Würfel und wollen wissen, was die Chance ist, eine 1 statt eine andere Ziffer zu sehen. Wenn der Würfel fair ist, werden alle sechs Ergebnisse $\{1, \ldots, 6\}$ gleichermaßen wahrscheinlich auftreten, und so würden wir einen $1$ in einem von sechs Fällen sehen. Formal sagen wir, dass $1$ mit Wahrscheinlichkeit $\frac{1}{6}$ auftritt.

Für einen echten Würfel, den wir von einer Fabrik erhalten, kennen wir diese Proportionen vielleicht nicht und wir müssten prüfen, ob er verdorben ist. Die einzige Möglichkeit, den Würfel zu untersuchen, besteht darin, ihn mehrmals zu werfen und die Ergebnisse zu erfassen. Für jeden Stempelguss werden wir einen Wert in $\{1, \ldots, 6\}$ beobachten. Angesichts dieser Ergebnisse wollen wir die Wahrscheinlichkeit untersuchen, jedes Ergebnis zu beobachten.

Ein natürlicher Ansatz für jeden Wert besteht darin, die individuelle Zählung für diesen Wert zu nehmen und ihn durch die Gesamtzahl der Wurfe zu dividieren. Dies gibt uns eine *Schätzung* für die Wahrscheinlichkeit eines gegebenen *Ereignis*. Das *Gesetz der großen Zahlen* sagt uns, dass diese Schätzung, wenn die Anzahl der Werfe wächst, näher und näher an die wahre zugrunde liegende Wahrscheinlichkeit heranrückt. Bevor wir uns mit den Details dessen befassen, was hier vor sich geht, lassen Sie es uns ausprobieren.

Um zu beginnen, lassen Sie uns die notwendigen Pakete importieren.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions import multinomial
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
```

Als nächstes wollen wir in der Lage sein, den Würfel zu werfen. In der Statistik nennen wir diesen Prozess der Zeichnung von Beispielen aus Wahrscheinlichkeitsverteilungen *Sampling*. Die Verteilung, die Wahrscheinlichkeiten einer Reihe von diskreten Auswahlmöglichkeiten zuweist, wird als
*multinomiale Verteilung*. Wir geben eine formellere Definition von
*Verteilung* später, aber auf hohem Niveau, denken Sie es als nur eine Zuweisung von
Wahrscheinlichkeiten zu Ereignissen.

Um eine einzelne Stichprobe zu zeichnen, übergeben wir einfach einen Vektor von Wahrscheinlichkeiten. Die Ausgabe ist ein weiterer Vektor gleicher Länge: ihr Wert bei Index $i$ entspricht der Häufigkeit, in der das Sampling-Ergebnis $i$ entspricht.

```{.python .input}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

```{.python .input}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfp.distributions.Multinomial(1, fair_probs).sample()
```

Wenn Sie den Sampler mehrmals ausführen, werden Sie feststellen, dass Sie jedes Mal zufällige Werte erhalten. Wie bei der Schätzung der Fairness eines Düsens wollen wir oft viele Stichproben aus der gleichen Verteilung generieren. Es wäre unerträglich langsam, dies mit einer Python `for`-Schleife zu tun, so dass die Funktion, die wir verwenden, unterstützt das Zeichnen mehrerer Samples auf einmal und gibt ein Array von unabhängigen Samples in jeder gewünschten Form zurück.

```{.python .input}
np.random.multinomial(10, fair_probs)
```

```{.python .input}
#@tab pytorch
multinomial.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfp.distributions.Multinomial(10, fair_probs).sample()
```

Jetzt, da wir wissen, wie man Rollen einer Matrize probiert, können wir 1000 Rollen simulieren. Wir können dann durchgehen und zählen, nach jedem der 1000 Rollen, wie oft jede Zahl gerollt wurde. Konkret berechnen wir die relative Häufigkeit als Schätzung der wahren Wahrscheinlichkeit.

```{.python .input}
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input}
#@tab pytorch
# Store the results as 32-bit floats for division
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # Relative frequency as the estimate
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
counts / 1000
```

Da wir die Daten aus einem fairen sterben generiert haben, wissen wir, dass jedes Ergebnis eine echte Wahrscheinlichkeit $\frac{1}{6}$ hat, ungefähr $0.167$, so dass die obigen Ausgabeschätzungen gut aussehen.

Wir können auch visualisieren, wie diese Wahrscheinlichkeiten im Laufe der Zeit in Richtung der wahren Wahrscheinlichkeit konvergieren. Lassen Sie uns 500 Gruppen von Experimenten durchführen, in denen jede Gruppe 10 Proben zieht.

```{.python .input}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Jede Volumenkörperkurve entspricht einem der sechs Werte der Düse und gibt unsere geschätzte Wahrscheinlichkeit an, dass die Düse diesen Wert anwendet, wie sie nach jeder Gruppe von Experimenten beurteilt wird. Die gestrichelte schwarze Linie gibt die wahre zugrunde liegende Wahrscheinlichkeit an. Wenn wir mehr Daten erhalten, indem wir mehr Experimente durchführen, konvergieren die $6$ Volumenkurven zur wahren Wahrscheinlichkeit.

### Axiome der Wahrscheinlichkeitstheorie

Wenn wir mit den Rollen eines Stempels umgehen, nennen wir das Set $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ den *Beispielraum* oder *Ergebnisbereich*, wobei jedes Element ein *Ergebnis* ist. Ein *event* ist eine Reihe von Ergebnissen aus einem gegebenen Beispielraum. Zum Beispiel sind „Sehen eines $5$" ($\{5\}$) und „Sehen einer ungeraden Zahl“ ($\{1, 3, 5\}$) beide gültige Ereignisse des Walzens eines Düsens. Beachten Sie, dass, wenn das Ergebnis eines zufälligen Experiments im Ereignis $\mathcal{A}$ ist, Ereignis $\mathcal{A}$ eingetreten ist. Das heißt, wenn $3$ Punkte nach dem Walzen eines Würfel auftauchten, seit $3 \in \{1, 3, 5\}$, können wir sagen, dass das Ereignis „eine ungerade Zahl sehen“ stattgefunden hat.

Formal kann *Wahrscheinlichkeit* an eine Funktion gedacht werden, die einen Satz auf einen realen Wert abbildet. Die Wahrscheinlichkeit eines Ereignisses $\mathcal{A}$ im angegebenen Stichprobenraum $\mathcal{S}$, bezeichnet als $P(\mathcal{A})$, erfüllt die folgenden Eigenschaften:

* Für jeden Fall $\mathcal{A}$ ist seine Wahrscheinlichkeit niemals negativ, d. h. $P(\mathcal{A}) \geq 0$;
* Die Wahrscheinlichkeit des gesamten Probenraums beträgt $1$, d. h. $P(\mathcal{S}) = 1$;
* Für jede zählbare Sequenz von Ereignissen $\mathcal{A}_1, \mathcal{A}_2, \ldots$, die sich gegenseitig ausschließen* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ für alle $i \neq j$), ist die Wahrscheinlichkeit, dass irgendein geschieht, gleich der Summe ihrer individuellen Wahrscheinlichkeiten, d.h. $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

Dies sind auch die Axiome der Wahrscheinlichkeitstheorie, vorgeschlagen von Kolmogorov im Jahr 1933. Dank dieses Axiom-Systems können wir jeden philosophischen Streit über Zufälligkeit vermeiden; stattdessen können wir rigoros mit einer mathematischen Sprache begründen. Wenn wir beispielsweise das Ereignis $\mathcal{A}_1$ den gesamten Probenraum und $\mathcal{A}_i = \emptyset$ für alle $i > 1$ haben, können wir beweisen, dass $P(\emptyset) = 0$, d. h. die Wahrscheinlichkeit eines unmöglichen Ereignisses $0$ ist.

### Zufällige Variablen

In unserem zufälligen Experiment, einen Würfel zu werfen, haben wir die Vorstellung einer *zufälligen Variable* eingeführt. Eine Zufallsvariable kann so ziemlich jede Menge sein und ist nicht deterministisch. Es könnte einen Wert unter einer Reihe von Möglichkeiten in einem zufälligen Experiment nehmen. Betrachten Sie eine Zufallsvariable $X$, deren Wert sich im Stichprobenraum $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ des Walzens eines Düsens befindet. Wir können das Ereignis „sehen ein $5$“ als $\{X = 5\}$ oder $X = 5$ bezeichnen, und seine Wahrscheinlichkeit als $P(\{X = 5\})$ oder $P(X = 5)$. Mit $P(X = a)$ unterscheiden wir zwischen der Zufallsvariablen $X$ und den Werten (z. B. $a$), die $X$ annehmen kann. Eine solche Pedanterie führt jedoch zu einer umständlichen Notation. Für eine kompakte Notation können wir einerseits $P(X)$ als *Verteilung* über die Zufallsvariable $X$ bezeichnen: Die Verteilung sagt uns die Wahrscheinlichkeit, dass $X$ einen beliebigen Wert annimmt. Auf der anderen Seite können wir einfach $P(a)$ schreiben, um die Wahrscheinlichkeit zu bezeichnen, dass eine Zufallsvariable den Wert $a$ nimmt. Da ein Ereignis in der Wahrscheinlichkeitstheorie eine Reihe von Ergebnissen aus dem Stichprobenraum ist, können wir einen Wertebereich für eine zufällige Variable angeben. Beispiel: $P(1 \leq X \leq 3)$ bezeichnet die Wahrscheinlichkeit des Ereignisses $\{1 \leq X \leq 3\}$, was $\{X = 1, 2, \text{or}, 3\}$ bedeutet. Entsprechung $P(1 \leq X \leq 3)$ stellt die Wahrscheinlichkeit dar, dass die Zufallsvariable $X$ einen Wert von $\{1, 2, 3\}$ annehmen kann.

Beachten Sie, dass es einen subtilen Unterschied zwischen *diskreten* Zufallsvariablen, wie den Seiten einer Matrize, und *kontinuierlich* gibt, wie dem Gewicht und der Größe einer Person. Es hat wenig Sinn zu fragen, ob zwei Personen genau die gleiche Höhe haben. Wenn wir genau genug Messungen durchführen, werden Sie feststellen, dass keine zwei Menschen auf dem Planeten genau die gleiche Höhe haben. In der Tat, wenn wir eine fein genug Messung nehmen, werden Sie nicht die gleiche Höhe haben, wenn Sie aufwachen und wenn Sie schlafen gehen. So gibt es keinen Zweck, nach der Wahrscheinlichkeit zu fragen, dass jemand 1.80139278291028719210196740527486202 Meter hoch ist. Angesichts der Weltbevölkerung von Menschen ist die Wahrscheinlichkeit praktisch 0. Es macht in diesem Fall sinnvoller zu fragen, ob die Höhe eines Menschen in ein bestimmtes Intervall fällt, sagen wir zwischen 1,79 und 1,81 Metern. In diesen Fällen quantifizieren wir die Wahrscheinlichkeit, dass wir einen Wert als *Dichte* sehen. Die Höhe von genau 1,80 Metern hat keine Wahrscheinlichkeit, aber eine Dichte ungleich Null. Im Intervall zwischen zwei verschiedenen Höhen haben wir eine Wahrscheinlichkeit ungleich Null. Im Rest dieses Abschnitts betrachten wir die Wahrscheinlichkeit im diskreten Raum. Für die Wahrscheinlichkeit gegenüber kontinuierlichen Zufallsvariablen, können Sie :numref:`sec_random_variables` beziehen.

## Umgang mit mehreren Zufallsvariablen

Sehr oft werden wir mehr als eine zufällige Variable gleichzeitig betrachten wollen. Zum Beispiel möchten wir vielleicht die Beziehung zwischen Krankheiten und Symptomen modellieren. Bei einer Krankheit und einem Symptom, sagen Sie „Grippe“ und „Husten“, kann oder nicht bei einem Patienten mit einer gewissen Wahrscheinlichkeit auftreten. Obwohl wir hoffen, dass die Wahrscheinlichkeit beider annähernd Null liegt, möchten wir diese Wahrscheinlichkeiten und ihre Beziehungen zueinander schätzen, damit wir unsere Schlüsse anwenden können, um eine bessere medizinische Versorgung zu erzielen.

Als komplizierteres Beispiel enthalten Bilder Millionen von Pixeln, also Millionen von Zufallsvariablen. Und in vielen Fällen werden Bilder mit einem Etikett geliefert, das Objekte im Bild identifiziert. Wir können uns das Label auch als Zufallsvariable vorstellen. Wir können sogar alle Metadaten als Zufallsvariablen wie Standort, Zeit, Blende, Brennweite, ISO, Fokusentfernung und Kameratyp vorstellen. All dies sind Zufallsvariablen, die gemeinsam auftreten. Wenn wir mit mehreren Zufallsvariablen umgehen, gibt es mehrere Mengen von Interesse.

### Gelenkwahrscheinlichkeit

Die erste wird die *gemeinsame Wahrscheinlichkeit* $P(A = a, B=b)$ genannt. Angesichts der Werte $a$ und $b$, die gemeinsame Wahrscheinlichkeit lässt uns antworten, was ist die Wahrscheinlichkeit, dass $A=a$ und $B=b$ gleichzeitig? Beachten Sie, dass für alle Werte $a$ und $b$, $P(A=a, B=b) \leq P(A=a)$. Dies muss der Fall sein, denn damit $A=a$ und $B=b$ geschehen, muss $A=a$ geschehen, *und* $B=b$ muss ebenfalls geschehen (und umgekehrt). Daher können $A=a$ und $B=b$ nicht wahrscheinlicher sein als $A=a$ oder $B=b$ einzeln.

### Bedingte Wahrscheinlichkeit

Dies bringt uns zu einem interessanten Verhältnis: $0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$. Wir nennen dieses Verhältnis eine *bedingte Wahrscheinlichkeit* und bezeichnen es mit $P(B=b \mid A=a)$: Es ist die Wahrscheinlichkeit von $B=b$, vorausgesetzt, $A=a$ ist aufgetreten.

### Satz von Bayes

Anhand der Definition von bedingten Wahrscheinlichkeiten können wir eine der nützlichsten und gefeierten Gleichungen in der Statistik ableiten: *Bayes' Theorem*. Es geht wie folgt. Durch den Bau haben wir die *Multiplikationsregel, die $P(A, B) = P(B \mid A) P(A)$. Durch Symmetrie gilt dies auch für $P(A, B) = P(A \mid B) P(B)$. Angenommen, $P(B) > 0$. Lösung für eine der bedingten Variablen erhalten wir

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

Beachten Sie, dass wir hier die kompaktere Notation verwenden, bei der $P(A, B)$ eine *gemeinsame Verteilung* und $P(A \mid B)$ eine *bedingte Verteilung* ist. Solche Verteilungen können für bestimmte Werte $A = a, B=b$ ausgewertet werden.

### Marginalisierung

Bayes' Satz ist sehr nützlich, wenn wir eine Sache von der anderen ableiten wollen, sagen Ursache und Wirkung, aber wir wissen nur die Eigenschaften in umgekehrter Richtung, wie wir später in diesem Abschnitt sehen werden. Eine wichtige Operation, die wir brauchen, um diese Arbeit zu machen, ist *Marginalisierung*. Es ist der Betrieb der Bestimmung $P(B)$ von $P(A, B)$. Wir können sehen, dass die Wahrscheinlichkeit von $B$ darin besteht, alle möglichen Entscheidungen von $A$ zu bilanzieren und die gemeinsamen Wahrscheinlichkeiten über alle zusammenzufassen:

$$P(B) = \sum_{A} P(A, B),$$

, die auch als „*Summenregel“ bezeichnet wird. Die Wahrscheinlichkeit oder Verteilung als Folge der Marginalisierung wird als *marginale Wahrscheinlichkeit* oder als eine *marginale Verteilung* bezeichnet.

### Unabhängigkeit

Eine weitere nützliche Eigenschaft, nach der geprüft werden muss, ist *Abhängigkeit* vs. *Unabhängigkeit*. Zwei Zufallsvariablen $A$ und $B$, die unabhängig sind, bedeutet, dass das Auftreten eines Ereignisses von $A$ keine Informationen über das Auftreten eines Ereignisses von $B$ enthüllt. In diesem Fall $P(B \mid A) = P(B)$. Statistiker geben dies normalerweise als $A \perp  B$ aus. Aus Bayes' Satz folgt sofort, dass auch $P(A \mid B) = P(A)$. In allen anderen Fällen nennen wir $A$ und $B$ abhängig. Zum Beispiel sind zwei aufeinanderfolgende Walzen einer Matrize unabhängig. Im Gegensatz dazu sind die Position eines Lichtschalters und die Helligkeit im Raum nicht (sie sind jedoch nicht vollkommen deterministisch, da wir immer eine defekte Glühbirne, Stromausfall oder einen defekten Schalter haben könnten).

Da $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ $P(A, B) = P(A)P(B)$ entspricht, sind zwei Zufallsvariablen unabhängig, wenn ihre gemeinsame Verteilung das Produkt ihrer einzelnen Verteilungen ist. Ebenso sind zwei Zufallsvariablen $A$ und $B$ *bedingt unabhängig* bei einer anderen Zufallsvariablen $C$, wenn und nur bei $P(A, B \mid C) = P(A \mid C)P(B \mid C)$. Dies wird als $A \perp B \mid C$ ausgedrückt.

### Anwendung
:label:`subsec_probability_hiv_app`

Lassen Sie uns unsere Fähigkeiten auf die Probe stellen. Angenommen, ein Arzt verabreicht einen Aids-Test an einen Patienten. Dieser Test ist ziemlich genau und schlägt nur mit 1% Wahrscheinlichkeit fehl, wenn der Patient gesund ist, aber ihn als krank meldet. Darüber hinaus ist es nie versäumt, HIV zu erkennen, wenn der Patient es tatsächlich hat. Wir verwenden $D_1$, um die Diagnose anzuzeigen ($1$, wenn positiv und $0$, wenn negativ) und $H$, um den HIV-Status zu bezeichnen ($1$, wenn positiv und $0$, wenn negativ). :numref:`conditional_prob_D1` listet solche bedingten Wahrscheinlichkeiten auf.

:Bedingte Wahrscheinlichkeit von $P(D_1 \mid H)$.

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

Beachten Sie, dass die Spaltensummen alle 1 sind (aber die Zeilensummen sind nicht), da die bedingte Wahrscheinlichkeit bis zu 1 summieren muss, genau wie die Wahrscheinlichkeit. Lassen Sie uns die Wahrscheinlichkeit herausfinden, dass der Patient AIDS hat, wenn der Test positiv zurückkehrt, dh $P(H = 1 \mid D_1 = 1)$. Offensichtlich hängt dies davon ab, wie häufig die Krankheit ist, da sie die Anzahl der Fehlalarme beeinflusst. Angenommen, die Bevölkerung ist ziemlich gesund, z. B. $P(H=1) = 0.0015$. Um Bayes' Satz anzuwenden, müssen wir Marginalisierung und die Multiplikationsregel anwenden, um zu bestimmen

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

So erhalten wir

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

Mit anderen Worten, es besteht nur eine 13,06% Chance, dass der Patient tatsächlich AIDS hat, obwohl er einen sehr genauen Test verwendet. Wie wir sehen können, kann die Wahrscheinlichkeit kontraintuitiv sein.

Was sollte ein Patient tun, wenn er solche schrecklichen Nachrichten erhält? Wahrscheinlich würde der Patient den Arzt bitten, einen anderen Test zu verabreichen, um Klarheit zu erhalten. Der zweite Test hat unterschiedliche Eigenschaften und ist nicht so gut wie der erste, wie in :numref:`conditional_prob_D2` gezeigt.

:Bedingte Wahrscheinlichkeit von $P(D_2 \mid H)$.

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

Leider kommt auch der zweite Test positiv zurück. Lassen Sie uns die erforderlichen Wahrscheinlichkeiten erarbeiten, um Bayes' Satz herbeizurufen, indem wir die bedingte Unabhängigkeit annehmen:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$

Jetzt können wir Marginalisierung und die Multiplikationsregel anwenden:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

Am Ende ist die Wahrscheinlichkeit, dass der Patient mit AIDS beide positiven Tests erhalten hat, ist

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

Das heißt, der zweite Test erlaubte uns, viel höheres Vertrauen zu gewinnen, dass nicht alles gut ist. Obwohl der zweite Test deutlich weniger genau war als der erste, hat er unsere Schätzung immer noch deutlich verbessert.

## Erwartung und Abweichung

Um die wichtigsten Merkmale von Wahrscheinlichkeitsverteilungen zusammenzufassen, benötigen wir einige Maßnahmen. Die *Erwartung* (oder Durchschnitt) der Zufallsvariablen $X$ wird als

$$E[X] = \sum_{x} x P(X = x).$$

Wenn die Eingabe einer Funktion $f(x)$ eine Zufallsvariable ist, die aus der Verteilung $P$ mit unterschiedlichen Werten $x$ gezogen wird, wird die Erwartung von $f(x)$ als

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$

In vielen Fällen wollen wir messen, wie stark die Zufallsvariable $X$ von ihrer Erwartung abweicht. Dies kann durch die Varianz quantifiziert werden

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

Seine Quadratwurzel wird *Standardabweichung* genannt. Die Varianz einer Funktion einer Zufallsvariablen misst, wie stark die Funktion von der Erwartung der Funktion abweicht, da verschiedene Werte $x$ der Zufallsvariablen aus ihrer Verteilung abgeleitet werden:

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$

## Zusammenfassung

* Wir können Stichproben aus Wahrscheinlichkeitsverteilungen.
* Wir können mehrere Zufallsvariablen mit Hilfe von Gelenkverteilung, bedingter Verteilung, Bayes-Theorem, Marginalisierung und Unabhängigkeit Annahmen analysieren.
* Erwartung und Abweichung bieten nützliche Maßnahmen, um die wichtigsten Merkmale von Wahrscheinlichkeitsverteilungen zusammenzufassen.

## Übungen

1. Wir führten $m=500$ Experimentengruppen durch, bei denen jede Gruppe $n=10$ Proben zieht. Variieren Sie $m$ und $n$. Beobachten und analysieren Sie die experimentellen Ergebnisse.
1. Bei zwei Ereignissen mit Wahrscheinlichkeit $P(\mathcal{A})$ und $P(\mathcal{B})$ berechnen Sie obere und untere Grenzen auf $P(\mathcal{A} \cup \mathcal{B})$ und $P(\mathcal{A} \cap \mathcal{B})$. (Hinweis: Zeigen Sie die Situation mit einem [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram) an.)
1. Angenommen, wir haben eine Sequenz von Zufallsvariablen, sagen $A$, $B$ und $C$, wobei $B$ nur von $A$ abhängt, und $C$ hängt nur von $B$, können Sie die gemeinsame Wahrscheinlichkeit vereinfachen $P(A, B, C)$? (Hinweis: Dies ist ein [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain).)
1. In :numref:`subsec_probability_hiv_app` ist der erste Test genauer. Warum führen Sie nicht einfach den ersten Test ein zweites Mal aus?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:
