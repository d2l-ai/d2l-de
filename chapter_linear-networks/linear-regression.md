# Lineare Regression
:label:`sec_linear_regression`

*Regression* bezieht sich auf eine Reihe von Methoden für die Modellierung
die Beziehung zwischen einer oder mehreren unabhängigen Variablen und einer abhängigen Variablen. In den Natur- und Sozialwissenschaften ist der Zweck der Regression am häufigsten
*charakterisieren* die Beziehung zwischen den Ein- und Ausgängen.
Maschinelles Lernen hingegen beschäftigt sich meistens mit *Vorhersage*.

Regressionsprobleme tauchen auf, wenn wir einen numerischen Wert vorhersagen wollen. Häufige Beispiele sind die Vorhersage der Preise (von Häusern, Lagerbeständen usw.), die Vorhersage der Aufenthaltsdauer (für Patienten im Krankenhaus), die Nachfrageprognose (für Einzelhandelsverkäufe) und unzählige andere. Nicht jedes Vorhersageproblem ist ein klassisches Regressionsproblem. In den nachfolgenden Abschnitten werden wir Klassifizierungsprobleme einführen, bei denen das Ziel ist, die Mitgliedschaft unter einer Reihe von Kategorien vorherzusagen.

## Grundelemente der linearen Regression

*Lineare Regression* kann sowohl die einfachste sein
und beliebtesten unter den Standardwerkzeugen zur Regression. Die lineare Regression stammt aus der Morgendämmerung des 19. Jahrhunderts und fließt aus wenigen einfachen Annahmen. Zunächst gehen wir davon aus, dass die Beziehung zwischen den unabhängigen Variablen $\mathbf{x}$ und der abhängigen Variablen $y$ linear ist, d.h., dass $y$ als gewichtete Summe der Elemente in $\mathbf{x}$ ausgedrückt werden kann, wenn die Beobachtungen etwas rauschen. Zweitens gehen wir davon aus, dass jeder Lärm gut erzogen ist (nach einer Gaußschen Verteilung).

Um den Ansatz zu motivieren, lassen Sie uns mit einem laufenden Beispiel beginnen. Angenommen, wir möchten die Preise von Häusern (in Dollar) basierend auf ihrer Fläche (in Quadratfuß) und Alter (in Jahren) schätzen. Um tatsächlich ein Modell für die Vorhersage von Immobilienpreisen zu passen, müssten wir uns an einen Datensatz wenden, der aus Verkäufen besteht, für den wir den Verkaufspreis, die Fläche und das Alter für jedes Haus kennen. In der Terminologie des maschinellen Lernens wird der Datensatz als *Trainingsdatensatz* oder *Trainingssatz* bezeichnet, und jede Zeile (hier die Daten, die einem Verkauf entsprechen) wird als *Beispiel* (oder *Datenpunkt*, *Dateninstanz*, *Beispiel*) bezeichnet. Die Sache, die wir vorherzusagen versuchen (Preis) wird als *label* (oder *target*) bezeichnet. Die unabhängigen Variablen (Alter und Fläche), auf denen die Vorhersagen basieren, werden *Features* (oder *Kovariaten*) genannt.

Typischerweise verwenden wir $n$, um die Anzahl der Beispiele in unserem Datensatz zu bezeichnen. Wir indizieren die Datenpunkte durch $i$ und bezeichnen jede Eingabe als $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$ und die entsprechende Beschriftung als $y^{(i)}$.

### Lineares Modell
:label:`subsec_linear_model`

Die Linearitätsannahme besagt nur, dass das Ziel (Preis) als gewichtete Summe der Merkmale (Fläche und Alter) ausgedrückt werden kann:

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$
:eqlabel:`eq_price-area`

In :eqref:`eq_price-area` werden $w_{\mathrm{area}}$ und $w_{\mathrm{age}}$ *Gewichte* genannt, und $b$ wird als „*Bias*“ bezeichnet (auch als „*Offset*“ oder „Intercept*“ bezeichnet). Die Gewichtungen bestimmen den Einfluss jedes Features auf unsere Vorhersage und die Neigung sagt nur, welchen Wert der prognostizierte Preis nehmen sollte, wenn alle Features den Wert 0 annehmen. Auch wenn wir niemals Häuser mit Null-Bereich sehen werden, oder die genau null Jahre alt sind, brauchen wir immer noch die Vorspannung, sonst werden wir die Ausdruckskraft unseres Modells einschränken. Streng genommen ist :eqref:`eq_price-area` eine *affine Transformation* von Eingabe-Features, die durch eine *lineare Transformation* von Features über gewichtete Summe gekennzeichnet ist, kombiniert mit einer *Übersetzung* über die hinzugefügte Bias.

Bei einem Datensatz ist es unser Ziel, die Gewichte $\mathbf{w}$ und die Neigung $b$ so zu wählen, dass im Durchschnitt die Vorhersagen, die nach unserem Modell gemacht werden, am besten zu den tatsächlichen Preisen passen, die in den Daten beobachtet werden. Modelle, deren Ausgabeprognose durch die affine Transformation von Eingabe-Features bestimmt wird, sind *lineare Modelle*, wobei die affine Transformation durch die gewählten Gewichtungen und Verzerrungen angegeben wird.

In Disziplinen, in denen es üblich ist, sich auf Datasets mit nur wenigen Features zu konzentrieren, ist explizit das Ausdrücken von Modellen mit langer Form üblich. Im maschinellen Lernen arbeiten wir normalerweise mit hochdimensionalen Datensätzen, so dass es bequemer ist, lineare Algebra-Notation zu verwenden. Wenn unsere Eingaben aus $d$ Features bestehen, drücken wir unsere Vorhersage $\hat{y}$ (im Allgemeinen bezeichnet das „Hut“ -Symbol Schätzungen) als

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

Sammeln Sie alle Funktionen in einen Vektor $\mathbf{x} \in \mathbb{R}^d$ und alle Gewichte in einen Vektor $\mathbf{w} \in \mathbb{R}^d$, können wir unser Modell kompakt mit einem Punktprodukt ausdrücken:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

In :eqref:`eq_linreg-y` entspricht der Vektor $\mathbf{x}$ den Merkmalen eines einzelnen Datenpunkts. Wir werden es oft bequem finden, über die *design matrix* $\mathbf{X} \in \mathbb{R}^{n \times d}$ auf Funktionen unseres gesamten Datensatzes mit $n$ Beispielen zu verweisen. Hier enthält $\mathbf{X}$ eine Zeile für jedes Beispiel und eine Spalte für jedes Feature.

Für eine Sammlung von Merkmalen $\mathbf{X}$ können die Vorhersagen $\hat{\mathbf{y}} \in \mathbb{R}^n$ über das Matrix-Vektor-Produkt ausgedrückt werden:

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$

wo während der Summierung Rundfunk (siehe :numref:`subsec_broadcasting`) durchgeführt wird. Angesichts der Merkmale eines Trainingsdatensets $\mathbf{X}$ und der entsprechenden (bekannten) Labels $\mathbf{y}$ ist das Ziel der linearen Regression, den Gewichtsvektor $\mathbf{w}$ und den Biasbegriff $b$ zu finden, dass angesichts der Merkmale eines neuen Datenpunkts, der aus der gleichen Verteilung wie $\mathbf{X}$ stammt, die Bezeichnung des neuen Datenpunkts (in Erwartung) mit dem niedrigsten Fehler vorhergesagt werden.

Selbst wenn wir glauben, dass das beste Modell für die Vorhersage von $y$ bei $\mathbf{x}$ linear ist, würden wir nicht erwarten, einen realen Datensatz mit $n$ Beispielen zu finden, bei denen $y^{(i)}$ genau $\mathbf{w}^\top \mathbf{x}^{(i)}+b$ für alle $1 \leq i \leq n$ entspricht. Zum Beispiel können bei allen Instrumenten, die wir verwenden, um die Funktionen $\mathbf{X}$ und die Etiketten $\mathbf{y}$ zu beobachten, geringe Mengen an Messfehlern leiden. Selbst wenn wir zuversichtlich sind, dass die zugrunde liegende Beziehung linear ist, werden wir einen Rauschbegriff einbeziehen, um solche Fehler zu berücksichtigen.

Bevor wir über die Suche nach den besten *Parameter* (oder *Modellparameter*) $\mathbf{w}$ und $b$ gehen können, benötigen wir zwei weitere Dinge: (i) eine Qualitätsmaßnahme für ein bestimmtes Modell; und (ii) ein Verfahren zur Aktualisierung des Modells zur Verbesserung seiner Qualität.

### Verlust-Funktion

Bevor wir darüber nachdenken, wie wir unser Modell anpassen*, müssen wir ein Maß für *Fitness* bestimmen. Die *Verlustfunktion* quantifiziert den Abstand zwischen dem *real* und dem *prognostizierten Wert des Ziels. Der Verlust wird normalerweise eine nicht-negative Zahl sein, bei der kleinere Werte besser sind und perfekte Vorhersagen einen Verlust von 0 ergeben. Die beliebteste Verlustfunktion bei Regressionsproblemen ist der quadrierte Fehler. Wenn unsere Vorhersage für ein Beispiel $i$ $\hat{y}^{(i)}$ ist und die entsprechende wahre Beschriftung $y^{(i)}$ ist, wird der quadrierte Fehler angegeben durch:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

Die Konstante $\frac{1}{2}$ macht keinen wirklichen Unterschied, wird sich aber notationell als bequem erweisen, wenn wir die Ableitung des Verlustes übernehmen. Da uns der Trainingsdatensatz gegeben wird und somit außerhalb unserer Kontrolle liegt, ist der empirische Fehler nur eine Funktion der Modellparameter. Um die Dinge konkreter zu machen, betrachten Sie das folgende Beispiel, in dem wir ein Regressionsproblem für einen eindimensionalen Fall darstellen, wie in :numref:`fig_fit_linreg` gezeigt.

![Fit data with a linear model.](../img/fit_linreg.svg)
:label:`fig_fit_linreg`

Beachten Sie, dass große Unterschiede zwischen Schätzungen $\hat{y}^{(i)}$ und Beobachtungen $y^{(i)}$ aufgrund der quadratischen Abhängigkeit zu noch größeren Verlustbeiträgen führen. Um die Qualität eines Modells auf dem gesamten Datensatz von $n$ Beispielen zu messen, berechnen wir einfach die Verluste auf dem Trainingsset (oder gleichwertig summieren).

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Beim Training des Modells möchten wir Parameter ($\mathbf{w}^*, b^*$) finden, die den Gesamtverlust in allen Trainingsbeispielen minimieren:

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### Analytische Lösung

Lineare Regression ist ein ungewöhnlich einfaches Optimierungsproblem. Im Gegensatz zu den meisten anderen Modellen, die wir in diesem Buch begegnen werden, kann die lineare Regression analytisch gelöst werden, indem eine einfache Formel angewendet wird. Um zu beginnen, können wir die Voreingenommenheit $b$ in den Parameter $\mathbf{w}$ subsumieren, indem wir eine Spalte an die Designmatrix anhängen, die aus allen besteht. Dann ist unser Vorhersageproblem, $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$ zu minimieren. Es gibt nur einen kritischen Punkt auf der Verlustfläche und er entspricht dem Minimum des Verlustes über die gesamte Domäne. Die Ableitung des Verlusts in Bezug auf $\mathbf{w}$ und die Einstellung auf Null ergibt die analytische (geschlossene) Lösung:

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

Obwohl einfache Probleme wie lineare Regression analytische Lösungen zulassen können, sollten Sie sich nicht an solch ein Glück gewöhnen. Obwohl analytische Lösungen eine nette mathematische Analyse ermöglichen, ist die Anforderung einer analytischen Lösung so restriktiv, dass sie alle von Deep Learning ausschließen würde.

### Minibatch Stochastischer Gradientenabstieg

Selbst in Fällen, in denen wir die Modelle nicht analytisch lösen können, stellt sich heraus, dass wir Modelle in der Praxis noch effektiv trainieren können. Darüber hinaus erweisen sich diese schwer zu optimierenden Modelle für viele Aufgaben als so viel besser, dass herauszufinden, wie man sie trainiert, die Mühe wert ist.

Die Schlüsseltechnik zur Optimierung nahezu jedes Deep-Learning-Modells, die wir in diesem Buch aufrufen werden, besteht darin, den Fehler iterativ zu reduzieren, indem die Parameter in die Richtung aktualisiert werden, die die Verlustfunktion inkrementell senkt. Dieser Algorithmus wird *Gradientenabstieg* genannt.

Die naivste Anwendung des Gradientenabstiegs besteht darin, die Ableitung der Verlustfunktion zu übernehmen, was ein Durchschnitt der Verluste ist, die in jedem einzelnen Beispiel im Datensatz berechnet werden. In der Praxis kann dies extrem langsam sein: Wir müssen den gesamten Datensatz übergeben, bevor Sie eine einzelne Aktualisierung durchführen. Daher werden wir uns oft damit begnügen, eine zufällige Minibatch von Beispielen jedes Mal, wenn wir die Aktualisierung berechnen müssen, eine Variante namens *minibatch stochastische Gradientenabstieg*.

In jeder Iteration nehmen wir zunächst zufällig eine Minibatch $\mathcal{B}$ ab, die aus einer festen Anzahl von Trainingsbeispielen besteht. Anschließend berechnen wir die Ableitung (Gradient) des durchschnittlichen Verlustes auf dem Minibatch in Bezug auf die Modellparameter. Schließlich multiplizieren wir den Gradienten mit einem vorgegebenen positiven Wert $\eta$ und subtrahieren den resultierenden Term von den aktuellen Parameterwerten.

Wir können das Update mathematisch wie folgt ausdrücken ($\partial$ bezeichnet die partielle Ableitung):

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

Zusammenfassend sind die Schritte des Algorithmus die folgenden: (i) wir initialisieren die Werte der Modellparameter, typischerweise zufällig; (ii) wir nehmen iterativ zufällige Minibatches aus den Daten ab und aktualisieren die Parameter in Richtung des negativen Gradienten. Bei quadratischen Verlusten und affinen Transformationen können wir dies explizit wie folgt ausschreiben:

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

Beachten Sie, dass $\mathbf{w}$ und $\mathbf{x}$ Vektoren in :eqref:`eq_linreg_batch_update` sind. Hier macht die elegantere Vektornotation die Mathematik viel lesbarer als das Ausdrücken von Dingen in Bezug auf Koeffizienten, sagen $w_1, w_2, \ldots, w_d$. Die Set-Kardinalität $|\mathcal{B}|$ stellt die Anzahl der Beispiele in jedem Minibatch dar (die *Chargengröße*) und $\eta$ bezeichnet die *Lernrate*. Wir betonen, dass die Werte der Chargengröße und Lernrate manuell vorgegeben werden und in der Regel nicht durch Modellschulungen erlernt werden. Diese Parameter, die in der Trainingsschleife abstimmbar sind, aber nicht aktualisiert werden, heißen *Hyperparameter*.
*Hyperparameter-tuning* ist der Prozess, durch den Hyperparameter gewählt werden,
und erfordert in der Regel, dass wir sie basierend auf den Ergebnissen der Trainingsschleife anpassen, die in einem separaten *Validierungsdataset* (oder *Validierungssatz*) bewertet werden.

Nach dem Training für einige vorbestimmte Anzahl von Iterationen (oder bis einige andere Stoping-Kriterien erfüllt sind), nehmen wir die geschätzten Modellparameter auf, bezeichnet $\hat{\mathbf{w}}, \hat{b}$. Beachten Sie, dass, selbst wenn unsere Funktion wirklich linear und geräuschlos ist, diese Parameter nicht die exakten Minimierer des Verlustes sein werden, weil der Algorithmus, obwohl er langsam zu den Minimierern konvergiert, nicht genau in einer endlichen Anzahl von Schritten erreichen kann.

Lineare Regression ist zufällig ein Lernproblem, bei dem es nur ein Minimum über die gesamte Domäne gibt. Für kompliziertere Modelle, wie tiefe Netzwerke, enthalten die Verlustflächen jedoch viele Minima. Glücklicherweise kämpfen Deep Learning-Praktizierende aus Gründen, die noch nicht vollständig verstanden sind, selten darum, Parameter zu finden, die den Verlust bei Trainingssets* minimieren. Die gewaltigere Aufgabe ist es, Parameter zu finden, die einen geringen Verlust an Daten erreichen, die wir noch nicht gesehen haben, eine Herausforderung namens *Verallgemeinerung*. Wir kehren zu diesen Themen während des gesamten Buches zurück.

### Vorhersagen mit dem gelernten Modell

Angesichts des gelernten linearen Regressionsmodells $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$ können wir nun den Preis eines neuen Hauses (nicht in den Trainingsdaten enthalten) aufgrund seiner Fläche $x_1$ und Alter $x_2$ schätzen. Das Schätzen von Zielen gegebene Features wird üblicherweise *Vorhersage* oder *Inference* genannt.

Wir werden versuchen, bei *Vorhersage* zu bleiben, weil der Aufruf dieses Schrittes *inference*, obwohl es als Standardjargon im Deep Learning auftaucht, etwas falsch ist. In der Statistik bezeichnet *inference* häufiger Schätzparameter basierend auf einem Datensatz. Dieser Missbrauch der Terminologie ist eine häufige Quelle der Verwirrung, wenn Deep Learning Praktiker mit Statistikern sprechen.

## Vektorisierung für Geschwindigkeit

Beim Training unserer Modelle wollen wir in der Regel ganze Minibatches von Beispielen gleichzeitig verarbeiten. Dies effizient zu tun, erfordert, dass wir die Berechnungen vektorisieren und schnelle lineare Algebra-Bibliotheken nutzen, anstatt kostspielige for-Schleifen in Python zu schreiben.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

Um zu veranschaulichen, warum das so wichtig ist, können wir zwei Methoden zum Hinzufügen von Vektoren in Betracht ziehen. Zu Beginn instanziieren wir zwei 10000-dimensionale Vektoren, die alle enthalten. In einer Methode werden wir die Vektoren mit einer Python for-Schleife durchlaufen. In der anderen Methode werden wir auf einen einzigen Aufruf von `+` verlassen.

```{.python .input}
#@tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

Da wir die Laufzeit häufig in diesem Buch vergleichen werden, lassen Sie uns einen Timer definieren.

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```

Jetzt können wir die Workloads vergleichen. Zuerst fügen wir sie, eine Koordinate nach dem anderen, mit einer for-Schleife hinzu.

```{.python .input}
#@tab mxnet, pytorch
c = d2l.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab tensorflow
c = tf.Variable(d2l.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'
```

Alternativ können wir uns auf den neu geladenen Operator `+` verlassen, um die elementweise Summe zu berechnen.

```{.python .input}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

Sie haben wahrscheinlich bemerkt, dass die zweite Methode dramatisch schneller ist als die erste. Vektorisierender Code führt oft zu einer Geschwindigkeit von Größenordnung. Darüber hinaus schieben wir mehr von der Mathematik in die Bibliothek und müssen nicht so viele Berechnungen selbst schreiben, wodurch das Fehlerpotenzial reduziert wird.

## Die Normalverteilung und der quadratische Verlust
:label:`subsec_normal_distribution_and_squared_loss`

Während Sie Ihre Hände bereits mit den oben genannten Informationen schmutzig machen können, können wir im Folgenden das Quadratverlust-Ziel durch Annahmen über die Verteilung von Lärm formeller motivieren.

Die lineare Regression wurde 1795 von Gauss erfunden, der auch die Normalverteilung (auch Gaussian* genannt) entdeckte. Es stellt sich heraus, dass die Verbindung zwischen der Normalverteilung und der linearen Regression tiefer verläuft als die gemeinsame Abstammung. Um Ihren Speicher zu aktualisieren, wird die Wahrscheinlichkeitsdichte einer Normalverteilung mit dem Mittelwert $\mu$ und der Varianz $\sigma^2$ (Standardabweichung $\sigma$) als

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

Im Folgenden definieren wir eine Python-Funktion, um die Normalverteilung zu berechnen.

```{.python .input}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

Wir können nun die Normalverteilungen visualisieren.

```{.python .input}
#@tab all
# Use numpy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

Wie wir sehen können, entspricht das Ändern des Mittelwerts einer Verschiebung entlang der $x$-Achse, und das Erhöhen der Varianz breitet sich die Verteilung aus und senkt ihren Höhepunkt.

Eine Möglichkeit, eine lineare Regression mit der Funktion zur mittleren quadratischen Fehlerverlustfunktion (oder einfach quadratischer Verlust) zu motivieren, besteht darin, formal anzunehmen, dass Beobachtungen aus lauten Beobachtungen entstehen, bei denen das Rauschen normalerweise wie folgt verteilt wird:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

So können wir jetzt die *likelihood* schreiben, einen bestimmten $y$ für eine gegebene $\mathbf{x}$ über

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

Nun, nach dem Prinzip der maximalen Wahrscheinlichkeit, sind die besten Werte der Parameter $\mathbf{w}$ und $b$ diejenigen, die die *Likelihood* des gesamten Datensatzes maximieren:

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

Schätzer, die nach dem Prinzip der maximalen Wahrscheinlichkeit ausgewählt werden, werden *Maximum-Likelihood-Schätzungen* genannt. Während die Maximierung des Produkts vieler exponentieller Funktionen schwierig aussehen könnte, können wir die Dinge erheblich vereinfachen, ohne das Ziel zu ändern, indem wir stattdessen das Protokoll der Wahrscheinlichkeit maximieren. Aus historischen Gründen werden Optimierungen häufiger als Minimierung und nicht als Maximierung ausgedrückt. Also, ohne etwas zu ändern, können wir die *negative log-Likelihood* $-\log P(\mathbf y \mid \mathbf X)$ minimieren. Das Ausarbeiten der Mathematik gibt uns:

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Jetzt brauchen wir nur noch eine Annahme, dass $\sigma$ eine feste Konstante ist. So können wir den ersten Begriff ignorieren, weil er nicht von $\mathbf{w}$ oder $b$ abhängt. Nun ist der zweite Begriff identisch mit dem zuvor eingeführten quadratischen Fehlerverlust, mit Ausnahme der multiplikativen Konstante $\frac{1}{\sigma^2}$. Glücklicherweise hängt die Lösung nicht von $\sigma$ ab. Daraus folgt, dass die Minimierung des mittleren quadrierten Fehlers der maximalen Wahrscheinlichkeitsschätzung eines linearen Modells unter der Annahme des additiven Gaußschen Rauschens entspricht.

## Von linearer Regression zu tiefen Netzwerken

Bisher haben wir nur über lineare Modelle gesprochen. Während neuronale Netze eine viel reichere Modellfamilie abdecken, können wir anfangen, das lineare Modell als neuronales Netzwerk zu betrachten, indem wir es in der Sprache neuronaler Netze ausdrücken. Beginnen wir damit, Dinge in einer „Ebene“ -Notation neu zu schreiben.

### Diagramm des neuronalen Netzwerks

Deep Learning-Praktizierende zeichnen gerne Diagramme, um zu visualisieren, was in ihren Modellen geschieht. In :numref:`fig_single_neuron` stellen wir unser lineares Regressionsmodell als neuronales Netzwerk dar. Beachten Sie, dass diese Diagramme das Konnektivitätsmuster hervorheben, z. B. wie jede Eingabe mit der Ausgabe verbunden ist, aber nicht die Werte, die von den Gewichtungen oder Verzerrungen übernommen werden.

![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
:label:`fig_single_neuron`

Für das in :numref:`fig_single_neuron` gezeigte neuronale Netzwerk sind die Eingänge $x_1, \ldots, x_d$, daher ist die *Anzahl der Eingängen* (oder *Feature-Dimensionality*) im Eingabe-Layer $d$. Die Ausgabe des Netzwerks in :numref:`fig_single_neuron` ist $o_1$, also ist die *Anzahl der Ausgabe* in der Ausgabeschicht 1. Beachten Sie, dass die Eingabewerte alle *given* sind und es nur ein einzelnes *computed* Neuron gibt. Konzentrieren wir uns darauf, wo die Berechnung stattfindet, üblicherweise betrachten wir die Eingabe-Schicht beim Zählen von Ebenen nicht. Das heißt, die *Anzahl der Schichten* für das neuronale Netzwerk in :numref:`fig_single_neuron` ist 1. Wir können uns lineare Regressionsmodelle als neuronale Netze vorstellen, die nur aus einem einzigen künstlichen Neuron bestehen, oder als einschichtige neuronale Netze.

Da bei linearer Regression jeder Eingang mit jedem Ausgang verbunden ist (in diesem Fall gibt es nur einen Ausgang), können wir diese Transformation (die Ausgabeschicht in :numref:`fig_single_neuron`) als eine *voll verbundene Schicht* oder *dichte Schicht* betrachten. Im nächsten Kapitel werden wir viel mehr über Netzwerke sprechen, die aus solchen Schichten bestehen.

### Biologie

Da die lineare Regression (erfunden 1795) vor der Computerneurowissenschaft liegt, scheint es anachronistisch zu sein, lineare Regression als neuronales Netzwerk zu beschreiben. Um zu sehen, warum lineare Modelle ein natürlicher Ort waren zu beginnen, als die Cybernetiker/Neurophysiologen Warren McCulloch und Walter Pitts begannen, Modelle von künstlichen Neuronen zu entwickeln, betrachten Sie das karikaturische Bild eines biologischen Neurons in :numref:`fig_Neuron`, bestehend aus
*Dendrites* (Eingangsklemmen),
der *Kern* (CPU), der *Axon* (Ausgangsleitung) und die *Axonklemmen* (Ausgangsklemmen), die Verbindungen zu anderen Neuronen über *Synapses* ermöglichen.

![The real neuron.](../img/Neuron.svg)
:label:`fig_Neuron`

Informationen $x_i$, die von anderen Neuronen (oder Umgebungssensoren wie der Netzhaut) ankommen, werden in den Dendriten empfangen. Insbesondere werden diese Informationen mit *synaptischen Gewichtungen* $w_i$ gewichtet, die die Wirkung der Eingaben bestimmen (z. B. Aktivierung oder Hemmung über das Produkt $x_i w_i$). Die gewichteten Eingänge, die aus mehreren Quellen stammen, werden im Kern als gewichtete Summe $y = \sum_i x_i w_i + b$ aggregiert, und diese Informationen werden dann zur weiteren Verarbeitung im Axon $y$ gesendet, typischerweise nach einer nichtlinearen Verarbeitung über $\sigma(y)$. Von dort erreicht es entweder sein Ziel (z.B. einen Muskel) oder wird über seine Dendriten in ein anderes Neuron eingespeist.

Natürlich verdankt die hochrangige Idee, dass viele solcher Einheiten zusammen mit der richtigen Konnektivität und dem richtigen Lernalgorithmus gepflastert werden könnten, um viel interessanteres und komplexeres Verhalten zu erzeugen, als jedes einzelne Neuron allein könnte unserer Untersuchung realer biologischer neuronaler Systeme zum Ausdruck bringen.

Gleichzeitig zieht die meiste Forschung im Deep Learning heute wenig direkte Inspiration in der Neurowissenschaft. Wir rufen Stuart Russell und Peter Norvig auf, die in ihrem klassischen KI-Textbuch
*Künstliche Intelligence: A Modern Approach* :cite:`Russell.Norvig.2016`,
wies darauf hin, dass, obwohl Flugzeuge von Vögeln inspiriert worden sein könnten, Ornithologie seit einigen Jahrhunderten nicht der Haupttreiber für die Innovation in der Luftfahrt war. Ebenso kommt Inspiration für Deep Learning heutzutage in gleichem oder größerem Maße aus Mathematik, Statistik und Informatik.

## Zusammenfassung

* Hauptbestandteile in einem Machine Learning-Modell sind Trainingsdaten, eine Verlustfunktion, ein Optimierungsalgorithmus und ganz offensichtlich das Modell selbst.
* Vektorisierung macht alles besser (meist mathematisch) und schneller (meist Code).
* Die Minimierung einer objektiven Funktion und die Durchführung einer maximalen Wahrscheinlichkeitsschätzung kann dasselbe bedeuten.
* Lineare Regressionsmodelle sind auch neuronale Netze.

## Übungen

1. Angenommen, wir haben einige Daten $x_1, \ldots, x_n \in \mathbb{R}$. Unser Ziel ist es, eine konstante $b$ so zu finden, dass $\sum_i (x_i - b)^2$ minimiert wird.
    * Finden Sie eine analytische Lösung für den optimalen Wert von $b$.
    * Wie bezieht sich dieses Problem und seine Lösung auf die Normalverteilung?
1. Ableiten der analytischen Lösung für das Optimierungsproblem für lineare Regression mit quadrierten Fehlern. Um die Dinge einfach zu halten, können Sie die Voreingenommenheit $b$ vom Problem weglassen (wir können dies prinzipiell tun, indem wir eine Spalte zu $\mathbf X$ hinzufügen, die aus allen besteht).
    * Schreiben Sie das Optimierungsproblem in Matrix- und Vektornotation auf (behandeln Sie alle Daten als eine einzelne Matrix und alle Zielwerte als einen einzelnen Vektor).
    * Berechnen Sie den Gradienten des Verlustes in Bezug auf $w$.
    * Finden Sie die analytische Lösung, indem Sie den Farbverlauf gleich Null festlegen und die Matrixgleichung lösen.
    * Wann könnte dies besser sein als die Verwendung stochastischer Gradientenabstieg? Wann könnte diese Methode brechen?
1. Angenommen, das Rauschmodell für das additive Rauschen $\epsilon$ ist die exponentielle Verteilung. Das heißt, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    * Schreiben Sie die negative Log-Likelihood der Daten unter dem Modell $-\log P(\mathbf y \mid \mathbf X)$ aus.
    * Finden Sie eine geschlossene Formularlösung?
    * Schlagen Sie einen stochastischen Gradientenabstiegsalgorithmus vor, um dieses Problem zu lösen. Was könnte möglicherweise schief gehen (Hinweis: Was passiert in der Nähe des stationären Punktes, während wir die Parameter weiter aktualisieren)? Kannst du das reparieren?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:
