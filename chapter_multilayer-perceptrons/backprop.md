# Vorwärtspropagation, Rückwärtspropagation und Berechnungsdiagramme
:label:`sec_backprop`

Bisher haben wir unsere Modelle mit minibatch stochastischen Gradientenabstieg trainiert. Als wir jedoch den Algorithmus implementiert haben, machten wir uns nur Sorgen um die Berechnungen, die an der *Forward-Propagation* durch das Modell beteiligt sind. Als es an der Zeit kam, die Gradienten zu berechnen, haben wir gerade die Backpropagation-Funktion aufgerufen, die vom Deep Learning-Framework bereitgestellt wird.

Die automatische Berechnung von Gradienten (automatische Differenzierung) vereinfacht die Implementierung von Deep Learning-Algorithmen grundlegend. Vor der automatischen Differenzierung mussten selbst kleine Änderungen an komplizierten Modellen komplizierte Derivate manuell neu berechnet werden. Überraschenderweise mussten akademische Arbeiten zahlreiche Seiten zuordnen, um Aktualisierungsregeln abzuleiten. Während wir uns weiterhin auf automatische Differenzierung verlassen müssen, damit wir uns auf die interessanten Teile konzentrieren können, sollten Sie wissen, wie diese Gradienten unter der Haube berechnet werden, wenn Sie über ein flaches Verständnis von Deep Learning hinausgehen wollen.

In diesem Abschnitt nehmen wir einen tiefen Einblick in die Details von *Rückwärtspropagation* (häufiger als *Backpropagation*). Um Einblicke sowohl für die Techniken als auch deren Implementierungen zu vermitteln, verlassen wir uns auf einige grundlegende Mathematik und Rechendiagramme. Zunächst fokussieren wir unsere Exposition auf eine ein-versteckte Schicht MLP mit Gewichtszerfall ($L_2$ Regularisierung).

## Weiterleitung

*Forward Propagation* (oder *Forward Pass*) bezieht sich auf die Berechnung und Speicherung
von Zwischenvariablen (einschließlich Ausgängen) für ein neuronales Netzwerk in der Reihenfolge von der Eingabe-Schicht zur Ausgabe-Schicht. Wir arbeiten nun Schritt für Schritt durch die Mechanik eines neuronalen Netzwerks mit einer versteckten Schicht. Das mag mühsam erscheinen, aber in den ewigen Worten des Funkvirtuosen James Brown muss man „die Kosten bezahlen, um der Chef zu sein“.

Der Einfachheit halber nehmen wir an, dass das Eingabebeispiel $\mathbf{x}\in \mathbb{R}^d$ ist und dass unsere versteckte Schicht keinen Biasbegriff enthält. Hier ist die Zwischenvariable:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

wobei $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ der Gewichtsparameter der versteckten Schicht ist. Nachdem wir die Zwischenvariable $\mathbf{z}\in \mathbb{R}^h$ über die Aktivierungsfunktion $\phi$ ausgeführt haben, erhalten wir unseren versteckten Aktivierungsvektor der Länge $h$.

$$\mathbf{h}= \phi (\mathbf{z}).$$

Die versteckte Variable $\mathbf{h}$ ist ebenfalls eine Zwischenvariable. Unter der Annahme, dass die Parameter der Ausgabeschicht nur ein Gewicht von $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ besitzen, können wir eine Ausgabeschichtvariable mit einem Vektor der Länge $q$ erhalten:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Unter der Annahme, dass die Verlustfunktion $l$ und die Beispielbezeichnung $y$ ist, können wir dann den Verlustbegriff für ein einzelnes Datenbeispiel berechnen,

$$L = l(\mathbf{o}, y).$$

Gemäß der Definition von $L_2$ Regularisierung, angesichts des Hyperparameters $\lambda$, ist der Regularisierungsbegriff

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

wobei die Frobenius-Norm der Matrix einfach die Norm $L_2$ ist, die nach dem Abflachen der Matrix in einen Vektor angewendet wird. Schließlich ist der regularisierte Verlust des Modells für ein gegebenes Datenbeispiel:

$$J = L + s.$$

Wir bezeichnen $J$ als die *objektive Funktion* in der folgenden Diskussion.

## Berechnungsdiagramm der Vorwärtspropagation

Das Plotten von Rechengraphen* hilft uns, die Abhängigkeiten von Operatoren und Variablen innerhalb der Berechnung zu visualisieren. :numref:`fig_forward` enthält den Graphen, der mit dem oben beschriebenen einfachen Netzwerk verknüpft ist, wobei Quadrate Variablen und Kreise Operatoren bezeichnen. Die untere linke Ecke bedeutet die Eingabe und die obere rechte Ecke ist die Ausgabe. Beachten Sie, dass die Richtungen der Pfeile (die den Datenfluss veranschaulichen) hauptsächlich nach rechts und nach oben verlaufen.

![Computational graph of forward propagation.](../img/forward.svg)
:label:`fig_forward`

## Rückverbreitung

*Backpropagation* bezieht sich auf die Methode der Berechnung
der Gradient der neuronalen Netzwerkparameter. Kurz gesagt, die Methode durchquert das Netzwerk in umgekehrter Reihenfolge, von der Ausgabe zur Eingabe-Schicht, entsprechend der *Kettenregel* aus dem Kalkül. Der Algorithmus speichert alle Zwischenvariablen (partielle Derivate), die während der Berechnung des Gradienten in Bezug auf einige Parameter erforderlich sind. Angenommen, wir haben Funktionen $\mathsf{Y}=f(\mathsf{X})$ und $\mathsf{Z}=g(\mathsf{Y})$, in denen die Eingabe und der Ausgang $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ Tensoren beliebiger Formen sind. Durch die Verwendung der Kettenregel können wir die Ableitung von $\mathsf{Z}$ in Bezug auf $\mathsf{X}$ über

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Hier verwenden wir den Operator $\text{prod}$, um seine Argumente zu multiplizieren, nachdem die notwendigen Operationen, wie Transposition und Austausch von Eingabepositionen, durchgeführt wurden. Für Vektoren ist das einfach: Es ist einfach Matrix-Multiplikation. Für höherdimensionale Tensoren verwenden wir das entsprechende Gegenstück. Der Operator $\text{prod}$ verbirgt alle Notation Overhead.

Daran erinnern, dass die Parameter des einfachen Netzwerks mit einer versteckten Schicht, deren Rechendiagramm in :numref:`fig_forward` ist, $\mathbf{W}^{(1)}$ und $\mathbf{W}^{(2)}$ sind. Ziel der Backpropagation ist die Berechnung der Gradienten $\partial J/\partial \mathbf{W}^{(1)}$ und $\partial J/\partial \mathbf{W}^{(2)}$. Um dies zu erreichen, wenden wir die Kettenregel an und berechnen wiederum den Verlauf jeder Zwischenvariablen und Parameter. Die Reihenfolge der Berechnungen wird relativ zu denen umgekehrt, die in der Vorwärtspropagation durchgeführt werden, da wir mit dem Ergebnis des Berechnungsdiagramms beginnen und unseren Weg zu den Parametern arbeiten müssen. Der erste Schritt besteht darin, die Gradienten der objektiven Funktion $J=L+s$ in Bezug auf die Verlustdauer $L$ und den Regularisationsbegriff $s$ zu berechnen.

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

Als nächstes berechnen wir den Gradient der Objektivfunktion in Bezug auf die Variable der Ausgabeschicht $\mathbf{o}$ gemäß der Kettenregel:

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Als nächstes berechnen wir die Gradienten des Regularisierungsbegriffs in Bezug auf beide Parameter:

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Jetzt sind wir in der Lage, den Gradienten $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ der Modellparameter zu berechnen, die der Ausgabeschicht am nächsten liegen. Die Verwendung der Kettenregel ergibt:

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

Um den Farbverlauf in Bezug auf $\mathbf{W}^{(1)}$ zu erhalten, müssen wir die Backpropagation entlang der Ausgabeebene auf die versteckte Ebene fortsetzen. Der Farbverlauf in Bezug auf die Ausgänge des versteckten Layers $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ wird durch

$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Da die Aktivierungsfunktion $\phi$ elementweise gilt, erfordert die Berechnung des Gradienten $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ der Zwischenvariable $\mathbf{z}$, dass wir den elementweisen Multiplikationsoperator verwenden, den wir mit $\odot$ bezeichnen:

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Schließlich können wir den Gradient $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ der Modellparameter erhalten, die der Eingabe-Schicht am nächsten sind. Nach der Kettenregel erhalten wir

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

## Schulung neuronaler Netzwerke

Beim Training neuronaler Netze hängen Vorwärts- und Rückwärtsausbreitung voneinander ab. Insbesondere für die Vorwärtspropagierung durchqueren wir den Berechnungsgraph in Richtung der Abhängigkeiten und berechnen alle Variablen auf seinem Pfad. Diese werden dann für die Backpropagation verwendet, bei der die Rechenreihenfolge im Diagramm umgekehrt wird.

Nehmen Sie das oben erwähnte einfache Netzwerk als Beispiel zu veranschaulichen. Einerseits hängt die Berechnung des Regularisierungsterms :eqref:`eq_forward-s` während der Forward-Propagation von den aktuellen Werten der Modellparameter $\mathbf{W}^{(1)}$ und $\mathbf{W}^{(2)}$ ab. Sie werden durch den Optimierungsalgorithmus entsprechend der Backpropagation in der letzten Iteration gegeben. Andererseits hängt die Gradientenberechnung für den Parameter `eq_backprop-J-h` während der Backpropagation vom aktuellen Wert der versteckten Variablen $\mathbf{h}$ ab, der durch Vorwärtspropagation angegeben wird.

Daher wechseln wir beim Training neuronaler Netze nach der Initialisierung von Modellparametern die Vorwärtspropagation mit Rückpropagation ab und aktualisieren Modellparameter mithilfe von Verläufen, die durch Rückpropagation gegeben werden. Beachten Sie, dass die Backpropagation die gespeicherten Zwischenwerte aus der Vorwärtspropagation wiederverwendet, um doppelte Berechnungen zu vermeiden. Eine der Konsequenzen ist, dass wir die Zwischenwerte beibehalten müssen, bis die Backpropagation abgeschlossen ist. Dies ist auch einer der Gründe, warum Training deutlich mehr Speicher benötigt als einfache Vorhersage. Außerdem ist die Größe solcher Zwischenwerte in etwa proportional zur Anzahl der Netzwerkschichten und der Stapelgröße. So führt das Training tieferer Netzwerke mit größeren Chargengrößen leichter zu Fehlern im Arbeitsspeicher*.

## Zusammenfassung

* Die Forward-Propagation berechnet und speichert Zwischenvariablen in dem vom neuronalen Netzwerk definierten Berechnungsdiagramm. Es geht von der Eingabe in die Ausgabe-Ebene.
* Backpropagation berechnet und speichert die Gradienten von Zwischenvariablen und Parametern innerhalb des neuronalen Netzwerks in umgekehrter Reihenfolge.
* Beim Training von Deep Learning-Modellen sind die Vorwärtsausbreitung und die Rückenausbreitung voneinander abhängig.
* Training erfordert deutlich mehr Speicher als Vorhersage.

## Übungen

1. Angenommen, die Eingaben $\mathbf{X}$ zu einigen skalaren Funktion $f$ sind $n \times m$ Matrizen. Wie groß ist die Dimensionalität des Gradienten von $f$ in Bezug auf $\mathbf{X}$?
1. Fügen Sie der in diesem Abschnitt beschriebenen verdeckten Ebene des Modells eine Verzerrung hinzu.
    * Zeichnen Sie das entsprechende Rechendiagramm.
    * Ableiten der Vorwärts- und Rückwärtsausbreitungsgleichungen.
1. Berechnen Sie den Speicherbedarf für Training und Vorhersage in dem in diesem Abschnitt beschriebenen Modell.
1. Angenommen Sie, Sie möchten zweite Derivate berechnen. Was passiert mit dem Berechnungsdiagramm? Wie lange erwarten Sie, dass die Berechnung dauert?
1. Angenommen, das Berechnungsdiagramm ist zu groß für Ihre GPU.
    * Können Sie es über mehr als eine GPU partitionieren?
    * Was sind die Vor- und Nachteile gegenüber dem Training auf einem kleineren Minibatch?

[Discussions](https://discuss.d2l.ai/t/102)
