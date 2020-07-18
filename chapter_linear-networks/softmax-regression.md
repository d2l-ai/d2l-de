# Softmax-Regression
:label:`sec_softmax`

In :numref:`sec_linear_regression` haben wir lineare Regression eingeführt, indem wir Implementierungen von Grund auf neu in :numref:`sec_linear_scratch` durchlaufen und wieder High-Level-APIs eines Deep Learning-Frameworks in :numref:`sec_linear_concise` verwendet haben, um das schwere Heben zu erledigen.

Regression ist der Hammer, den wir erreichen, wenn wir antworten wollen, *wie viel?* oder *wie viele?* Fragen. Wenn Sie die Anzahl der Dollar (Preis) vorhersagen möchten, zu denen ein Haus verkauft wird, oder die Anzahl der Gewinne, die ein Baseball-Team haben könnte, oder die Anzahl der Tage, an denen ein Patient vor der Entladung ins Krankenhaus eingeliefert wird, dann suchen Sie wahrscheinlich nach einem Regressionsmodell.

In der Praxis sind wir öfter an *Klassifikation* interessiert: Fragen nicht „wie viel“, sondern „welches“:

* Gehört diese E-Mail in den Spam-Ordner oder in den Posteingang?
* Ist dieser Kunde wahrscheinlicher, sich ** anzumelden* oder sich nicht für einen Abonnementservice anzumelden*?
* Zeigt dieses Bild einen Esel, einen Hund, eine Katze oder einen Hahn?
* Welcher Film wird Aston am wahrscheinlichsten als nächstes sehen?

Umgangssprachlich überladen die Praktiker des maschinellen Lernens das Wort „Klassifikation*“, um zwei subtil unterschiedliche Probleme zu beschreiben: (i) diejenigen, bei denen wir nur an harten Zuweisungen von Beispielen zu Kategorien (Klassen) interessiert sind; und (ii) diejenigen, bei denen wir weiche Aufgaben vornehmen möchten, d. h. um die Wahrscheinlichkeit zu beurteilen, dass jede Kategorie gilt. Die Unterscheidung neigt zum Teil dazu, verschwommen zu werden, denn oft, selbst wenn wir uns nur um harte Zuweisungen kümmern, verwenden wir immer noch Modelle, die weiche Zuweisungen machen.

## Klassifizierungsproblem
:label:`subsec_classification-problem`

Um unsere Füße nass zu machen, lassen Sie uns mit einem einfachen Bildklassifizierungsproblem beginnen. Hier besteht jeder Eingang aus einem $2\times2$ Graustufenbild. Wir können jeden Pixelwert mit einem einzigen Skalar darstellen, was uns vier Funktionen $x_1, x_2, x_3, x_4$. Lassen Sie uns ferner davon ausgehen, dass jedes Bild zu einem der Kategorien „Katze“, „Huhn“ und „Hund“ gehört.

Als nächstes müssen wir wählen, wie die Etiketten dargestellt werden sollen. Wir haben zwei offensichtliche Entscheidungen. Vielleicht wäre der natürlichste Impuls, $y \in \{1, 2, 3\}$ zu wählen, wobei die ganzen Zahlen $\{\text{dog}, \text{cat}, \text{chicken}\}$ bzw. $\{\text{dog}, \text{cat}, \text{chicken}\}$ darstellen. Dies ist eine großartige Möglichkeit, derartige Informationen auf einem Computer zu speichern*. Wenn die Kategorien eine natürliche Reihenfolge unter ihnen hatten, sagen wir, wenn wir $\{\text{baby}, \text{toddler}, \text{adolescent}, \text{young adult}, \text{adult}, \text{geriatric}\}$ vorhersagen wollten, dann könnte es sogar sinnvoll sein, dieses Problem als Regression zu werfen und die Etiketten in diesem Format zu behalten.

Aber allgemeine Klassifizierungsprobleme kommen nicht mit natürlichen Ordnungen unter den Klassen. Glücklicherweise haben Statistiker vor langer Zeit eine einfache Möglichkeit erfunden, kategorische Daten darzustellen: die *one-hot codierung*. Eine One-Hot-Codierung ist ein Vektor mit so vielen Komponenten wie wir Kategorien haben. Die Komponente, die der Kategorie bestimmter Exemplare entspricht, wird auf 1 gesetzt, und alle anderen Komponenten werden auf 0 gesetzt. In unserem Fall wäre ein Etikett $y$ ein dreidimensionaler Vektor, mit $(1, 0, 0)$ entsprechend „Katze“, $(0, 1, 0)$ zu „Huhn“ und $(0, 0, 1)$ zu „Hund“:

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

## Netzwerkarchitektur

Um die bedingten Wahrscheinlichkeiten zu schätzen, die mit allen möglichen Klassen verbunden sind, benötigen wir ein Modell mit mehreren Ausgängen, eine pro Klasse. Um die Klassifizierung mit linearen Modellen zu adressieren, benötigen wir so viele affine Funktionen wie Ausgänge. Jeder Ausgang entspricht seiner eigenen affinen Funktion. In unserem Fall, da wir 4 Features und 3 mögliche Ausgabekategorien haben, benötigen wir 12 Skalare, um die Gewichtungen darzustellen ($w$ mit Tiefpunkten), und 3 Skalare, um die Verzerrungen darzustellen ($b$ mit Tiefpunkten). Wir berechnen diese drei Logits*, $o_1, o_2$ und $o_3$ für jede Eingabe:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

Wir können diese Berechnung mit dem neuronalen Netzdiagramm abbilden, das in :numref:`fig_softmaxreg` gezeigt wird. Wie bei der linearen Regression ist die softmax-Regression auch ein einschichtiges neuronales Netzwerk. Und da die Berechnung jeder Ausgabe $o_1, o_2$ und $o_3$ von allen Eingängen ($x_1$, $x_2$, $x_3$ und $x_4$) abhängt, kann die Ausgabeschicht der softmax-Regression auch als vollständig verbundene Schicht beschrieben werden.

![Softmax regression is a single-layer neural network.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

Um das Modell kompakter auszudrücken, können wir lineare Algebra-Notation verwenden. In Vektorform erreichen wir $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$, ein Formular, das sowohl für Mathematik als auch für das Schreiben von Code besser geeignet ist. Beachten Sie, dass wir alle unsere Gewichte in einer $3 \times 4$ Matrix gesammelt haben und dass für Merkmale eines gegebenen Datenpunkts $\mathbf{x}$ unsere Ausgänge durch ein Matrix-Vektorprodukt unserer Gewichte durch unsere Eingabefunktionen plus unsere Vorspannung $\mathbf{b}$ gegeben werden.

## Softmax-Betrieb

Der Hauptansatz, den wir hier verfolgen werden, ist die Ausgänge unseres Modells als Wahrscheinlichkeiten zu interpretieren. Wir optimieren unsere Parameter, um Wahrscheinlichkeiten zu erzeugen, die die Wahrscheinlichkeit der beobachteten Daten maximieren. Um dann Vorhersagen zu generieren, werden wir einen Schwellenwert festlegen, zum Beispiel die Beschriftung mit den maximal prognostizierten Wahrscheinlichkeiten auswählen.

Formal ausgedrückt möchten wir, dass jede Ausgabe $\hat{y}_j$ als die Wahrscheinlichkeit interpretiert wird, dass ein bestimmtes Element zur Klasse $j$ gehört. Dann können wir die Klasse mit dem größten Ausgabewert als unsere Vorhersage $\operatorname*{argmax}_j y_j$ wählen. Zum Beispiel, wenn $\hat{y}_1$, $\hat{y}_2$ und $\hat{y}_3$ 0,1, 0,8 bzw. 0,1 sind, dann prognostizieren wir Kategorie 2, die (in unserem Beispiel) „Huhn“ darstellt.

Sie könnten versucht sein, vorzuschlagen, dass wir die Logits $o$ direkt als unsere Interesseergebnisse interpretieren. Es gibt jedoch einige Probleme bei der direkten Interpretation der Ausgabe des linearen Layers als Wahrscheinlichkeit. Auf der einen Seite beschränkt nichts diese Zahlen auf 1 zu summieren. Auf der anderen Seite können sie abhängig von den Eingaben negative Werte annehmen. Diese verletzen grundlegende Axiome der Wahrscheinlichkeit in :numref:`sec_prob`

Um unsere Ausgaben als Wahrscheinlichkeiten zu interpretieren, müssen wir garantieren, dass (auch bei neuen Daten) sie nicht negativ sind und bis zu 1 summieren. Darüber hinaus brauchen wir ein Ausbildungsziel, das das Modell dazu ermutigt, die Wahrscheinlichkeiten treu zu schätzen. Von allen Instanzen, in denen ein Klassifikator 0,5 ausgibt, hoffen wir, dass die Hälfte dieser Beispiele tatsächlich zur vorhergesagten Klasse gehört. Dies ist eine Eigenschaft namens *calibration*.

Genau das tut die *softmax-Funktion*, die 1959 vom Sozialwissenschaftler R. Duncan Luce im Kontext von *Choice-Models* erfunden wurde. Um unsere Logits so zu transformieren, dass sie nicht negativ werden und zu 1 summieren, während das Modell differenzierbar bleibt, exponentiieren wir zuerst jedes Logit (Gewährleistung der Nichtnegativität) und teilen dann durch seine Summe (Gewährleistung, dass sie auf 1 summieren):

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}. $$
:eqlabel:`eq_softmax_y_and_o`

Es ist leicht, $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$ mit $0 \leq \hat{y}_j \leq 1$ für alle $j$ zu sehen. Somit ist $\hat{\mathbf{y}}$ eine richtige Wahrscheinlichkeitsverteilung, deren Elementwerte entsprechend interpretiert werden können. Beachten Sie, dass der softmax-Vorgang die Reihenfolge zwischen den Logits $\mathbf{o}$ nicht ändert. Dies sind einfach die Pre-Softmax-Werte, die die Wahrscheinlichkeiten bestimmen, die jeder Klasse zugewiesen sind. Daher können wir während der Vorhersage immer noch die wahrscheinlichste Klasse auswählen, indem wir

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

Obwohl softmax eine nichtlineare Funktion ist, werden die Ausgänge der softmax-Regression immer noch *durch eine affine Transformation von Eingabe-Features bestimmt; daher ist die softmax-Regression ein lineares Modell.

## Vektorisierung für Minibatches
:label:`subsec_softmax_vectorization`

Um die Recheneffizienz zu verbessern und GPUs zu nutzen, führen wir in der Regel Vektorberechnungen für Minibatches von Daten durch. Angenommen, wir erhalten eine Minibatch $\mathbf{X}$ von Beispielen mit Feature-Dimensionalität (Anzahl der Eingänge) $d$ und Batch-Größe $n$. Darüber hinaus gehen wir davon aus, dass wir $q$ Kategorien in der Ausgabe haben. Dann sind die Minibatch Features $\mathbf{X}$ in $\mathbb{R}^{n \times d}$, Gewichte $\mathbf{W} \in \mathbb{R}^{d \times q}$, und die Neige erfüllt $\mathbf{b} \in \mathbb{R}^{1\times q}$.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

Dies beschleunigt die dominante Operation in ein Matrix-Matrix-Produkt $\mathbf{X} \mathbf{W}$ im Vergleich zu den Matrix-Vektor-Produkten, die wir ausführen würden, wenn wir jeweils ein Beispiel verarbeiten würden. Da jede Zeile in $\mathbf{X}$ einen Datenpunkt darstellt, kann die softmax-Operation selbst berechnet werden*rowwise*: Für jede Zeile von $\mathbf{O}$ exponentiieren Sie alle Einträge und normalisieren sie dann durch die Summe. Auslösen des Rundfunks während der Summierung $\mathbf{X} \mathbf{W} + \mathbf{b}$ in :eqref:`eq_minibatch_softmax_reg`, sowohl die Minibatch-Logits $\mathbf{O}$ als auch die Ausgangswahrscheinlichkeiten $\hat{\mathbf{Y}}$ sind $n \times q$ Matrizen.

## Verlust-Funktion

Als nächstes benötigen wir eine Verlustfunktion, um die Qualität unserer prognostizierten Wahrscheinlichkeiten zu messen. Wir werden uns auf die Maximalwahrscheinlichkeitsschätzung verlassen, das gleiche Konzept, das wir bei der Bereitstellung einer probabilistischen Begründung für das mittlere quadrierte Fehlerziel in der linearen Regression (:numref:`subsec_normal_distribution_and_squared_loss`) begegnet sind.

### Log-Likelihood

Die softmax-Funktion gibt uns einen Vektor $\hat{\mathbf{y}}$, den wir als geschätzte bedingte Wahrscheinlichkeiten jeder Klasse bei jeder Eingabe $\mathbf{x}$ interpretieren können, zB $\hat{y}_1$ = $P(y=\text{cat} \mid \mathbf{x})$. Angenommen, der gesamte Dataset $\{\mathbf{X}, \mathbf{Y}\}$ enthält $n$ Beispiele, wobei das durch $i$ indizierte Beispiel aus einem Feature-Vektor $\mathbf{x}^{(i)}$ und einem Ein-Hot-Beschriftungsvektor $\mathbf{y}^{(i)}$ besteht. Wir können die Schätzungen mit der Realität vergleichen, indem wir überprüfen, wie wahrscheinlich die tatsächlichen Klassen nach unserem Modell sind, angesichts der Merkmale:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

Gemäß der Maximalwahrscheinlichkeitsschätzung maximieren wir $P(\mathbf{Y} \mid \mathbf{X})$, was der Minimierung der negativen Log-Likelihood entspricht:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

wo für jedes Paar von Etikett $\mathbf{y}$ und Modellprognose $\hat{\mathbf{y}}$ über $q$ Klassen, die Verlustfunktion $l$ ist

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

Aus später erläuterten Gründen wird die Verlustfunktion in :eqref:`eq_l_cross_entropy` üblicherweise als *Cross-Entropieverlust* bezeichnet. Da $\mathbf{y}$ ein einheißer Vektor der Länge $q$ ist, verschwindet die Summe über alle seine Koordinaten $j$ für alle außer einem Begriff. Da alle $\hat{y}_j$ vorausgesagte Wahrscheinlichkeiten sind, ist ihr Logarithmus nie größer als $0$. Folglich kann die Verlustfunktion nicht weiter minimiert werden, wenn wir das tatsächliche Etikett korrekt mit *Gewissheit* vorhersagen, d.h. wenn die vorhergesagte Wahrscheinlichkeit $P(\mathbf{y} \mid \mathbf{x}) = 1$ für das tatsächliche Etikett $\mathbf{y}$. Beachten Sie, dass dies oft unmöglich ist. Beispielsweise könnte es Beschriftungsrauschen im Dataset geben (einige Beispiele sind möglicherweise falsch beschriftet). Es kann auch nicht möglich sein, wenn die Eingabe-Features nicht ausreichend informativ sind, um jedes Beispiel perfekt zu klassifizieren.

### Softmax und Derivate
:label:`subsec_softmax_and_derivatives`

Da Softmax und der entsprechende Verlust so häufig sind, lohnt es sich, ein bisschen besser zu verstehen, wie es berechnet wird. Stecken Sie :eqref:`eq_softmax_y_and_o` in die Definition des Verlustes in :eqref:`eq_l_cross_entropy` und verwenden Sie die Definition des softmax erhalten wir:

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

Um ein bisschen besser zu verstehen, was vor sich geht, betrachten Sie die Ableitung in Bezug auf jeden Logit $o_j$. Wir bekommen

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

Mit anderen Worten, die Ableitung ist die Differenz zwischen der von unserem Modell zugewiesenen Wahrscheinlichkeit, wie sie durch die softmax-Operation ausgedrückt wird, und dem, was tatsächlich passiert ist, wie sie von Elementen im ein-heißen Label-Vektor ausgedrückt wird. In diesem Sinne ist es sehr ähnlich zu dem, was wir in der Regression gesehen haben, wo der Gradienten war der Unterschied zwischen der Beobachtung $y$ und Schätzung $\hat{y}$. Das ist kein Zufall. In jeder exponentiellen Familie (siehe [online appendix on distributions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html)) Modell werden die Gradienten der Log-Likelihood durch genau diesen Begriff angegeben. Diese Tatsache macht die Berechnung von Gradienten in der Praxis einfach.

### Kreuzentropie-Verlust

Betrachten Sie nun den Fall, in dem wir nicht nur ein einziges Ergebnis, sondern eine ganze Verteilung über die Ergebnisse beobachten. Wir können die gleiche Darstellung wie zuvor für das Etikett $\mathbf{y}$ verwenden. Der einzige Unterschied besteht darin, dass wir anstelle eines Vektors, der nur binäre Einträge enthält, sagen $(0, 0, 1)$, jetzt einen generischen Wahrscheinlichkeitsvektor haben, sagen wir $(0.1, 0.2, 0.7)$. Die Mathematik, die wir zuvor verwendet haben, um den Verlust $l$ in :eqref:`eq_l_cross_entropy` zu definieren, funktioniert immer noch gut, nur dass die Interpretation etwas allgemeiner ist. Es ist der erwartete Wert des Verlustes für eine Verteilung über Etiketten. Dieser Verlust wird als *Cross-Entropieverlust* bezeichnet und ist einer der am häufigsten verwendeten Verluste für Klassifizierungsprobleme. Wir können den Namen entmystifizieren, indem wir nur die Grundlagen der Informationstheorie einführen. Wenn Sie mehr Details der Informationstheorie verstehen möchten, können Sie sich weiter auf die [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html) beziehen.

## Grundlagen der Informationstheorie

*Informationstheorie* befasst sich mit dem Problem der Kodierung, Decodierung, Übertragung,
und die Bearbeitung von Informationen (auch als Daten bezeichnet) in möglichst prägnanter Form.

### Entropie

Die zentrale Idee in der Informationstheorie ist die Quantifizierung des Informationsinhalts in Daten. Diese Menge setzt eine harte Begrenzung für unsere Fähigkeit, die Daten zu komprimieren. In der Informationstheorie wird diese Menge die *Entropie* einer Verteilung $p$ genannt, und sie wird durch die folgende Gleichung erfasst:

$$H[p] = \sum_j - p(j) \log p(j).$$
:eqlabel:`eq_softmax_reg_entropy`

Einer der grundlegenden Theoreme der Informationstheorie besagt, dass wir, um Daten, die zufällig aus der Verteilung $p$ gezogen werden, mindestens $H[p]$ „nats“ benötigen, um sie zu kodieren. Wenn Sie sich fragen, was ein „nat“ ist, ist es das Äquivalent von Bit, aber wenn Sie einen Code mit Basis $e$ anstelle eines mit Basis 2 verwenden. Somit ist ein nat $\frac{1}{\log(2)} \approx 1.44$ Bit.

### Überraschenderweise

Vielleicht fragen Sie sich, was die Komprimierung mit der Vorhersage zu tun hat. Stellen Sie sich vor, wir haben einen Datenstrom, den wir komprimieren möchten. Wenn es für uns immer einfach ist, das nächste Token vorherzusagen, dann sind diese Daten einfach zu komprimieren! Nehmen Sie das extreme Beispiel, bei dem jedes Token im Stream immer den gleichen Wert annimmt. Das ist ein sehr langweiliger Datenstrom! Und nicht nur es ist langweilig, sondern es ist auch leicht vorherzusagen. Da sie immer gleich sind, müssen wir keine Informationen übermitteln, um den Inhalt des Streams zu kommunizieren. Einfach vorherzusagen, einfach zu komprimieren.

Wenn wir jedoch nicht jedes Ereignis perfekt vorhersagen können, dann werden wir manchmal überrascht sein. Unsere Überraschung ist größer, wenn wir ein Ereignis geringere Wahrscheinlichkeit zugewiesen. Claude Shannon ließ sich am $\log \frac{1}{P(j)} = -\log P(j)$ nieder, um seine *überraschend* bei der Beobachtung eines Ereignisses $j$ zu quantifizieren, nachdem er ihm eine (subjektive) Wahrscheinlichkeit $P(j)$ zugewiesen hatte. Die Entropie, die in :eqref:`eq_softmax_reg_entropy` definiert ist, ist dann die *erwartete Überraschung*, wenn man die richtigen Wahrscheinlichkeiten zugewiesen hat, die wirklich mit dem Datengenerierungsprozess übereinstimmen.

### Kreuzentropie Revisited

Wenn die Entropie also ein Überraschungsgrad ist, der von jemandem erfahren wird, der die wahre Wahrscheinlichkeit kennt, dann fragen Sie sich vielleicht, was ist Kreuzentropie? Die Kreuzentropie *von* $p$ *bis* $q$, bezeichnet $H(p, q)$, ist die erwartete Überraschung eines Beobachters mit subjektiven Wahrscheinlichkeiten $q$, wenn Daten gesehen wurden, die tatsächlich nach Wahrscheinlichkeiten $p$ generiert wurden. Die niedrigste mögliche Kreuzentropie wird erreicht, wenn $p=q$. In diesem Fall beträgt die Kreuzentropie von $p$ bis $q$ $H(p, p)= H(p)$.

Kurz gesagt, wir können uns das Ziel der Cross-Entropie-Klassifikation auf zwei Arten vorstellen: (i) als Maximierung der Wahrscheinlichkeit der beobachteten Daten; und (ii) als Minimierung unserer Überraschung (und damit die Anzahl der Bits), die für die Kommunikation der Labels erforderlich sind.

## Modellprognose und -auswertung

Nach dem Training des softmax-Regressionsmodells können wir anhand von Beispielfunktionen die Wahrscheinlichkeit jeder Ausgabeklasse vorhersagen. Normalerweise verwenden wir die Klasse mit der höchsten prognostizierten Wahrscheinlichkeit als Ausgabeklasse. Die Vorhersage ist korrekt, wenn sie mit der tatsächlichen Klasse (Label) konsistent ist. Im nächsten Teil des Experiments werden wir *genauigkeit* verwenden, um die Leistung des Modells zu bewerten. Dies ist gleich dem Verhältnis zwischen der Anzahl der korrekten Vorhersagen und der Gesamtzahl der Vorhersagen.

## Zusammenfassung

* Die softmax-Operation nimmt einen Vektor und ordnet ihn in Wahrscheinlichkeiten zu.
* Die Softmax-Regression gilt für Klassifizierungsprobleme. Es verwendet die Wahrscheinlichkeitsverteilung der Ausgabeklasse in der softmax-Operation.
* Die Cross-Entropie ist ein gutes Maß für die Differenz zwischen zwei Wahrscheinlichkeitsverteilungen. Es misst die Anzahl der Bits, die benötigt werden, um die Daten zu codieren, die unser Modell gegeben.

## Übungen

1. Wir können die Verbindung zwischen exponentiellen Familien und dem softmax noch tiefer untersuchen.
    * Berechnen Sie die zweite Ableitung des Kreuzentropienverlustes $l(\mathbf{y},\hat{\mathbf{y}})$ für den Softmax.
    * Berechnen Sie die Varianz der durch $\mathrm{softmax}(\mathbf{o})$ angegebenen Verteilung und zeigen Sie, dass sie mit der zweiten oben berechneten Ableitung übereinstimmt.
1. Angenommen, wir haben drei Klassen, die mit gleicher Wahrscheinlichkeit auftreten, dh der Wahrscheinlichkeitsvektor ist $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    * Was ist das Problem, wenn wir versuchen, einen Binärcode dafür zu entwerfen?
    * Können Sie einen besseren Code entwerfen? Hinweis: Was passiert, wenn wir versuchen, zwei unabhängige Beobachtungen zu kodieren? Was ist, wenn wir $n$ Beobachtungen gemeinsam kodieren?
1. Softmax ist eine falsche Nomer für das oben eingeführte Mapping (aber jeder im Deep Learning verwendet es). Der echte Softmax ist definiert als $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    * Beweisen Sie das $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    * Beweisen Sie, dass dies gilt für $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, vorausgesetzt, $\lambda > 0$.
    * Zeigen Sie, dass wir $\lambda \to \infty$ $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$ haben.
    * Wie sieht der Soft-Min aus?
    * Erweitern Sie dies auf mehr als zwei Zahlen.

[Discussions](https://discuss.d2l.ai/t/46)
