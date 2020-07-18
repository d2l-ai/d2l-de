# Umwelt- und Verteilungswechsel

In den vorangegangenen Abschnitten haben wir eine Reihe von praktischen Anwendungen des maschinellen Lernens durchgearbeitet, die Modelle an eine Vielzahl von Datensätzen anpassten. Und doch haben wir nie aufgehört, darüber nachzudenken, woher Daten überhaupt kommen oder was wir letztendlich mit den Ausgängen unserer Modelle machen wollen. Zu oft eilen Machine Learning-Entwickler im Besitz von Daten, um Modelle zu entwickeln, ohne anzuhalten, um diese grundlegenden Probleme zu berücksichtigen.

Viele fehlgeschlagene Machine Learning-Bereitstellungen können auf dieses Muster zurückverfolgt werden. Manchmal scheinen Modelle wunderbar zu funktionieren, wie sie durch die Genauigkeit des Testsatzes gemessen werden, scheitern aber bei der Bereitstellung katastrophal, wenn sich die Verteilung der Daten plötzlich verschiebt. Noch heimtückischer ist, dass die Bereitstellung eines Modells manchmal der Katalysator sein kann, der die Datenverteilung unterwirrt. Sagen wir zum Beispiel, dass wir ein Modell ausgebildet haben, um vorherzusagen, wer zurückzahlen wird gegen Ausfall eines Darlehens, wobei festgestellt wird, dass die Wahl der Schuhe eines Antragstellers mit dem Risiko eines Ausfalls verbunden war (Oxfords zeigen Rückzahlung, Turnschuhe zeigen Standard an). Wir könnten dazu neigen, danach allen Bewerbern, die Oxfords tragen, Kredite zu gewähren und allen Bewerbern, die Turnschuhe tragen, zu verweigern.

In diesem Fall könnte unser unüberlegter Sprung von der Mustererkennung zur Entscheidungsfindung und unser Versäumnis, die Umwelt kritisch zu betrachten, katastrophale Folgen haben. Zunächst einmal, sobald wir begannen, Entscheidungen auf der Grundlage von Schuhen zu treffen, würden die Kunden ihr Verhalten auffangen und ändern. Früher würden alle Bewerber Oxfords tragen, ohne dass eine gleichzeitige Verbesserung der Kreditwürdigkeit erzielt wird. Nehmen Sie sich eine Minute Zeit, um dies zu verdauen, weil ähnliche Probleme in vielen Anwendungen des maschinellen Lernens vorhanden sind: Durch die Einführung unserer modellbasierten Entscheidungen in die Umgebung könnten wir das Modell brechen.

Wir können diesen Themen zwar nicht in einem Abschnitt eine vollständige Behandlung geben, aber wir wollen hier einige gemeinsame Bedenken aufdecken und das kritische Denken stimulieren, das erforderlich ist, um diese Situationen frühzeitig zu erkennen, Schäden zu mildern und maschinelles Lernen verantwortungsvoll zu nutzen. Einige der Lösungen sind einfach (fragen Sie nach den „richtigen“ Daten), einige sind technisch schwierig (implementieren Sie ein Verstärkungslernsystem), andere erfordern, dass wir ganz außerhalb des Bereichs der statistischen Vorhersage treten und sich mit schwierigen philosophischen Fragen bezüglich der ethischen Anwendung von Algorithmen.

## Arten der Verteilungsschicht

Zunächst bleiben wir bei der passiven Vorhersageeinstellung, wobei wir die verschiedenen Möglichkeiten berücksichtigen, mit denen sich Datenverteilungen verändern können, und was getan werden könnte, um die Modellleistung zu retten. In einem klassischen Setup gehen wir davon aus, dass unsere Trainingsdaten aus einer Distribution $p_S(\mathbf{x},y)$ entnommen wurden, aber dass unsere Testdaten aus unbeschrifteten Beispielen bestehen, die aus einer anderen Distribution $p_T(\mathbf{x},y)$ stammen. Schon jetzt müssen wir uns einer ernüchternden Realität stellen. Ohne Annahmen darüber, wie sich $p_S$ und $p_T$ zueinander beziehen, ist das Erlernen eines robusten Klassifikators unmöglich.

Betrachten Sie ein binäres Klassifizierungsproblem, bei dem wir zwischen Hunden und Katzen unterscheiden möchten. Wenn sich die Verteilung beliebig verschieben kann, erlaubt unser Setup den pathologischen Fall, in dem die Verteilung über Eingänge konstant bleibt: $p_S(\mathbf{x}) = p_T(\mathbf{x})$, aber die Etiketten sind alle umgedreht: $p_S(y | \mathbf{x}) = 1 - p_T(y | \mathbf{x})$. Mit anderen Worten, wenn Gott plötzlich entscheiden kann, dass in Zukunft alle „Katzen“ jetzt Hunde sind und was wir zuvor „Hunde“ genannt haben, jetzt Katzen sind — ohne jede Änderung der Verteilung der Eingaben $p(\mathbf{x})$, dann können wir diese Einstellung nicht von einer unterscheiden, in der sich die Verteilung überhaupt nicht geändert hat.

Glücklicherweise können prinzipielle Algorithmen unter einigen eingeschränkten Annahmen über die Art und Weise, wie sich unsere Daten in der Zukunft ändern könnten, Verschiebungen erkennen und manchmal sogar im laufenden Betrieb anpassen, was die Genauigkeit des ursprünglichen Klassifikators verbessert.

### Kovariate Verschiebung

Unter den Kategorien der Verteilungsverschiebung kann die kovariate Verschiebung am weitesten untersucht werden. Hier gehen wir davon aus, dass sich die Verteilung der Eingaben im Laufe der Zeit ändern kann, die Beschriftungsfunktion, d.h., die bedingte Verteilung $P(y \mid \mathbf{x})$ ändert sich nicht. Statistiker nennen diese *kovariate shift*, da das Problem aufgrund einer Verschiebung der Verteilung der Kovariaten (Features) auftritt. Während wir manchmal über Verteilungsverschiebung Grund haben können, ohne Kausalität aufzurufen, stellen wir fest, dass kovariate Verschiebung die natürliche Annahme ist, in Einstellungen aufzurufen, in denen wir glauben, dass $\mathbf{x}$ $y$ verursacht.

Betrachten Sie die Herausforderung, Katzen und Hunde zu unterscheiden. Unsere Trainingsdaten können aus Bildern der Art in :numref:`fig_cat-dog-train` bestehen.

![Training data for distinguishing cats and dogs.](../img/cat-dog-train.svg)
:label:`fig_cat-dog-train`

Zur Testzeit werden wir gebeten, die Bilder in :numref:`fig_cat-dog-test` zu klassifizieren.

![Test data for distinguishing cats and dogs.](../img/cat-dog-test.svg)
:label:`fig_cat-dog-test`

Das Trainingsset besteht aus Fotos, während das Testset nur Cartoons enthält. Schulungen zu einem Datensatz mit wesentlich anderen Eigenschaften als dem Testsatz können Probleme ohne einen kohärenten Plan für die Anpassung an die neue Domäne buchstabieren.

### Beschriftung-Shift

*Label shift* beschreibt das umgekehrte Problem.
Hier gehen wir davon aus, dass sich die Beschriftung marginal $P(y)$ ändern kann, aber die klassenbedingte Verteilung $P(\mathbf{x} \mid y)$ bleibt domänenübergreifend fixiert. Etikettenverschiebung ist eine vernünftige Annahme zu machen, wenn wir glauben, dass $y$ $\mathbf{x}$ verursacht. Zum Beispiel möchten wir möglicherweise Diagnosen aufgrund ihrer Symptome (oder anderer Manifestationen) vorhersagen, auch wenn sich die relative Prävalenz von Diagnosen im Laufe der Zeit ändert. Die Labelverschiebung ist hier die angemessene Annahme, weil Krankheiten Symptome verursachen. In einigen degenerierten Fällen können die Annahmen der Beschriftungsverschiebung und Kovariate Verschiebung gleichzeitig gehalten werden. Wenn die Beschriftung beispielsweise deterministisch ist, wird die Annahme der Kovariatenverschiebung erfüllt, selbst wenn $y$ $\mathbf{x}$ verursacht. Interessanterweise ist es in diesen Fällen oft vorteilhaft, mit Methoden zu arbeiten, die aus der Annahme der Beschriftungsverschiebung fließen. Das liegt daran, dass diese Methoden dazu neigen, Objekte zu manipulieren, die wie Labels aussehen (oft niederdimensional), im Gegensatz zu Objekten, die wie Eingaben aussehen, die im Deep Learning eher hochdimensional sind.

### Konzept Verschiebung

Wir können auch auf das damit verbundene Problem von *concept shift* stoßen, das entsteht, wenn sich die Definitionen von Labels ändern können. Das klingt seltsam — eine *Katze* ist eine *Katze*, nicht wahr? Andere Kategorien unterliegen jedoch Änderungen in der Nutzung im Laufe der Zeit. Diagnostische Kriterien für Geisteskrankheiten, was für Mode gilt, und Jobtitel unterliegen erheblichen Mengen an Konzeptverlagerung. Es stellt sich heraus, dass, wenn wir durch die Vereinigten Staaten navigieren und die Quelle unserer Daten nach Geographie verschieben, wir eine beträchtliche Konzeptverschiebung in Bezug auf die Verteilung der Namen für *Softdrinks* finden werden, wie in :numref:`fig_popvssoda` gezeigt.

![Concept shift on soft drink names in the United States.](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

Wenn wir ein maschinelles Übersetzungssystem bauen würden, könnte die Verteilung $P(y \mid \mathbf{x})$ je nach Standort unterschiedlich sein. Dieses Problem kann schwierig sein, zu erkennen. Wir hoffen vielleicht, das Wissen zu nutzen, dass die Verschiebung nur allmählich erfolgt, entweder in einem zeitlichen oder geographischen Sinn.

## Beispiele für Distributionsverschiebung

Bevor wir uns mit Formalismus und Algorithmen befassen, können wir einige konkrete Situationen diskutieren, in denen Kovariate oder Konzeptverschiebung möglicherweise nicht offensichtlich sind.

### Medizinische Diagnostik

Stellen Sie sich vor, Sie möchten einen Algorithmus zur Erkennung von Krebs entwerfen. Sie sammeln Daten von gesunden und kranken Menschen und trainieren Ihren Algorithmus. Es funktioniert gut, gibt Ihnen eine hohe Genauigkeit und Sie schließen, dass Sie bereit für eine erfolgreiche Karriere in der medizinischen Diagnostik sind.
*Nicht so schnell.*

Die Distributionen, die zu den Trainingsdaten geführt haben und denen, die Sie in freier Wildbahn begegnen, können sich erheblich unterscheiden. Dies geschah mit einem unglücklichen Startup, mit dem einige von uns (Autoren) vor Jahren gearbeitet haben. Sie entwickelten einen Bluttest für eine Krankheit, die überwiegend ältere Männer betrifft, und hofften, sie anhand von Blutproben zu untersuchen, die sie von Patienten gesammelt hatten. Es ist jedoch wesentlich schwieriger, Blutproben von gesunden Männern zu erhalten als kranke Patienten, die sich bereits im System befinden. Um zu kompensieren, forderte das Startup Blutspenden von Studenten auf einem Universitätscampus, um als gesunde Kontrolle bei der Entwicklung ihres Tests zu dienen. Dann fragten sie, ob wir ihnen helfen könnten, einen Klassifikator für die Erkennung der Krankheit zu bauen.

Wie wir ihnen erklärt haben, wäre es in der Tat leicht, zwischen den gesunden und kranken Kohorten mit nahezu perfekter Genauigkeit zu unterscheiden. Das liegt jedoch daran, dass sich die Testpersonen in Alter, Hormonspiegel, körperlicher Aktivität, Ernährung, Alkoholkonsum und vielen weiteren Faktoren unterschieden, die nicht mit der Krankheit zusammenhängen. Dies war unwahrscheinlich, dass es bei echten Patienten der Fall ist. Aufgrund ihres Sampling-Verfahrens könnten wir erwarten, dass eine extreme Kovariatenverschiebung auftritt. Darüber hinaus war es unwahrscheinlich, dass dieser Fall mit herkömmlichen Methoden korrigierbar ist. Kurz gesagt, sie verschwendeten eine beträchtliche Summe Geld.

### Selbstfahrende Autos

Angenommen, ein Unternehmen wollte maschinelles Lernen nutzen, um selbstfahrende Autos zu entwickeln. Eine Schlüsselkomponente ist hier ein Straßendetektor. Da echte kommentierte Daten teuer sind, hatten sie die (intelligente und fragwürdige) Idee, synthetische Daten aus einer Spiel-Rendering-Engine als zusätzliche Trainingsdaten zu verwenden. Dies funktionierte wirklich gut bei „Testdaten“, die aus der Rendering-Engine gezogen wurden. Ach, in einem echten Auto war es eine Katastrophe. Wie sich herausstellte, war der Straßenrand mit einer sehr einfachen Textur gerendert worden. Noch wichtiger ist, dass *all* der Straßenrand mit der *gleiche* Textur gerendert wurde und der Straßenseiten-Detektor sehr schnell von diesem „Feature“ erfuhr.

Ähnliches passierte der US-Armee, als sie zum ersten Mal versuchten, Panzer im Wald zu erkennen. Sie machten Luftaufnahmen des Waldes ohne Panzer, fuhren dann die Tanks in den Wald und machten weitere Bilder. Der Klassifikator schien *perfekt* zu funktionieren. Leider hatte sie nur gelernt, Bäume mit Schatten von Bäumen ohne Schatten zu unterscheiden — die erste Reihe von Bildern wurde am frühen Morgen aufgenommen, der zweite Satz am Mittag.

### Nichtstationäre Verteilungen

Eine viel subtilere Situation entsteht, wenn sich die Verteilung langsam ändert (auch bekannt als *nichtstationäre Verteilung*) und das Modell nicht adäquat aktualisiert wird. Im Folgenden finden Sie einige typische Fälle.

* Wir trainieren ein rechnerisches Werbemodell und aktualisieren es dann nicht häufig (z. B. vergessen wir zu integrieren, dass gerade ein obskures neues Gerät namens iPad gestartet wurde).
* Wir bauen einen Spam-Filter. Es funktioniert gut, alle Spam zu erkennen, die wir bisher gesehen haben. Aber dann sind die Spammer aufgetaucht und erstellen neue Nachrichten, die anders aussehen als alles, was wir zuvor gesehen haben.
* Wir bauen ein Produktempfehlungssystem. Es funktioniert den ganzen Winter, aber dann weiterhin Santa Hüte lange nach Weihnachten empfehlen.

### Mehr Anekdoten

* Wir bauen einen Gesichtsdetektor. Es funktioniert gut auf allen Benchmarks. Leider scheitert es bei Testdaten — die beleidigensten Beispiele sind Nahaufnahmen, bei denen das Gesicht das gesamte Bild füllt (keine solchen Daten waren im Trainingsset).
* Wir bauen eine Web-Suchmaschine für den US-Markt und wollen sie in Großbritannien einsetzen.
* Wir trainieren einen Bildklassifikator, indem wir einen großen Datensatz kompilieren, bei dem jeder von einem großen Satz von Klassen gleichmäßig im Datensatz dargestellt wird, beispielsweise 1000 Kategorien, dargestellt durch jeweils 1000 Bilder. Dann setzen wir das System in der realen Welt ein, wo die tatsächliche Etikettenverteilung von Fotografien eindeutig ungleichmäßig ist.

## Korrektur der Verteilungsverschiebung

Wie wir besprochen haben, gibt es viele Fälle, in denen Schulungs- und Testverteilungen $P(\mathbf{x}, y)$ unterschiedlich sind. In einigen Fällen haben wir Glück und die Modelle funktionieren trotz Kovariaten-, Label- oder Konzeptverschiebung. In anderen Fällen können wir es besser machen, indem wir prinzipienorientierte Strategien einsetzen, um mit dem Wandel fertig zu werden. Der Rest dieses Abschnitts wächst wesentlich technischer. Der ungeduldige Leser könnte mit dem nächsten Abschnitt fortfahren, da dieses Material keine Voraussetzung für nachfolgende Konzepte ist.

### Empirisches Risiko und echtes Risiko

Lassen Sie uns zuerst darüber nachdenken, was genau während des Modelltrainings passiert: Wir iterieren über Features und zugehörige Beschriftungen von Trainingsdaten $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ und aktualisieren die Parameter eines Modells $f$ nach jedem Minibatch. Der Einfachheit halber betrachten wir Regularisierung nicht, so dass wir den Verlust auf dem Training weitgehend minimieren:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n l(f(\mathbf{x}_i), y_i),$$
:eqlabel:`eq_empirical-risk-min`

wobei $l$ die Verlustfunktion ist, die „wie schlecht“ misst, wird die Vorhersage $f(\mathbf{x}_i)$ das zugehörige Etikett $y_i$ gegeben. Statistiker nennen den Begriff in :eqref:`eq_empirical-risk-min` *empirisches Risiko*.
*Empirisches Risiko* ist ein durchschnittlicher Verlust gegenüber den Trainingsdaten
Annäherung an das *wahre Risiko*, das ist die Erwartung des Verlustes von Daten aus ihrer tatsächlichen Verteilung über die gesamte Population $p(\mathbf{x},y)$:

$$E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)] = \int\int l(f(\mathbf{x}), y) p(\mathbf{x}, y) \;d\mathbf{x}dy.$$
:eqlabel:`eq_true-risk`

In der Praxis können wir jedoch in der Regel nicht die gesamte Datenpopulation erhalten. So ist *empirische Risikominimierung*, die empirische Risiken in :eqref:`eq_empirical-risk-min` minimiert, eine praktische Strategie für maschinelles Lernen, mit der Hoffnung, die Minimierung des wahren Risikos annähernd zu minimieren.

### Korrektur der Kovariate Verschiebung
:label:`subsec_covariate-shift-correction`

Angenommen, wir möchten eine Abhängigkeit $P(y \mid \mathbf{x})$ schätzen, für die wir Daten $(\mathbf{x}_i, y_i)$ beschriftet haben. Leider stammen die Beobachtungen $\mathbf{x}_i$ aus irgendeiner *Quellenverteilung* $q(\mathbf{x})$ und nicht aus der *Zielverteilung* $p(\mathbf{x})$. Glücklicherweise bedeutet die Annahme der Abhängigkeit, dass sich die bedingte Verteilung nicht ändert: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$. Wenn die Quellverteilung $q(\mathbf{x})$ „falsch“ ist, können wir dies korrigieren, indem wir die folgende einfache Identität in echtes Risiko verwenden:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy = 
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy.
\end{aligned}
$$

Mit anderen Worten, wir müssen jeden Datenpunkt nach dem Verhältnis der Wahrscheinlichkeit neu abwägen, dass er von der richtigen Verteilung zu der von der falschen gezogenen Verteilung gezogen worden wäre:

$$\beta_i \stackrel{\mathrm{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}.$$

Einstecken des Gewichts $\beta_i$ für jeden Datenpunkt $(\mathbf{x}_i, y_i)$ können wir unser Modell mit
*gewichtete empirische Risikominimierung*:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i).$$
:eqlabel:`eq_weighted-empirical-risk-min`

Leider wissen wir dieses Verhältnis nicht, also bevor wir etwas Nützliches tun können, müssen wir es schätzen. Es stehen viele Methoden zur Verfügung, darunter einige ausgefallene operatortheoretische Ansätze, die versuchen, den Erwartungsoperator direkt anhand eines Minimum-Norm- oder eines Maximal-Entropie-Prinzips neu zu kalibrieren. Beachten Sie, dass wir für einen solchen Ansatz Proben benötigen, die aus beiden Distributionen gezogen werden — die „wahre“ $p$, z. B. durch Zugriff auf Testdaten, und die zum Generieren des Trainingssets $q$ verwendet werden (letzteres ist trivial verfügbar). Beachten Sie jedoch, dass wir nur Funktionen $\mathbf{x} \sim p(\mathbf{x})$ benötigen; wir müssen nicht auf die Etiketten $y \sim p(y)$ zugreifen.

In diesem Fall gibt es einen sehr effektiven Ansatz, der fast genauso gute Ergebnisse liefert wie das Original: die logistische Regression, die ein Sonderfall der Softmax-Regression für die binäre Klassifikation ist. Dies ist alles, was benötigt wird, um geschätzte Wahrscheinlichkeitsverhältnisse zu berechnen. Wir lernen einen Klassifikator, um zwischen Daten aus $p(\mathbf{x})$ und Daten aus $q(\mathbf{x})$ zu unterscheiden. Wenn es unmöglich ist, zwischen den beiden Distributionen zu unterscheiden, bedeutet dies, dass die zugehörigen Instanzen gleichermaßen wahrscheinlich aus einer der beiden Distributionen stammen. Andererseits sollten alle Fälle, die gut diskriminiert werden können, entsprechend signifikant übergewichtet oder untergewichtet werden. Der Einfachheit halber wird davon ausgegangen, dass wir eine gleiche Anzahl von Instanzen aus beiden Distributionen $p(\mathbf{x})$ bzw. $q(\mathbf{x})$ haben. Bezeichnen Sie nun mit $z$ Etiketten, die $1$ für Daten aus $p$ und $-1$ für Daten aus $q$ gezogen sind. Dann wird die Wahrscheinlichkeit in einem gemischten Dataset durch

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ and hence } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Wenn wir also einen logistischen Regressionsansatz verwenden, bei dem $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-h(\mathbf{x}))}$ ($h$ eine parametrisierte Funktion ist), folgt, dass

$$
\beta_i = \frac{1/(1 + \exp(-h(\mathbf{x}_i)))}{\exp(-h(\mathbf{x}_i))/(1 + \exp(-h(\mathbf{x}_i)))} = \exp(h(\mathbf{x}_i)).
$$

Als Ergebnis müssen wir zwei Probleme lösen: Erstens, um zwischen Daten aus beiden Verteilungen zu unterscheiden, und dann ein gewichtetes empirisches Risikominimierungsproblem in :eqref:`eq_weighted-empirical-risk-min`, wo wir Terme mit $\beta_i$ wiegen.

Jetzt sind wir bereit, einen Korrekturalgorithmus zu beschreiben. Angenommen, wir haben ein Trainingsset $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ und ein unbeschriftetes Testset $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$. Für die Kovariatenverschiebung gehen wir davon aus, dass $\mathbf{x}_i$ für alle $1 \leq i \leq n$ aus einer Quellverteilung gezogen werden und $\mathbf{u}_i$ für alle $1 \leq i \leq m$ aus der Zielverteilung gezogen werden. Hier ist ein prototypischer Algorithmus zur Korrektur der Kovariatenverschiebung:

1. Generieren Sie ein Schulungsset zur Binärklassifizierung: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1), \ldots, (\mathbf{u}_m, 1)\}$.
1. Trainieren Sie einen binären Klassifikator mit logistischer Regression, um die Funktion $h$ zu erhalten.
1. Wiegen Sie Trainingsdaten mit $\beta_i = \exp(h(\mathbf{x}_i))$ oder besser $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$ für einige konstante $c$.
1. Verwenden Sie Gewichte $\beta_i$ für das Training auf $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ in :eqref:`eq_weighted-empirical-risk-min`.

Beachten Sie, dass der obige Algorithmus auf einer entscheidenden Annahme beruht. Damit dieses Schema funktioniert, müssen wir, dass jeder Datenpunkt in der Zielverteilung (z. B. Testzeit) eine Wahrscheinlichkeit ungleich Null hatte, zur Trainingszeit aufzutreten, Wenn wir einen Punkt finden, an dem $p(\mathbf{x}) > 0$ aber $q(\mathbf{x}) = 0$, dann sollte die entsprechende Wichtigkeit Gewicht unendlich sein.

### Korrektur der Beschriftungsverschiebung

Angenommen, wir haben eine Klassifizierungsaufgabe mit $k$ Kategorien zu tun. Mit der gleichen Notation in :numref:`subsec_covariate-shift-correction`, $q$ und $p$ werden die Quellverteilung (z.B. Trainingszeit) bzw. die Zielverteilung (z.B. Testzeit) verwendet. Angenommen, die Verteilung der Beschriftungen im Laufe der Zeit verschiebt: $q(y) \neq p(y)$, aber die Klassenbedingte Verteilung bleibt gleich: $q(\mathbf{x} \mid y)=p(\mathbf{x} \mid y)$. Wenn die Quellverteilung $q(y)$ „falsch“ ist, können wir dies entsprechend der folgenden Identität in echtes Risiko korrigieren, wie in :eqref:`eq_true-risk` definiert:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(\mathbf{x} \mid y)p(y) \;d\mathbf{x}dy = 
\int\int l(f(\mathbf{x}), y) q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)} \;d\mathbf{x}dy.
\end{aligned}
$$

Hier entsprechen unsere Wichtigkeitsgewichte den Kennzeichnungswahrscheinlichkeitsverhältnissen

$$\beta_i \stackrel{\mathrm{def}}{=} \frac{p(y_i)}{q(y_i)}.$$

Eine nette Sache an der Etikettenverschiebung ist, dass wir, wenn wir ein relativ gutes Modell für die Quellverteilung haben, konsistente Schätzungen dieser Gewichte erhalten können, ohne jemals mit der Umgebungsdimension umgehen zu müssen. Beim Deep Learning sind die Eingaben in der Regel hochdimensionale Objekte wie Bilder, während die Beschriftungen oft einfachere Objekte wie Kategorien sind.

Um die Ziellabel-Verteilung zu schätzen, nehmen wir zuerst unseren relativ guten Standard-Klassifikator (typischerweise auf den Trainingsdaten trainiert) und berechnen seine Verwirrungsmatrix mithilfe des Validierungssatzes (auch aus der Schulungsverteilung). Die *Verwirrungsmatrix*, $\mathbf{C}$, ist einfach eine $k \times k$ Matrix, wobei jede Spalte der Beschriftungskategorie entspricht (Grundwahrheit) und jede Zeile der prognostizierten Kategorie unseres Modells entspricht. Der Wert jeder Zelle $c_{ij}$ ist der Bruchteil der Gesamtprognosen auf dem Validierungssatz, bei dem die wahre Beschriftung $j$ betrug und unser Modell $i$ vorausgesagt hat.

Nun können wir die Verwirrungsmatrix für die Zieldaten nicht direkt berechnen, da wir die Beschriftungen für die Beispiele, die wir in freier Wildbahn sehen, nicht sehen können, es sei denn, wir investieren in eine komplexe Echtzeit-Annotation-Pipeline. Was wir jedoch tun können, ist durchschnittlich alle unsere Modelle Vorhersagen zur Testzeit zusammen, was die mittleren Modellausgänge $\mu(\hat{\mathbf{y}}) \in \mathbb{R}^k$ ergibt, deren $i^\mathrm{th}$ Element $\mu(\hat{y}_i)$ der Bruchteil der Gesamtprognosen auf dem Testsatz ist, wo unser Modell $i$ vorhergesagt hat.

Es stellt sich heraus, dass unter einigen milden Bedingungen — wenn unser Klassifikator in erster Linie ziemlich genau war und wenn die Zieldaten nur Kategorien enthalten, die wir zuvor gesehen haben, und wenn die Annahme der Beschriftungsverschiebung in erster Linie gilt (die stärkste Annahme hier), dann können wir das Testsatz-Label abschätzen. Verteilung durch Lösen eines einfachen linearen Systems

$$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}}),$$

da $\sum_{j=1}^k c_{ij} p(y_j) = \mu(\hat{y}_i)$ als Schätzung für alle $1 \leq i \leq k$ gilt, wobei $p(y_j)$ das $j^\mathrm{th}$-Element des $k$-dimensionalen Etikettenverteilungsvektors $p(\mathbf{y})$ ist. Wenn unser Klassifikator zunächst ausreichend genau ist, wird die Verwirrungsmatrix $\mathbf{C}$ umkehrbar sein, und wir erhalten eine Lösung $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$.

Da wir die Etiketten auf den Quelldaten beobachten, ist es leicht, die Verteilung $q(y)$ zu schätzen. Dann können wir für jedes Trainingsbeispiel $i$ mit dem Etikett $y_i$ das Verhältnis unserer geschätzten $p(y_i)/q(y_i)$ nehmen, um das Gewicht $\beta_i$ zu berechnen, und dies in die gewichtete empirische Risikominimierung in :eqref:`eq_weighted-empirical-risk-min` einstecken.

### Concept Shift-Korrektur

Die Konzeptverschiebung ist viel schwieriger, prinzipiell zu beheben. In einer Situation, in der sich plötzlich das Problem von der Unterscheidung von Katzen von Hunden zu einer Unterscheidung von Weiß von schwarzen Tieren ändert, wird es unvernünftig sein, anzunehmen, dass wir viel besser machen können, als nur neue Etiketten zu sammeln und von Grund auf neu zu trainieren. Glücklicherweise sind solche extremen Verschiebungen in der Praxis selten. Stattdessen passiert normalerweise, dass sich die Aufgabe langsam ändert. Um die Dinge konkreter zu machen, sind hier einige Beispiele:

* In der computergestützten Werbung werden neue Produkte auf den Markt gebracht,
alte Produkte werden weniger populär. Dies bedeutet, dass sich die Verteilung über Anzeigen und ihre Popularität allmählich ändert und jeder Click-Through-Rate Prädiktor muss sich schrittweise damit ändern.
* Verkehrskamera-Objektive verschlechtern sich allmählich aufgrund von Umweltverschleiß, was die Bildqualität schrittweise beeinträchtigt.
* Der Inhalt der Nachrichten ändert sich allmählich (d. h., die meisten Nachrichten bleiben unverändert, aber neue Geschichten erscheinen).

In solchen Fällen können wir den gleichen Ansatz verwenden, den wir für Schulungsnetzwerke verwendet haben, um sie an die Änderung der Daten anzupassen. Mit anderen Worten, wir verwenden die vorhandenen Netzwerkgewichte und führen einfach ein paar Aktualisierungsschritte mit den neuen Daten durch, anstatt von Grund auf neu zu trainieren.

## Eine Taxonomie von Lernproblemen

Bewaffnet mit Wissen darüber, wie man mit Veränderungen in Distributionen umgeht, können wir nun einige andere Aspekte der Formulierung von maschinellem Lernen berücksichtigen.

### Batch-Lernen

In *batch learning* haben wir Zugriff auf Trainingsfunktionen und Labels $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$, die wir verwenden, um ein Modell $f(\mathbf{x})$ zu trainieren. Später stellen wir dieses Modell bereit, um neue Daten $(\mathbf{x}, y)$ aus derselben Distribution zu bewerten. Dies ist die Standardannahme für eines der Probleme, die wir hier diskutieren. Zum Beispiel könnten wir einen Katzendetektor trainieren, der auf vielen Bildern von Katzen und Hunden basiert. Sobald wir es trainiert haben, versenden wir es als Teil eines intelligenten Catdoor Computer-Vision-Systems, das nur Katzen reinlässt. Diese wird dann bei einem Kunden zu Hause installiert und wird nie wieder aktualisiert (abgesehen von extremen Umständen).

### Online-Lernen

Stellen Sie sich nun vor, dass die Daten $(\mathbf{x}_i, y_i)$ jeweils eine Probe ankommen. Genauer gesagt, nehmen wir an, dass wir zuerst $\mathbf{x}_i$ beobachten, dann müssen wir mit einer Schätzung $f(\mathbf{x}_i)$ kommen und erst wenn wir dies getan haben, beobachten wir $y_i$ und damit erhalten wir eine Belohnung oder einen Verlust, angesichts unserer Entscheidung. Viele echte Probleme fallen in diese Kategorie. Zum Beispiel müssen wir den Aktienkurs von morgen vorhersagen, dies ermöglicht es uns, basierend auf dieser Schätzung zu handeln und am Ende des Tages finden wir heraus, ob unsere Schätzung uns erlaubt hat, einen Gewinn zu erzielen. Mit anderen Worten, in *Online-Lernen*, haben wir den folgenden Zyklus, in dem wir unser Modell ständig verbessern, wenn neue Beobachtungen gegeben.

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{data} ~ \mathbf{x}_t \longrightarrow
\mathrm{estimate} ~ f_t(\mathbf{x}_t) \longrightarrow
\mathrm{observation} ~ y_t \longrightarrow
\mathrm{loss} ~ l(y_t, f_t(\mathbf{x}_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

### Banditen

*Banditen* sind ein Sonderfall des obigen Problems. Während wir in den meisten Lernproblemen eine kontinuierlich parametrisierte Funktion $f$ haben, wo wir ihre Parameter lernen wollen (z.B. ein tiefes Netzwerk), haben wir bei einem *Bandit*-Problem nur eine endliche Anzahl von Armen, die wir ziehen können, d.h. eine endliche Anzahl von Aktionen, die wir ergreifen können. Es ist nicht sehr verwunderlich, dass für dieses einfachere Problem stärkere theoretische Garantien in Bezug auf Optimalität erzielt werden können. Wir listen es hauptsächlich auf, da dieses Problem oft (verwirrend) so behandelt wird, als wäre es eine eindeutige Lerneinstellung.

### Kontrolle

In vielen Fällen erinnert sich die Umwelt an das, was wir getan haben. Nicht notwendigerweise in einer gegnerischen Weise, aber es wird sich nur erinnern und die Antwort wird davon abhängen, was vorher passiert ist. So beobachtet beispielsweise ein Kaffeekesselregler unterschiedliche Temperaturen, je nachdem, ob er den Heizkessel zuvor erhitzt hat. PID-Controller-Algorithmen (proportional-integral-derivative) sind dort eine beliebte Wahl. Ebenso hängt das Verhalten eines Nutzers auf einer Nachrichtenseite davon ab, was wir ihr zuvor gezeigt haben (z. B. wird sie die meisten Nachrichten nur einmal lesen). Viele solcher Algorithmen bilden ein Modell der Umgebung, in der sie so handeln, dass ihre Entscheidungen weniger zufällig erscheinen. In jüngster Zeit wurde die Steuerungstheorie (z.B. PID-Varianten) auch verwendet, um Hyperparameter automatisch zu optimieren, um eine bessere Entwirrungs- und Rekonstruktionsqualität zu erzielen und die Vielfalt des generierten Textes und die Rekonstruktionsqualität generierter Bilder zu verbessern :cite:`Shao.Yao.Sun.ea.2020`.

### Verstärkung Lernen

Im allgemeineren Fall einer Umgebung mit Speicher können wir Situationen treffen, in denen die Umgebung versucht, mit uns zu kooperieren (kooperative Spiele, insbesondere für Nicht-Nullsummen-Spiele), oder andere, in denen die Umgebung versuchen wird, zu gewinnen. Schach, Go, Backgammon oder StarCraft sind einige der Fälle in *Verstärkungslernen*. Ebenso möchten wir vielleicht einen guten Controller für autonome Autos bauen. Die anderen Autos reagieren wahrscheinlich auf den Fahrstil des autonomen Autos auf nicht-triviale Weise, z. B. versuchen, es zu vermeiden, versuchen, einen Unfall zu verursachen, und versuchen, mit ihm zusammenzuarbeiten.

### Berücksichtigung der Umwelt

Eine wesentliche Unterscheidung zwischen den verschiedenen oben genannten Situationen besteht darin, dass die gleiche Strategie, die im Falle einer stationären Umgebung durchgängig funktioniert haben könnte, möglicherweise nicht während der gesamten Zeit funktioniert, wenn sich die Umgebung anpassen kann. Zum Beispiel wird eine Arbitrage-Gelegenheit, die von einem Händler entdeckt wird, wahrscheinlich verschwinden, sobald er beginnt, sie zu nutzen. Die Geschwindigkeit und Art und Weise, mit der sich die Umgebung verändert, bestimmen zu einem großen Teil die Art von Algorithmen, die wir zu ertragen bringen können. Wenn wir zum Beispiel wissen, dass sich die Dinge nur langsam ändern können, können wir jede Schätzung dazu zwingen, sich nur langsam zu ändern. Wenn wir wissen, dass sich die Umwelt augenblicklich ändern könnte, aber nur sehr selten, können wir dafür sorgen. Diese Arten von Wissen sind entscheidend für die aufstrebende Datenwissenschaftlerin, um mit der Konzeptverschiebung umzugehen, dh, wenn das Problem, dass sie versucht, Veränderungen im Laufe der Zeit zu lösen.

## Fairness, Rechenschaftspflicht und Transparenz im maschinellen Lernen

Schließlich ist es wichtig, sich daran zu erinnern, dass Sie bei der Bereitstellung von Machine Learning-Systemen nicht nur ein Vorhersagemodell optimieren, sondern in der Regel ein Tool bereitstellen, mit dem Entscheidungen (teilweise oder vollständig) automatisiert werden können. Diese technischen Systeme können sich auf das Leben von Einzelpersonen auswirken, die den daraus resultierenden Entscheidungen unterliegen. Der Sprung von der Betrachtung von Vorhersagen zu Entscheidungen wirft nicht nur neue technische Fragen auf, sondern auch eine ganze Reihe ethischer Fragen, die sorgfältig geprüft werden müssen. Wenn wir ein medizinisches Diagnosesystem einsetzen, müssen wir wissen, für welche Populationen es funktionieren kann und welche nicht. Der Blick auf vorhersehbare Risiken für das Wohlergehen einer Subpopulation könnte dazu führen, dass wir minderwertige Pflege verabreichen. Darüber hinaus müssen wir, wenn wir Entscheidungssysteme betrachten, zurücktreten und überdenken, wie wir unsere Technologie bewerten. Unter anderem werden wir feststellen, dass *Genauigkeit* selten die richtige Maßnahme ist. Wenn wir beispielsweise Vorhersagen in Aktionen umsetzen, wollen wir oft die potenzielle Kostensensitivität von irrenden auf verschiedene Arten berücksichtigen. Wenn eine Art, ein Bild falsch zu klassifizieren, als rassische Handschlitten wahrgenommen werden könnte, während eine Fehlklassifizierung in eine andere Kategorie harmlos wäre, dann möchten wir vielleicht unsere Schwellen entsprechend anpassen, wobei gesellschaftliche Werte bei der Gestaltung des Entscheidungsprotokolls berücksichtigt werden. Wir möchten auch darauf achten, wie Prognosesysteme zu Feedback-Schleifen führen können. Betrachten Sie beispielsweise vorausschauende Polizeisysteme, die Patrouillebeamte Gebieten mit hoch prognostizierten Kriminalität zuordnen. Es ist leicht zu sehen, wie ein besorgniserregendes Muster entstehen kann:

 1. Nachbarschaften mit mehr Kriminalität bekommen mehr Patrouillen.
 1. Folglich werden in diesen Nachbarschaften mehr Verbrechen entdeckt, indem die Trainingsdaten eingegeben werden, die für zukünftige Iterationen verfügbar sind.
 1. Ausgesetzt positiveren, prognostiziert das Modell noch mehr Kriminalität in diesen Nachbarschaften.
 1. In der nächsten Iteration zielt das aktualisierte Modell noch stärker auf die gleiche Nachbarschaft ab, was zu noch mehr Verbrechen entdeckt usw. führt.

Häufig werden die verschiedenen Mechanismen, mit denen die Vorhersagen eines Modells mit seinen Trainingsdaten gekoppelt werden, im Modellierungsprozess nicht berücksichtigt. Dies kann dazu führen, was Forscher *Runaway Feedback-Loops* nennen. Darüber hinaus wollen wir vorsichtig sein, ob wir das richtige Problem überhaupt ansprechen. Prädiktive Algorithmen spielen jetzt eine übergroße Rolle bei der Vermittlung der Verbreitung von Informationen. Sollte die Nachricht, dass eine individuelle Begegnung durch die Menge von Facebook-Seiten bestimmt werden, die sie haben*Gefällt mir? Dies sind nur einige der vielen drängenden ethischen Dilemmata, denen Sie in einer Karriere im maschinellen Lernen begegnen könnten.

## Zusammenfassung

* In vielen Fällen kommen Trainings- und Testsets nicht aus der gleichen Verteilung. Dies wird als Verteilungsverschiebung bezeichnet.
* Wahres Risiko ist die Erwartung des Verlustes von Daten, die aus ihrer tatsächlichen Verteilung gezogen werden, über die gesamte Population. Diese gesamte Bevölkerung ist jedoch in der Regel nicht verfügbar. Empirisches Risiko ist ein durchschnittlicher Verlust gegenüber den Trainingsdaten zur Annäherung des wahren Risikos. In der Praxis führen wir empirische Risikominimierung durch.
* Unter den entsprechenden Annahmen können Kovariaten- und Labelverschiebungen zur Testzeit erkannt und korrigiert werden. Die Nichtbeachtung dieser Voreingenommenheit kann zur Testzeit problematisch werden.
* In einigen Fällen kann sich die Umgebung an automatisierte Aktionen erinnern und auf überraschende Weise reagieren. Wir müssen diese Möglichkeit beim Bau von Modellen berücksichtigen und weiterhin lebende Systeme überwachen, offen für die Möglichkeit, dass sich unsere Modelle und die Umwelt in unerwarteter Weise verstricken werden.

## Übungen

1. Was könnte passieren, wenn wir das Verhalten einer Suchmaschine ändern? Was können die Benutzer tun? Was ist mit den Werbetreibenden?
1. Implementieren Sie einen kovariaten Schaltdetektor. Hinweis: Erstellen Sie einen Klassifikator.
1. Implementieren Sie einen Kovariatenschichtkorrektor.
1. Was könnte außer der Verteilungsverschiebung noch beeinflussen, wie empirisches Risiko das wahre Risiko annähert?

[Discussions](https://discuss.d2l.ai/t/105)
