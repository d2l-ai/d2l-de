#  Vorbereitungen
:label:`chap_preliminaries`

Um mit Deep Learning zu beginnen, müssen wir einige grundlegende Fähigkeiten entwickeln. Das gesamte maschinelle Lernen beschäftigt sich mit dem Extrahieren von Informationen aus Daten. So werden wir damit beginnen, die praktischen Fähigkeiten zum Speichern, Manipulieren und Vorverarbeiten von Daten zu erlernen.

Darüber hinaus erfordert maschinelles Lernen in der Regel das Arbeiten mit großen Datensätzen, die wir als Tabellen betrachten können, in denen die Zeilen Beispielen entsprechen und die Spalten Attributen entsprechen. Lineare Algebra gibt uns eine leistungsstarke Reihe von Techniken für die Arbeit mit tabellarischen Daten. Wir werden nicht zu weit in das Unkraut gehen, sondern uns auf die Grundlagen der Matrixoperationen und deren Umsetzung konzentrieren.

Darüber hinaus dreht sich bei Deep Learning um Optimierung. Wir haben ein Modell mit einigen Parametern und wir wollen diejenigen finden, die zu unseren Daten passen *die Beste*. Die Bestimmung, welche Art und Weise jeder Parameter bei jedem Schritt eines Algorithmus bewegt werden soll, erfordert ein wenig Kalkül, das kurz eingeführt wird. Glücklicherweise berechnet das Paket `autograd` automatisch die Differenzierung für uns, und wir werden es als nächstes abdecken.

Als nächstes geht es maschinelles Lernen darum, Vorhersagen zu machen: Was ist der wahrscheinliche Wert eines unbekannten Attributs angesichts der Informationen, die wir beobachten? Um rigoros unter Unsicherheit zu begründen, müssen wir die Sprache der Wahrscheinlichkeit aufrufen.

Am Ende enthält die offizielle Dokumentation viele Beschreibungen und Beispiele, die über dieses Buch hinausgehen. Zum Abschluss des Kapitels zeigen wir Ihnen, wie Sie die Dokumentation für die benötigten Informationen nachschlagen können.

Dieses Buch hat den mathematischen Inhalt auf das Minimum gehalten, das notwendig ist, um ein richtiges Verständnis von Deep Learning zu erhalten. Es bedeutet jedoch nicht, dass dieses Buch mathematikfrei ist. Daher bietet dieses Kapitel eine schnelle Einführung in die grundlegende und häufig verwendete Mathematik, um jedem zu ermöglichen, zumindest *die meist* des mathematischen Inhalts des Buches zu verstehen. Wenn Sie *alle* des mathematischen Inhalts verstehen möchten, sollte eine weitere Überprüfung des [online appendix on mathematics](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) ausreichend sein.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```
