# Zahnstein
:label:`sec_calculus`

Die Suche nach der Fläche eines Polygons war bis vor mindestens 2.500 Jahren geheimnisvoll geblieben, als alte Griechen ein Polygon in Dreiecke aufteilten und ihre Flächen summierten. Um den Bereich der gekrümmten Formen, wie einen Kreis, zu finden, haben alte Griechen Polygone in solche Formen eingeschrieben. Wie in :numref:`fig_circle_area` gezeigt, nähert sich ein beschriftetes Polygon mit mehr Seiten gleicher Länge dem Kreis besser. Dieser Prozess wird auch als die Methode der Erschöpfung* bezeichnet.

![Find the area of a circle with the method of exhaustion.](../img/polygon_circle.svg)
:label:`fig_circle_area`

Tatsächlich ist die Methode der Erschöpfung, aus der *integrale Berechnung* (wird in :numref:`sec_integral_calculus` beschrieben) stammt. Mehr als 2.000 Jahre später wurde der andere Zweig der Kalküle, *Differentialberechnung*, erfunden. Unter den kritischsten Anwendungen der Differentialrechnung betrachten Optimierungsprobleme, wie man etwas machen*das Beste*. Wie in :numref:`subsec_norms_and_objectives` diskutiert, sind solche Probleme im Deep Learning allgegenwärtig.

Im Deep Learning trainieren* Modelle und aktualisieren sie sukzessive, damit sie besser und besser werden, wenn sie mehr und mehr Daten sehen. Gewöhnlich bedeutet, besser zu werden, eine *Verlustfunktion* zu minimieren, eine Punktzahl, die die Frage beantwortet: „Wie *schlecht* ist unser Modell?“ Diese Frage ist subtiler als es scheint. Letztendlich ist es uns wirklich wichtig, ein Modell zu produzieren, das gut auf Daten funktioniert, die wir noch nie gesehen haben. Aber wir können das Modell nur an Daten anpassen, die wir tatsächlich sehen können. So können wir die Aufgabe der Anpassung von Modellen in zwei zentrale Anliegen zerlegen: i) *Optimierung*: den Prozess der Anpassung unserer Modelle an beobachtete Daten; ii) *Verallgemeinerung*: die mathematischen Prinzipien und die Weisheit der Praktiker, die dazu führen, wie Modelle zu produzieren, deren Gültigkeit über den genauen Satz von Daten hinausgeht Punkte, die verwendet werden, um sie zu trainieren.

Um Ihnen zu helfen, Optimierungsprobleme und Methoden in späteren Kapiteln zu verstehen, geben wir hier eine sehr kurze Grundierung auf Differentialrechnung, die häufig in Deep Learning verwendet wird.

## Derivate und Differenzierung

Wir beginnen mit der Berechnung von Derivaten, ein entscheidender Schritt in fast allen Deep Learning Optimierungsalgorithmen. Im Deep Learning wählen wir in der Regel Verlustfunktionen, die hinsichtlich der Parameter unseres Modells differenzierbar sind. Einfach ausgedrückt bedeutet dies, dass wir für jeden Parameter bestimmen können, wie schnell der Verlust zunehmen oder abnehmen würde, wenn wir diesen Parameter um einen unendlichen kleinen Betrag *erhöhen* oder *reduzieren*.

Angenommen, wir haben eine Funktion $f: \mathbb{R} \rightarrow \mathbb{R}$, deren Eingang und Ausgabe sind beide Skalare. Das *derivative* von $f$ ist definiert als

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h},$$
:eqlabel:`eq_derivative`

, wenn dieses Limit vorhanden ist. Wenn $f'(a)$ existiert, wird $f$ als *differenzierbar* bei $a$ angegeben. Wenn $f$ bei jeder Anzahl eines Intervalls differenzierbar ist, ist diese Funktion in diesem Intervall differenzierbar. Wir können das Derivat $f'(x)$ in :eqref:`eq_derivative` als die *sofortige Änderungsrate von $f(x)$ in Bezug auf $x$ interpretieren. Die sogenannte momentane Veränderungsrate basiert auf der Variation $h$ in $x$, die sich $0$ nähert.

Um Derivate zu veranschaulichen, lassen Sie uns mit einem Beispiel experimentieren. Definieren Sie $u = f(x) = 3x^2-4x$.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

Durch die Festsetzung von $x=1$ und $h$ Ansatz $0$, das numerische Ergebnis von $\frac{f(x+h) - f(x)}{h}$ in :eqref:`eq_derivative` nähert sich $2$. Obwohl dieses Experiment kein mathematischer Beweis ist, werden wir später sehen, dass das Derivat $u'$ $2$ ist, wenn $x=1$.

```{.python .input}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

Machen wir uns mit ein paar äquivalenten Notationen für Derivate vertraut. Da $y = f(x)$, wobei $x$ und $y$ die unabhängige Variable und die abhängige Variable der Funktion $f$ sind. Die folgenden Ausdrücke sind äquivalent:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

wobei die Symbole $\frac{d}{dx}$ und $D$ *Differenzierungsoperatoren* sind, die den Betrieb von *Differenzierung* anzeigen. Wir können die folgenden Regeln verwenden, um gemeinsame Funktionen zu unterscheiden:

* $DC = 0$ ($C$ ist eine Konstante),
* $Dx^n = nx^{n-1}$ (die *Leistungsregel*, $n$ ist eine beliebige reelle Zahl),
* $De^x = e^x$,
* $D\ln(x) = 1/x.$

Um eine Funktion zu unterscheiden, die von einigen einfacheren Funktionen wie den oben genannten gemeinsamen Funktionen gebildet wird, können die folgenden Regeln für uns nützlich sein. Angenommen, die Funktionen $f$ und $g$ sind sowohl differenzierbar und $C$ ist eine Konstante, wir haben die *konstante Mehrfachregel*

$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

die *Summenregel*

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

*Produktregel*

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

und die *Quotient-Regel*

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

Jetzt können wir ein paar der oben genannten Regeln anwenden, um $u' = f'(x) = 3 \frac{d}{dx} x^2-4\frac{d}{dx}x = 6x-4$ zu finden. Wenn wir also $x = 1$ setzen, haben wir $u' = 2$: Dies wird durch unser früheres Experiment in diesem Abschnitt unterstützt, wo sich das numerische Ergebnis $2$ nähert. Diese Ableitung ist auch die Neigung der Tangentiallinie zur Kurve $u = f(x)$, wenn $x = 1$.

Um eine solche Interpretation von Derivaten zu visualisieren, verwenden wir `matplotlib`, eine beliebte Plotbibliothek in Python. Um Eigenschaften der Figuren zu konfigurieren, die von `matplotlib` produziert werden, müssen wir ein paar Funktionen definieren. Im Folgenden gibt die Funktion `use_svg_display` das Paket `matplotlib` an, um die svg figures für schärfere Bilder auszugeben.

```{.python .input}
#@tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')
```

Wir definieren die Funktion `set_figsize`, um die Figurgrößen anzugeben. Beachten Sie, dass wir hier direkt `d2l.plt` verwenden, da die Importanweisung `from matplotlib import pyplot as plt` für das Speichern im Paket `d2l` im Vorwort markiert wurde.

```{.python .input}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

Die folgende Funktion `set_axes` setzt Eigenschaften von Achsen von Zahlen, die von `matplotlib` erzeugt werden.

```{.python .input}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

Mit diesen drei Funktionen für Figurkonfigurationen definieren wir die Funktion `plot`, um mehrere Kurven prägnant zu plotten, da wir viele Kurven im gesamten Buch visualisieren müssen.

```{.python .input}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

Jetzt können wir die Funktion $u = f(x)$ und ihre Tangentenlinie $y = 2x - 3$ bei $x=1$ darstellen, wobei der Koeffizient $2$ die Neigung der Tangentenlinie ist.

```{.python .input}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## Teilweise Derivate

Bisher haben wir uns mit der Differenzierung von Funktionen von nur einer Variablen befasst. Im Deep Learning hängen Funktionen oft von *viele* Variablen ab. Daher müssen wir die Ideen der Differenzierung auf diese *multivariaten Funktionen ausdehnen.

Lassen Sie $y = f(x_1, x_2, \ldots, x_n)$ eine Funktion mit $n$ Variablen sein. Die *partielle Ableitung* von $y$ in Bezug auf seinen $i^\mathrm{th}$ Parameter $x_i$ ist

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

Um $\frac{\partial y}{\partial x_i}$ zu berechnen, können wir $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ einfach als Konstanten behandeln und die Ableitung von $y$ in Bezug auf $x_i$ berechnen. Für die Notation von partiellen Derivaten sind die folgenden gleichwertig:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## Farbverläufe

Wir können partielle Derivate einer multivariaten Funktion in Bezug auf alle ihre Variablen verketten, um den *gradient* Vektor der Funktion zu erhalten. Angenommen, die Eingabe der Funktion $f: \mathbb{R}^n \rightarrow \mathbb{R}$ ist ein $n$-dimensionaler Vektor $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ und die Ausgabe ist ein Skalar. Der Gradient der Funktion $f(\mathbf{x})$ in Bezug auf $\mathbf{x}$ ist ein Vektor von $n$ partiellen Derivaten:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

wobei $\nabla_{\mathbf{x}} f(\mathbf{x})$ oft durch $\nabla f(\mathbf{x})$ ersetzt wird, wenn keine Mehrdeutigkeit vorliegt.

Lassen Sie $\mathbf{x}$ ein $n$-dimensionaler Vektor sein, die folgenden Regeln werden häufig verwendet, wenn multivariate Funktionen unterschieden werden:

* Für alle $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$,
* Für alle $\mathbf{A} \in \mathbb{R}^{n \times m}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$,
* Für alle $\mathbf{A} \in \mathbb{R}^{n \times n}$, $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$,
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

In ähnlicher Weise haben wir für jede Matrix $\mathbf{X}$ $\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$. Wie wir später sehen werden, sind Gradienten nützlich, um Optimierungsalgorithmen im Deep Learning zu entwerfen.

## Chain-Regel

Solche Gradienten können jedoch schwer zu finden sein. Dies liegt daran, dass multivariate Funktionen im Deep Learning oft *composite* sind. Daher können wir keine der oben genannten Regeln anwenden, um diese Funktionen zu unterscheiden. Glücklicherweise ermöglicht die *Chain-Regel* es uns, zusammengesetzte Funktionen zu unterscheiden.

Lassen Sie uns zuerst Funktionen einer einzelnen Variablen betrachten. Angenommen, die Funktionen $y=f(u)$ und $u=g(x)$ sind beide differenzierbar, dann besagt die Kettenregel, dass

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

Nun lassen Sie uns unsere Aufmerksamkeit auf ein allgemeineres Szenario richten, in dem Funktionen eine beliebige Anzahl von Variablen haben. Angenommen, die differenzierbare Funktion $y$ hat Variablen $u_1, u_2, \ldots, u_m$, wobei jede differenzierbare Funktion $u_i$ Variablen $x_1, x_2, \ldots, x_n$ aufweist. Beachten Sie, dass $y$ eine Funktion von $x_1, x_2, \ldots, x_n$ ist. Dann gibt die Kettenregel

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

für alle $i = 1, 2, \ldots, n$.

## Zusammenfassung

* Differentialrechnung und Integralrechnung sind zwei Zweige des Kalkels, in denen erstere auf die allgegenwärtigen Optimierungsprobleme im Deep Learning angewendet werden können.
* Ein Derivat kann als die momentane Veränderungsrate einer Funktion in Bezug auf ihre Variable interpretiert werden. Es ist auch die Neigung der Tangentiallinie zur Kurve der Funktion.
* Ein Gradient ist ein Vektor, dessen Komponenten die partiellen Derivate einer multivariaten Funktion in Bezug auf alle ihre Variablen sind.
* Die Kettenregel ermöglicht es uns, zusammengesetzte Funktionen zu unterscheiden.

## Übungen

1. Zeichnen Sie die Funktion $y = f(x) = x^3 - \frac{1}{x}$ und ihre Tangentenlinie, wenn $x = 1$.
1. Finden Sie den Farbverlauf der Funktion $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. Was ist der Gradient der Funktion $f(\mathbf{x}) = \|\mathbf{x}\|_2$?
1. Können Sie die Kettenregel für den Fall schreiben, in dem $u = f(x, y, z)$ und $x = x(a, b)$, $y = y(a, b)$ und $z = z(a, b)$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/197)
:end_tab:
