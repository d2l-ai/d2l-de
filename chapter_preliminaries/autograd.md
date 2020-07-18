# Automatische Differenzierung
:label:`sec_autograd`

Wie wir in :numref:`sec_calculus` erklärt haben, ist die Differenzierung ein entscheidender Schritt in fast allen Algorithmen zur Deep Learning-Optimierung. Während die Berechnungen für die Einnahme dieser Derivate einfach sind und nur einige grundlegende Kalkül erfordern, kann bei komplexen Modellen das Ausarbeiten der Updates von Hand schmerzhaft sein (und oft fehleranfällig).

Deep Learning-Frameworks beschleunigen diese Arbeit durch automatische Berechnung von Derivaten, d.h. *automatische Differenzierung*. In der Praxis erstellt das System basierend auf unserem entworfenen Modell ein *Rechendiagramm*, in dem nachverfolgt wird, welche Daten durch welche Operationen zur Erzeugung der Ausgabe kombiniert werden. Durch die automatische Differenzierung kann das System nachträglich Farbverläufe zurückpropagieren. Hierbei bedeutet *backpropagate* einfach, durch das Rechendiagramm nachzuverfolgen und die Teilableitungen in Bezug auf jeden Parameter auszufüllen.

```{.python .input}
from mxnet import autograd, np, npx
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

## Ein einfaches Beispiel

Sagen Sie als Spielzeugbeispiel, dass wir daran interessiert sind, die Funktion $y = 2\mathbf{x}^{\top}\mathbf{x}$ in Bezug auf den Spaltenvektor $\mathbf{x}$ zu unterscheiden. Um zu beginnen, lassen Sie uns die Variable `x` erstellen und ihr einen Anfangswert zuweisen.

```{.python .input}
x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

Bevor wir sogar den Gradienten von $y$ in Bezug auf $\mathbf{x}$ berechnen, benötigen wir einen Platz, um ihn zu speichern. Es ist wichtig, dass wir nicht jedes Mal neuen Speicher zuweisen, wenn wir eine Ableitung in Bezug auf einen Parameter nehmen, weil wir oft die gleichen Parameter Tausende oder Millionen von Malen aktualisieren und schnell der Speicher auslaufen könnte. Beachten Sie, dass ein Farbverlauf einer skalarwertigen Funktion in Bezug auf einen Vektor $\mathbf{x}$ selbst vektorwert ist und die gleiche Form wie $\mathbf{x}$ hat.

```{.python .input}
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
x.grad  # The default value is None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

Lassen Sie uns nun $y$ berechnen.

```{.python .input}
# Place our code inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

Da `x` ein Vektor der Länge 4 ist, wird ein inneres Produkt von `x` und `x` durchgeführt, was die skalare Ausgabe ergibt, die wir `y` zuweisen. Als nächstes können wir automatisch den Gradient von `y` in Bezug auf jede Komponente von `x` berechnen, indem wir die Funktion für Backpropagation aufrufen und den Farbverlauf drucken.

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

Der Gradient der Funktion $y = 2\mathbf{x}^{\top}\mathbf{x}$ in Bezug auf $\mathbf{x}$ sollte $4\mathbf{x}$ betragen. Lassen Sie uns schnell überprüfen, ob unser gewünschter Gradient korrekt berechnet wurde.

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

Lassen Sie uns nun eine andere Funktion von `x` berechnen.

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input}
#@tab pytorch
# PyTorch accumulates the gradient in default, we need to clear the previous 
# values
x.grad.zero_() 
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## Rückwärts für nicht skalare Variablen

Technisch gesehen ist, wenn `y` kein Skalar ist, die natürlichste Interpretation der Differenzierung eines Vektors `y` in Bezug auf einen Vektor `x` eine Matrix. Für höhere und höherdimensionale `y` und `x` könnte das Differenzierungsergebnis ein hochrangiger Tensor sein.

Während sich diese exotischeren Objekte jedoch im fortgeschrittenen maschinellen Lernen (auch im Deep Learning) zeigen, versuchen wir häufiger, wenn wir einen Vektor rückwärts aufrufen, die Ableitungen der Verlustfunktionen für jeden Bestandteil eines *Batch* von Trainingsbeispielen zu berechnen. Dabei geht es nicht darum, die Differenzierungsmatrix zu berechnen, sondern vielmehr die Summe der Teilderivate, die für jedes Beispiel im Batch individuell berechnet werden.

```{.python .input}
# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()
x.grad  # Equals to y = sum(x * x)
```

```{.python .input}
#@tab pytorch
# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalent to the below
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## Berechnung lösen

Manchmal möchten wir einige Berechnungen außerhalb des aufgezeichneten Rechendiagramms verschieben. Angenommen, `y` wurde als Funktion `x` berechnet und anschließend `z` als Funktion von `y` und `x` berechnet. Stellen Sie sich vor, dass wir den Gradienten von `z` in Bezug auf `x` berechnen wollten, aber aus irgendeinem Grund `y` als Konstante behandeln wollten und nur die Rolle berücksichtigen, die `x` spielte, nachdem `y` berechnet wurde.

Hier können wir `y` lösen, um eine neue Variable `u` zurückzugeben, die den gleichen Wert wie `y` hat, aber alle Informationen darüber verwirft, wie `y` im Berechnungsdiagramm berechnet wurde. Mit anderen Worten, der Farbverlauf fließt nicht rückwärts durch `u` bis `x`. Somit berechnet die folgende Rückpropagationsfunktion die partielle Ableitung von `z = u * x` in Bezug auf `x`, während `u` als Konstante behandelt wird, anstatt die partielle Ableitung von `z = x * x * x` in Bezug auf `x`.

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# Set `persistent=True` to run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

Da die Berechnung von `y` aufgezeichnet wurde, können wir anschließend Backpropagation auf `y` aufrufen, um die Ableitung von `y = x * x` in Bezug auf `x` zu erhalten, was `2 * x` ist.

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## Berechnen des Gradienten des Python-Kontrollflusses

Ein Vorteil der automatischen Differenzierung besteht darin, dass auch wenn der Berechnungsgraph einer Funktion erstellt wird, die durch ein Labyrinth des Python-Kontrollflusses (z. B. Bedingungen, Schleifen und beliebige Funktionsaufrufe) benötigt wird, wir immer noch den Gradient der resultierenden Variablen berechnen können. Beachten Sie im folgenden Snippet, dass die Anzahl der Iterationen der `while`-Schleife und die Auswertung der `if`-Anweisung jeweils vom Wert der Eingabe `a` abhängen.

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Lassen Sie uns den Gradient berechnen.

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

Wir können nun die oben definierte `f` Funktion analysieren. Beachten Sie, dass es in seinem Eingang `a` stückweise linear ist. Mit anderen Worten, für jede `a` gibt es eine konstante Skalar `k`, so dass `f(a) = k * a`, wo der Wert von `k` hängt von der Eingabe `a`. Daher können wir mit `d / a` überprüfen, ob der Farbverlauf korrekt ist.

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

## Zusammenfassung

* Deep Learning-Frameworks können die Berechnung von Derivaten automatisieren. Um es zu verwenden, fügen wir zuerst Gradienten an jene Variablen, in denen wir teilweise Derivate wünschen. Wir zeichnen dann die Berechnung unseres Zielwerts auf, führen seine Funktion für die Backpropagation aus und greifen auf den resultierenden Farbverlauf zu.

## Übungen

1. Warum ist das zweite Derivat viel teurer zu berechnen als das erste Derivat?
1. Nachdem Sie die Funktion für die Backpropagation ausgeführt haben, führen Sie sie sofort erneut aus und sehen Sie, was passiert.
1. In dem Steuerflussbeispiel, in dem wir die Ableitung von `d` in Bezug auf `a` berechnen, was passieren würde, wenn wir die Variable `a` in einen zufälligen Vektor oder eine Matrix ändern. An diesem Punkt ist das Ergebnis der Berechnung `f(a)` kein Skalar mehr. Was passiert mit dem Ergebnis? Wie analysieren wir das?
1. Gestalten Sie ein Beispiel für das Auffinden des Verlaufs des Steuerflusses neu. Führen Sie das Ergebnis aus und analysieren Sie es.
1. Lassen Sie $f(x) = \sin(x)$. Plot $f(x)$ und $\frac{df(x)}{dx}$, wobei letzteres berechnet wird, ohne diese $f'(x) = \cos(x)$ zu nutzen.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:
