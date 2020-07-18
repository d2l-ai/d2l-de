# Gewicht Zerfall
:label:`sec_weight_decay`

Jetzt, da wir das Problem der Überrüstung charakterisiert haben, können wir einige Standardtechniken für die Regularisierung von Modellen einführen. Erinnern Sie sich daran, dass wir Überanpassungen immer mindern können, indem wir ausgehen und mehr Trainingsdaten sammeln. Das kann kostspielig, zeitaufwändig sein oder ganz außerhalb unserer Kontrolle sein, was es kurzfristig unmöglich macht. Wir können vorerst davon ausgehen, dass wir bereits so viele qualitativ hochwertige Daten haben, wie unsere Ressourcen es erlauben und uns auf Regularisierungstechniken konzentrieren.

Erinnern Sie sich daran, dass wir in unserem Polynom-Regressionsbeispiel (:numref:`sec_model_selection`) die Kapazität unseres Modells einschränken konnten, indem wir einfach den Grad des angepassten Polynoms optimieren. In der Tat ist die Begrenzung der Anzahl der Features eine beliebte Technik, um Überanpassungen zu mildern. Das einfache Werfen von Funktionen kann jedoch zu stumpfen als Instrument für den Job sein. Überlegen Sie, was mit hochdimensionalen Eingaben passieren könnte, wenn Sie sich an das Beispiel für Polynom-Regression halten. Die natürlichen Erweiterungen von Polynomen zu multivariaten Daten heißen *Monomien*, die einfach Produkte von Potenzen von Variablen sind. Der Grad eines Monomials ist die Summe der Kräfte. Zum Beispiel sind $x_1^2 x_2$ und $x_3 x_5^2$ beide Monome von Grad 3.

Beachten Sie, dass die Anzahl der Begriffe mit dem Grad $d$ schnell explodiert, da $d$ größer wird. Bei $k$ Variablen beträgt die Anzahl der Monomien des Grades $d$ (d. h. $k$ Multichoose $d$) ${k - 1 + d} \choose {k - 1}$. Selbst kleine Veränderungen im Grad, sagen wir von $2$ bis $3$, erhöhen die Komplexität unseres Modells dramatisch. Daher benötigen wir oft ein feinkörniges Werkzeug zur Anpassung der Funktionskomplexität.

## Normen und Gewichtsverfall

Wir haben sowohl die Norm $L_2$ als auch die Norm $L_1$ beschrieben, die Sonderfälle der allgemeineren Norm $L_p$ in :numref:`subsec_lin-algebra-norms` sind.
*Gewichtszerfall* (üblicherweise $L_2$ Regularisierung genannt),
könnte die am weitesten verbreitete Technik zur Regularisierung parametrischer Machine Learning-Modelle sein. Die Technik wird durch die grundlegende Intuition motiviert, dass unter allen Funktionen $f$ die Funktion $f = 0$ (Zuweisung des Wertes $0$ zu allen Eingängen) in gewissem Sinne die *einfachste* ist, und dass wir die Komplexität einer Funktion durch ihren Abstand von Null messen können. Aber wie genau sollten wir den Abstand zwischen einer Funktion und Null messen? Es gibt keine einzige richtige Antwort. Tatsächlich widmen sich ganze Zweige der Mathematik, einschließlich Teile der funktionalen Analyse und der Theorie der Banach-Räume, der Beantwortung dieses Problems.

Eine einfache Interpretation könnte darin bestehen, die Komplexität einer linearen Funktion $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$ anhand einer Norm ihres Gewichtvektors zu messen, z. B. $\| \mathbf{w} \|^2$. Die gebräuchlichste Methode zur Sicherstellung eines kleinen Gewichtsvektors besteht darin, dem Problem der Minimierung des Verlustes seine Norm als Straffrist hinzuzufügen. So ersetzen wir unser ursprüngliches Ziel,
*Minimierung des Vorhersageverlusts auf den Trainingsetiketten*,
mit neuem Ziel,
*Minimierung der Summe des Vorhersageverlusts und der Strafe Term*.
Wenn unser Gewichtsvektor zu groß wird, könnte sich unser Lernalgorithmus darauf konzentrieren, die Gewichtsnorm $\| \mathbf{w} \|^2$ zu minimieren und den Trainingsfehler zu minimieren. Genau das wollen wir. Um Dinge im Code zu veranschaulichen, lassen Sie uns unser vorheriges Beispiel von :numref:`sec_linear_regression` für die lineare Regression wiederbeleben. Dort wurde unser Verlust von

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Daran erinnern, dass $\mathbf{x}^{(i)}$ die Features sind, $y^{(i)}$ sind Beschriftungen für alle Datenpunkte $i$ und $(\mathbf{w}, b)$ sind die Gewichts- und Verzerrungsparameter. Um die Größe des Gewichtsvektors zu bestrafen, müssen wir irgendwie $\| \mathbf{w} \|^2$ zur Verlustfunktion hinzufügen, aber wie sollte das Modell den Standardverlust für diese neue Zusatzstrafe abgeben? In der Praxis charakterisieren wir diesen Kompromiss über die *Regularisierungskonstante* $\lambda$, einen nicht-negativen Hyperparameter, den wir mit Validierungsdaten anpassen:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2,$$

Für $\lambda = 0$ stellen wir unsere ursprüngliche Verlustfunktion wieder her. Für $\lambda > 0$ beschränken wir die Größe von $\| \mathbf{w} \|$. Wir teilen durch $2$ durch Konvention: Wenn wir die Ableitung einer quadratischen Funktion nehmen, brechen die $2$ und $1/2$ aus, um sicherzustellen, dass der Ausdruck für das Update schön und einfach aussieht. Der kluge Leser fragt sich vielleicht, warum wir mit der quadrierten Norm arbeiten und nicht mit der Standardnorm (d.h. der euklidischen Entfernung). Wir tun dies für rechnerische Bequemlichkeit. Indem wir die Norm $L_2$ quadrieren, entfernen wir die Quadratwurzel, wobei die Summe der Quadrate jeder Komponente des Gewichtsvektors belassen wird. Dies macht die Ableitung der Strafe leicht zu berechnen: Die Summe der Derivate entspricht der Ableitung der Summe.

Darüber hinaus können Sie fragen, warum wir überhaupt mit der Norm $L_2$ arbeiten und nicht, sagen wir, die Norm $L_1$. In der Tat sind andere Auswahlmöglichkeiten gültig und beliebt in den Statistiken. Während $L_2$-regularisierte lineare Modelle den klassischen *ridge Regression* Algorithmus darstellen, ist $L_1$-regularisierte lineare Regression ein ähnlich grundlegendes Modell in der Statistik, das im Volksmund als *Lasso-Regression* bekannt ist.

Ein Grund, mit der Norm $L_2$ zu arbeiten, ist, dass sie eine übergroße Strafe auf große Komponenten des Gewichtsvektors setzt. Dies verzerrte unseren Lernalgorithmus auf Modelle, die das Gewicht gleichmäßig über eine größere Anzahl von Funktionen verteilen. In der Praxis könnte dies sie robuster für Messfehler in einer einzelnen Variablen machen. Dagegen führen $L_1$ Strafen zu Modellen, die Gewichte auf einen kleinen Satz von Features konzentrieren, indem die anderen Gewichte auf Null gelöscht werden. Dies wird *Feature-Auswahl* genannt, was aus anderen Gründen wünschenswert sein kann.

Unter Verwendung der gleichen Notation in :eqref:`eq_linreg_batch_update` folgt der Minibatch-stochastische Gradientenabstieg für $L_2$-regularisierte Regression:

$$
\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

Wie zuvor aktualisieren wir $\mathbf{w}$ basierend auf dem Betrag, um den sich unsere Schätzung von der Beobachtung unterscheidet. Allerdings schrumpfen wir auch die Größe von $\mathbf{w}$ auf Null. Deshalb wird die Methode manchmal als „Gewichtszerfall“ bezeichnet: Angesichts der Strafe allein, unser Optimierungsalgorithmus *verfällt das Gewicht bei jedem Trainingsschritt. Im Gegensatz zur Feature-Auswahl bietet uns der Gewichtsabfall einen kontinuierlichen Mechanismus zur Anpassung der Komplexität einer Funktion. Kleinere Werte von $\lambda$ entsprechen weniger eingeschränkten $\mathbf{w}$, während größere Werte von $\lambda$ $\mathbf{w}$ erheblich einschränken.

Ob wir eine entsprechende Verzerrungsstrafe $b^2$ einschließen, kann zwischen den Implementierungen variieren und über Schichten eines neuronalen Netzwerks variieren. Oft regularisieren wir den Verzerrungsbegriff der Ausgabeschicht eines Netzwerks nicht.

## Hochdimensionale lineare Regression

Wir können die Vorteile des Gewichtsverfalls anhand eines einfachen synthetischen Beispiels veranschaulichen.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torch.nn as nn
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Zuerst erzeugen wir einige Daten wie zuvor

$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$

Wir wählen unser Etikett als lineare Funktion unserer Eingänge, beschädigt durch Gaußschen Rauschen mit Null Mittelwert und Standardabweichung 0,01. Um die Auswirkungen der Überanpassung deutlich zu machen, können wir die Dimensionalität unseres Problems auf $d = 200$ erhöhen und mit einem kleinen Trainingsset arbeiten, das nur 20 Beispiele enthält.

```{.python .input}
#@tab all
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

## Implementierung von Grund auf neu

Im Folgenden werden wir Gewichtsabfall von Grund auf neu implementieren, indem wir einfach die quadrierte $L_2$ Strafe zur ursprünglichen Zielfunktion hinzufügen.

### Initialisieren von Modellparametern

Zuerst definieren wir eine Funktion, um unsere Modellparameter zufällig zu initialisieren.

```{.python .input}
def init_params():
    w = np.random.normal(scale=1, size=(num_inputs, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

```{.python .input}
#@tab pytorch
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

```{.python .input}
#@tab tensorflow
def init_params():
    w = tf.Variable(tf.random.normal(mean=1, shape=(num_inputs, 1)))
    b = tf.Variable(tf.zeros(shape=(1, )))
    return [w, b]
```

### Definieren der $L_2$ Norm Strafe

Der vielleicht bequemste Weg, um diese Strafe umzusetzen, besteht darin, alle Begriffe zu quadratischen und zusammenzufassen.

```{.python .input}
def l2_penalty(w):
    return (w**2).sum() / 2
```

```{.python .input}
#@tab pytorch
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

```{.python .input}
#@tab tensorflow
def l2_penalty(w):
    return tf.reduce_sum(tf.pow(w, 2)) / 2
```

### Definieren der Trainingsschleife

Der folgende Code passt zu einem Modell auf das Trainingsset und wertet es auf dem Testsatz aus. Das lineare Netzwerk und der quadratische Verlust haben sich seit :numref:`chap_linear` nicht geändert, daher werden wir sie einfach über `d2l.linreg` und `d2l.squared_loss` importieren. Die einzige Änderung hier ist, dass unser Verlust jetzt die Straffrist beinhaltet.

```{.python .input}
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(w))
```

```{.python .input}
#@tab pytorch
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', torch.norm(w).item())
```

```{.python .input}
#@tab tensorflow
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # The L2 norm penalty term has been added, and broadcasting
                # makes `l2_penalty(w)` a vector whose length is `batch_size`
                l = loss(net(X), y) + lambd * l2_penalty(w)
            grads = tape.gradient(l, [w, b])
            d2l.sgd([w, b], grads, lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(w).numpy())
```

### Schulung ohne Regularisierung

Wir führen diesen Code jetzt mit `lambd = 0` aus, Deaktivieren des Gewichtsverfalls. Beachten Sie, dass wir schlecht überpassen und den Trainingsfehler verringern, aber nicht den Testfehler — ein Fall von Überanpassung.

```{.python .input}
#@tab all
train(lambd=0)
```

### Verwenden von Gewichtsabfall

Unten laufen wir mit erheblichem Gewichtsverlust. Beachten Sie, dass der Trainingsfehler zunimmt, aber der Testfehler abnimmt. Genau das ist der Effekt, den wir von der Regularisierung erwarten.

```{.python .input}
#@tab all
train(lambd=3)
```

## Prägnante Implementierung

Da Gewichtsabfall bei der Optimierung neuronaler Netze allgegenwärtig ist, macht das Deep Learning-Framework es besonders bequem und integriert Gewichtsabfall in den Optimierungsalgorithmus selbst für eine einfache Verwendung in Kombination mit jeder Verlustfunktion. Darüber hinaus bietet diese Integration einen Rechenvorteil und ermöglicht Implementierungstricks, um dem Algorithmus Gewichtsabfall hinzuzufügen, ohne zusätzlichen Rechenaufwand. Da der Gewichtsabfall des Updates nur vom aktuellen Wert jedes Parameters abhängt, muss der Optimierer jeden Parameter sowieso einmal berühren.

:begin_tab:`mxnet`
Im folgenden Code geben wir den Gewicht-Decay-Hyperparameter direkt über `wd` an, wenn wir unsere `Trainer` instanziieren. Standardmäßig zerfällt Gluon sowohl Gewichte als auch Verzerrungen gleichzeitig. Beachten Sie, dass der Hyperparameter `wd` bei der Aktualisierung von Modellparametern mit `wd_mult` multipliziert wird. Wenn wir also `wd_mult` auf Null setzen, wird der Bias-Parameter $b$ nicht zerfallen.
:end_tab:

:begin_tab:`pytorch`
Im folgenden Code geben wir den Gewicht-Decay-Hyperparameter direkt über `weight_decay` an, wenn wir unseren Optimierer instanziieren. Standardmäßig verfällt PyTorch gleichzeitig Gewichte und Verzerrungen. Hier setzen wir nur `weight_decay` für das Gewicht, so dass der Biasparameter $b$ nicht zerfallen wird.
:end_tab:

:begin_tab:`tensorflow`
Im folgenden Code erstellen wir einen $L_2$ Regularizer mit dem Gewichtsverfall Hyperparameter `wd` und wenden ihn über das Argument `kernel_regularizer` auf die Ebene an.
:end_tab:

```{.python .input}
def train_concise(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    loss = gluon.loss.L2Loss()
    num_epochs, lr = 100, 0.003
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias"
    net.collect_params('.*bias').setattr('wd_mult', 0)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', np.linalg.norm(net[0].weight.data()))
```

```{.python .input}
#@tab pytorch
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # The bias parameter has not decayed
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                trainer.zero_grad()
                l = loss(net(X), y)
            l.backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', net[0].weight.norm().item())
```

```{.python .input}
#@tab tensorflow
def train_concise(wd):
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(wd)))
    net.build(input_shape=(1, num_inputs))
    w, b = net.trainable_variables
    loss = tf.keras.losses.MeanSquaredError()
    num_epochs, lr = 100, 0.003
    trainer = tf.keras.optimizers.SGD(learning_rate=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                # `tf.keras` requires retrieving and adding the losses from
                # layers manually for custom training loop.
                l = loss(net(X), y) + net.losses
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('L2 norm of w:', tf.norm(net.get_weights()[0]).numpy())
```

Die Plots sehen identisch mit denen aus, wenn wir Gewichtsverfall von Grund auf neu implementiert haben. Sie laufen jedoch deutlich schneller und sind einfacher zu implementieren, ein Vorteil, der bei größeren Problemen stärker ausgeprägt wird.

```{.python .input}
#@tab all
train_concise(0)
```

```{.python .input}
#@tab all
train_concise(3)
```

Bisher haben wir nur eine Vorstellung davon berührt, was eine einfache lineare Funktion ausmacht. Darüber hinaus kann das, was eine einfache nichtlineare Funktion ausmacht, eine noch komplexere Frage sein. [Reproduktion des Kernel Hilbert space (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) ermöglicht es beispielsweise, Werkzeuge anzuwenden, die für lineare Funktionen in einem nichtlinearen Kontext eingeführt wurden. Leider neigen RKHS-basierte Algorithmen dazu, rein auf große, hochdimensionale Daten zu skalieren. In diesem Buch werden wir auf die einfache Heuristik des Anwendens von Gewichtszerfall auf allen Schichten eines tiefen Netzwerks voreingestellt.

## Zusammenfassung

* Regularisierung ist eine gebräuchliche Methode für den Umgang mit Überanpassung. Es fügt eine Straffrist zur Verlustfunktion auf dem Trainingsset hinzu, um die Komplexität des erlernten Modells zu reduzieren.
* Eine besondere Wahl, um das Modell einfach zu halten, ist Gewichtsverfall mit einer $L_2$ Strafe. Dies führt zu einem Gewichtsverlust in den Aktualisierungsschritten des Lernalgorithmus.
* Die Gewicht-Zerfalls-Funktionalität wird in Optimierern von Deep Learning-Frameworks zur Verfügung gestellt.
* Verschiedene Parametersätze können unterschiedliche Aktualisierungsverhalten innerhalb derselben Trainingsschleife aufweisen.

## Übungen

1. Experimentieren Sie mit dem Wert $\lambda$ im Schätzungsproblem in diesem Abschnitt. Zeichnen Sie Training und Testgenauigkeit als Funktion von $\lambda$. Was beobachtest du?
1. Verwenden Sie einen Validierungssatz, um den optimalen Wert von $\lambda$ zu ermitteln. Ist es wirklich der optimale Wert? Ist das wichtig?
1. Wie würden die Update-Gleichungen aussehen, wenn wir anstelle von $\|\mathbf{w}\|^2$ $\sum_i |w_i|$ als unsere Strafe der Wahl ($L_1$ Regularisierung) verwenden?
1. Wir wissen, dass $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Können Sie eine ähnliche Gleichung für Matrizen finden (siehe Frobenius-Norm in :numref:`subsec_lin-algebra-norms`)?
1. Überprüfen Sie die Beziehung zwischen Schulungsfehler und Generalisierungsfehler. Neben Gewichtsverlust, erhöhtem Training und der Verwendung eines Modells von geeigneter Komplexität, welche anderen Möglichkeiten können Sie sich vorstellen, mit Überanpassung umzugehen?
1. In der Bayesischen Statistik verwenden wir das Produkt von vorheriger und wahrscheinlicher, um zu einem posterior über $P(w \mid x) \propto P(x \mid w) P(w)$ zu gelangen. Wie können Sie $P(w)$ mit Regularisierung identifizieren?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:
