# Vorhersagen von Hauspreisen auf Kaggle
:label:`sec_kaggle_house`

Nun, da wir einige grundlegende Werkzeuge für den Aufbau und das Training von tiefen Netzwerken eingeführt und sie mit Techniken wie Gewichtsabfall und Dropout reguliert haben, sind wir bereit, all dieses Wissen in die Praxis umzusetzen, indem wir an einem Kaggle Wettbewerb teilnehmen. Der Hauspreisprognosewettbewerb ist ein großartiger Ort, um zu beginnen. Die Daten sind ziemlich generisch und weisen keine exotische Struktur auf, die spezielle Modelle erfordern könnten (wie Audio oder Video). Dieser Datensatz, der 2011 von Bart de Cock :cite:`De-Cock.2011` gesammelt wurde, deckt die Immobilienpreise in Ames, IA ab dem Zeitraum 2006—2010 ab. Es ist deutlich größer als der berühmte [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) von Harrison und Rubinfeld (1978), mit mehr Beispielen und mehr Features.

In diesem Abschnitt führen wir Sie durch Details zur Datenvorverarbeitung, zum Modellentwurf und zur Auswahl von Hyperparametern. Wir hoffen, dass Sie durch einen praxisorientierten Ansatz einige Intuitionen erhalten, die Sie in Ihrer Karriere als Data Scientist führen werden.

## Datasets herunterladen und zwischenspeichern

Im Laufe des Buches werden wir Modelle auf verschiedenen heruntergeladenen Datensätzen trainieren und testen. Hier implementieren wir mehrere Utility-Funktionen, um das Herunterladen von Daten zu erleichtern. Zuerst pflegen wir ein Wörterbuch `DATA_HUB`, das eine Zeichenfolge (der *Name* des Datensatzes) einem Tupel zuordnet, das sowohl die URL zum Suchen des Datensatzes als auch den SHA-1-Schlüssel enthält, der die Integrität der Datei überprüft. Alle diese Datensätze werden auf der Site gehostet, deren Adresse `DATA_URL` ist.

```{.python .input}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

DATA_HUB = dict()  #@save
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'  #@save
```

Die folgende Funktion `download` lädt ein Dataset herunter, speichert es in einem lokalen Verzeichnis (standardmäßig `../data`) und gibt den Namen der heruntergeladenen Datei zurück. Wenn eine Datei, die diesem Datensatz entspricht, bereits im Cache-Verzeichnis vorhanden ist und sein SHA-1 mit der in `DATA_HUB` gespeicherten Datei übereinstimmt, verwendet unser Code die zwischengespeicherte Datei, um zu vermeiden, dass Ihr Internet mit redundanten Downloads verstopft wird.

```{.python .input}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    d2l.mkdir_if_not_exist(cache_dir)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

Wir implementieren auch zwei zusätzliche Dienstprogrammfunktionen: eine ist das Herunterladen und Extrahieren einer Zip- oder Tar-Datei und die andere, um alle in diesem Buch verwendeten Datensätze aus `DATA_HUB` in das Cache-Verzeichnis herunterzuladen.

```{.python .input}
#@tab all
def download_extract(name, folder=None):  #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)
```

## Kaggle

[Kaggle](https://www.kaggle.com) ist eine beliebte Plattform, die maschinelles Lernen Wettbewerbe veranstaltet. Jeder Wettbewerb konzentriert sich auf einen Datensatz und viele werden von Stakeholdern gesponsert, die Preise für die gewinnenden Lösungen anbieten. Die Plattform hilft Benutzern, über Foren und gemeinsam genutzten Code zu interagieren und fördert sowohl die Zusammenarbeit als auch den Wettbewerb. Während Leaderboard-Jagen oft außer Kontrolle geraten, wobei Forscher sich myopical auf Vorverarbeitungsschritte konzentrieren, anstatt grundlegende Fragen zu stellen, gibt es auch einen enormen Wert in der Objektivität einer Plattform, die direkte quantitative Vergleiche zwischen konkurrierenden Ansätzen sowie Code ermöglicht teilen, so dass jeder lernen kann, was hat und nicht funktioniert. Wenn Sie an einem Kaggle Wettbewerb teilnehmen möchten, müssen Sie sich zunächst für ein Konto registrieren (siehe :numref:`fig_kaggle`).

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

Wie in :numref:`fig_house_pricing` veranschaulicht, finden Sie den Datensatz (unter der Registerkarte „Daten“), geben Sie Vorhersagen ein und sehen Sie Ihr Ranking. Die URL ist hier:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![The house price prediction competition page.](../img/house_pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## Zugriff auf das Dataset und Lesen

Beachten Sie, dass die Wettbewerbsdaten in Trainings- und Testsätze unterteilt sind. Jeder Datensatz enthält den Eigenschaftswert des Hauses und Attribute wie Straßentyp, Baujahr, Dachtyp, Kellerzustand usw. Die Merkmale bestehen aus verschiedenen Datentypen. Beispielsweise wird das Baujahr durch eine ganze Zahl, der Dachtyp durch diskrete kategoriale Zuweisungen und andere Features durch Gleitkommazahlen dargestellt. Und hier ist, wo die Realität die Dinge erschwert: Für einige Beispiele fehlen einige Daten, wobei der fehlende Wert einfach als „na“ markiert ist. Der Preis jedes Hauses ist nur für das Trainingsset enthalten (es ist doch ein Wettkampf). Wir wollen das Schulungsset partitionieren, um einen Validierungssatz zu erstellen, aber wir können unsere Modelle erst nach dem Hochladen von Vorhersagen auf Kaggle auf dem offiziellen Testset bewerten. Die Registerkarte „Daten“ auf der Registerkarte Wettbewerb in :numref:`fig_house_pricing` enthält Links zum Herunterladen der Daten.

Um loszulegen, werden wir die Daten mit `pandas` einlesen und verarbeiten, die wir in :numref:`sec_pandas` eingeführt haben. Sie sollten also sicherstellen, dass Sie `pandas` installiert haben, bevor Sie fortfahren. Glücklicherweise, wenn Sie in Jupyter lesen, können wir Pandas installieren, ohne das Notebook zu verlassen.

```{.python .input}
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab tensorflow
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

Der Einfachheit halber können wir den Kaggle Gehäuse-Datensatz mit dem oben definierten Skript herunterladen und zwischenspeichern.

```{.python .input}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

Wir verwenden `pandas`, um die beiden CSV-Dateien mit Trainings- und Testdaten zu laden.

```{.python .input}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

Das Trainings-Dataset enthält 1460 Beispiele, 80 Features und 1 Label, während die Testdaten 1459 Beispiele und 80 Features enthalten.

```{.python .input}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

Lassen Sie uns einen Blick auf die ersten vier und letzten beiden Features sowie das Label (SalePrice) aus den ersten vier Beispielen werfen.

```{.python .input}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

Wir können sehen, dass in jedem Beispiel das erste Feature die ID ist. Dies hilft dem Modell, jedes Trainingsbeispiel zu identifizieren. Obwohl dies praktisch ist, enthält es keine Informationen für Vorhersagezwecke. Daher entfernen wir es aus dem Dataset, bevor die Daten in das Modell eingebracht werden.

```{.python .input}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## Vorverarbeitung von Daten

Wie oben erwähnt, haben wir eine Vielzahl von Datentypen. Wir müssen die Daten vorverarbeiten, bevor wir mit der Modellierung beginnen können. Beginnen wir mit den numerischen Merkmalen. Zuerst wenden wir eine Heuristik an, wobei alle fehlenden Werte durch den Mittelwert des entsprechenden Features ersetzt werden. Um dann alle Features auf einer gemeinsamen Skala zu setzen, standardisieren wir die Daten, indem wir Features auf Null Mittelwert und Einheitenvarianz neu skalieren:

$$x \leftarrow \frac{x - \mu}{\sigma}.$$

Um zu überprüfen, ob dies tatsächlich unsere Funktion (Variable) so transformiert, dass es Null Mittelwert und Einheitenvarianz hat, beachten Sie, dass $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$ und $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$. Intuitiv standardisieren wir die Daten aus zwei Gründen. Erstens erweist es sich als praktisch für die Optimierung. Zweitens, weil wir nicht wissen, *a priori* welche Funktionen relevant sind, wollen wir Koeffizienten, die einem Feature mehr als einem anderen Feature zugewiesen sind, nicht bestrafen.

```{.python .input}
#@tab all
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

Als nächstes beschäftigen wir uns mit diskreten Werten. Dazu gehören Funktionen wie „MSZoning“. Wir ersetzen sie durch eine One-Hot-Codierung auf die gleiche Weise, wie wir zuvor Multiklass-Labels in Vektoren transformiert haben (siehe :numref:`subsec_classification-problem`). Zum Beispiel nimmt „msZoning“ die Werte „RL“ und „RM“ an. Beim Löschen der Funktion „msZoning“ werden zwei neue Indikatorfunktionen „mszoning_rl“ und „mszoning_rm“ mit Werten entweder 0 oder 1 erstellt. Wenn der ursprüngliche Wert von „msZoning“ nach einer Hot-Codierung „RL“ ist, dann ist „mszoning_rl“ 1 und „mszoning_rm“ 0. Das Paket `pandas` erledigt dies automatisch für uns.

```{.python .input}
#@tab all
# `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

Sie können sehen, dass diese Konvertierung die Anzahl der Features von 79 auf 331 erhöht. Schließlich können wir über das Attribut `values` das NumPy-Format aus dem `pandas`-Format extrahieren und es in die Tensor-Darstellung für das Training konvertieren.

```{.python .input}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## Ausbildung

Zunächst trainieren wir ein lineares Modell mit quadratischem Verlust. Es überrascht nicht, dass unser lineares Modell nicht zu einer wettbewerbsgewinnenden Einreichung führen wird, aber es bietet eine vernünftige Prüfung, um zu sehen, ob es aussagekräftige Informationen in den Daten gibt. Wenn wir hier nicht besser als zufälliges Erraten können, dann könnte es eine gute Chance geben, dass wir einen Fehler bei der Datenverarbeitung haben. Und wenn die Dinge funktionieren, wird das lineare Modell als Grundlage dienen und uns eine gewisse Intuition darüber geben, wie nah das einfache Modell an die besten gemeldeten Modelle herankommt, was uns ein Gefühl dafür gibt, wie viel Gewinn wir von schickeren Modellen erwarten sollten.

```{.python .input}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return net
```

Bei Hauspreisen, wie bei Aktienkursen, kümmern wir uns um relative Mengen mehr als absolute Mengen. So neigen wir dazu, mehr über den relativen Fehler $\frac{y - \hat{y}}{y}$ zu kümmern als über den absoluten Fehler $y - \hat{y}$. Wenn zum Beispiel unsere Vorhersage bei der Schätzung des Preises eines Hauses in Rural Ohio um USD 100.000 ausfällt, wo der Wert eines typischen Hauses 125.000 USD beträgt, dann machen wir wahrscheinlich einen schrecklichen Job. Auf der anderen Seite, wenn wir in Los Altos Hills, Kalifornien, um diesen Betrag irren, könnte dies eine erstaunlich genaue Vorhersage darstellen (dort übersteigt der Medianpreis 4 Millionen USD).

Eine Möglichkeit, dieses Problem zu lösen, besteht darin, die Diskrepanz im Logarithmus der Preisschätzungen zu messen. In der Tat ist dies auch die offizielle Fehlermaßnahme, die vom Wettbewerb zur Bewertung der Qualität der Einreichungen verwendet wird. Schließlich übersetzt ein kleiner Wert $\delta$ für $|\log y - \log \hat{y}| \leq \delta$ in $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$. Dies führt zu folgendem Wurzel-Mittel-Quadrat-Fehler zwischen dem Logarithmus des prognostizierten Preises und dem Logarithmus des Etikettenpreises:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input}
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

```{.python .input}
#@tab pytorch
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(torch.mean(loss(torch.log(clipped_preds),
                                       torch.log(labels))))
    return rmse.item()
```

```{.python .input}
#@tab tensorflow
def log_rmse(y_true, y_pred):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

Anders als in den vorherigen Abschnitten basieren unsere Trainingsfunktionen auf dem Adam-Optimierer (wir werden ihn später ausführlicher beschreiben). Der Hauptanspruch dieses Optimierers besteht darin, dass trotz der unbegrenzten Ressourcen für die Hyperparameteroptimierung keine besseren (und manchmal schlimmer) gegeben werden, Menschen dazu neigen, zu finden, dass es deutlich weniger empfindlich auf die anfängliche Lernrate ist.

```{.python .input}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab pytorch
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab tensorflow
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls
```

## $K$-fach Kreuzvalidierung

Sie erinnern sich vielleicht, dass wir die $K$-fache Kreuzvalidierung in dem Abschnitt eingeführt haben, in dem wir besprochen haben, wie man mit der Modellauswahl umgeht (:numref:`sec_model_selection`). Wir werden dies gut nutzen, um das Modelldesign auszuwählen und die Hyperparameter anzupassen. Wir benötigen zunächst eine Funktion, die die $i^\mathrm{th}$-Faltung der Daten in einem $K$-fachen Kreuzvalidierungsverfahren zurückgibt. Es wird fortgesetzt, indem das $i^\mathrm{th}$-Segment als Validierungsdaten ausgeschnitten und der Rest als Trainingsdaten zurückgegeben wird. Beachten Sie, dass dies nicht der effizienteste Weg ist, Daten zu verarbeiten, und wir würden definitiv etwas viel intelligenteres tun, wenn unser Datensatz wesentlich größer wäre. Aber diese zusätzliche Komplexität könnte unseren Code unnötig verschleiern, so dass wir ihn hier aufgrund der Einfachheit unseres Problems sicher weglassen können.

```{.python .input}
#@tab all
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

Die Durchschnittswerte für Schulungs- und Verifizierungsfehler werden zurückgegeben, wenn wir $K$ mal in der $K$-fachen Kreuzvalidierung trainieren.

```{.python .input}
#@tab all
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs+1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse',
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## Modellauswahl

In diesem Beispiel wählen wir einen nicht abgestimmten Satz von Hyperparametern aus und überlassen es dem Leser, das Modell zu verbessern. Die Suche nach einer guten Wahl kann Zeit in Anspruch nehmen, je nachdem, wie viele Variablen man optimiert. Mit einem ausreichend großen Dataset und den normalen Arten von Hyperparametern ist die $K$-fache Kreuzvalidierung eher widerstandsfähig gegen mehrere Tests. Wenn wir jedoch eine unangemessen große Anzahl von Optionen versuchen, haben wir vielleicht nur Glück und finden, dass unsere Validierungsleistung nicht mehr repräsentativ für den wahren Fehler ist.

```{.python .input}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')
```

Beachten Sie, dass manchmal die Anzahl der Trainingsfehler für eine Reihe von Hyperparametern sehr gering sein kann, selbst wenn die Anzahl der Fehler bei der $K$-fachen Kreuzvalidierung erheblich höher ist. Dies deutet darauf hin, dass wir übermäßig sind. Während des Trainings werden Sie beide Zahlen überwachen wollen. Eine geringere Überanpassung könnte darauf hindeuten, dass unsere Daten ein leistungsfähigeres Modell unterstützen können. Massive Überrüstung könnte darauf hindeuten, dass wir durch die Einbeziehung von Regularisierungstechniken gewinnen können.

##  Vorhersagen auf Kaggle einreichen

Jetzt, da wir wissen, was eine gute Auswahl an Hyperparametern sein sollte, können wir auch alle Daten verwenden, um darauf zu trainieren (anstatt nur $1-1/K$ der Daten, die in den Kreuzvalidierungsscheiben verwendet werden). Das Modell, das wir auf diese Weise erhalten, kann dann auf den Testsatz angewendet werden. Das Speichern der Vorhersagen in einer CSV-Datei vereinfacht das Hochladen der Ergebnisse auf Kaggle.

```{.python .input}
#@tab all
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = d2l.numpy(net(test_features))
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

Ein schöner Vernunftigkeitsprüfung besteht darin, zu sehen, ob die Vorhersagen auf dem Testset denen des $K$-fachen Kreuzvalidierungsprozesses ähneln. Wenn dies der Fall ist, ist es an der Zeit, sie auf Kaggle hochzuladen. Der folgende Code erzeugt eine Datei namens `submission.csv`.

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

Als nächstes, wie in :numref:`fig_kaggle_submit2` gezeigt, können wir unsere Vorhersagen auf Kaggle einreichen und sehen, wie sie mit den tatsächlichen Immobilienpreisen (Etiketten) auf dem Testset vergleichen. Die Schritte sind ganz einfach:

* Melden Sie sich auf der Website von Kaggle an und besuchen Sie die Seite zum Preisvorhersagen des Hauses.
* Klicken Sie auf die Schaltfläche „Vorhersagen senden“ oder „Späte Übermittlung“ (ab diesem Schreiben befindet sich die Schaltfläche rechts).
* Klicken Sie im gestrichelten Feld unten auf der Seite auf die Schaltfläche „Übermittlungsdatei hochladen“ und wählen Sie die Prognosedatei aus, die Sie hochladen möchten.
* Klicken Sie unten auf der Seite auf die Schaltfläche „Senden“, um Ihre Ergebnisse anzuzeigen.

![Submitting data to Kaggle](../img/kaggle_submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Zusammenfassung

* Echte Daten enthalten oft eine Mischung aus verschiedenen Datentypen und müssen vorverarbeitet werden.
* Die Neukalierung von realen Werten auf den Mittelwert Null und die Varianz der Einheit ist ein guter Standard. So werden fehlende Werte durch ihren Mittelwert ersetzt.
* Die Umwandlung kategorialer Features in Indikatorfunktionen ermöglicht es uns, sie wie einheizende Vektoren zu behandeln.
* Wir können $K$-fache Kreuzvalidierung verwenden, um das Modell auszuwählen und die Hyperparameter anzupassen.
* Logarithmen sind für relative Fehler nützlich.

## Übungen

1. Senden Sie Ihre Prognosen für diesen Abschnitt an Kaggle. Wie gut sind Ihre Vorhersagen?
1. Können Sie Ihr Modell verbessern, indem Sie den Logarithmus der Preise direkt minimieren? Was passiert, wenn Sie versuchen, den Logarithmus des Preises und nicht den Preis vorherzusagen?
1. Ist es immer eine gute Idee, fehlende Werte durch ihren Mittelwert zu ersetzen? Tipp: Können Sie eine Situation konstruieren, in der die Werte nicht zufällig fehlen?
1. Verbessern Sie die Punktzahl auf Kaggle, indem Sie die Hyperparameter durch $K$-fache Kreuzvalidierung optimieren.
1. Verbessern Sie die Punktzahl, indem Sie das Modell verbessern (z. B. Schichten, Gewichtsabfall und Dropout).
1. Was passiert, wenn wir die kontinuierlichen numerischen Merkmale wie das, was wir in diesem Abschnitt getan haben, nicht standardisieren?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:
