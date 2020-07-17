# Datenvorverarbeitung
:label:`sec_pandas`

Bisher haben wir eine Vielzahl von Techniken eingeführt, um Daten zu manipulieren, die bereits in Tensoren gespeichert sind. Um Deep Learning auf die Lösung von realen Problemen anzuwenden, beginnen wir oft mit der Vorverarbeitung von Rohdaten, anstatt mit den gut vorbereiteten Daten im Tensor-Format. Unter den gängigen Datenanalyse-Tools in Python wird das Paket `pandas` häufig verwendet. Wie viele andere Erweiterungspakete im riesigen Python-Ökosystem kann `pandas` mit Tensoren zusammenarbeiten. So werden wir kurz Schritte zur Vorverarbeitung von Rohdaten mit `pandas` durchlaufen und in das Tensor-Format konvertieren. Wir werden mehr Datenvorverarbeitung Techniken in späteren Kapiteln abdecken.

## Lesen des Datensatzes

Als Beispiel erstellen wir zunächst ein künstliches Dataset, das in einer CSV-Datei (kommagetrennte Werte) `../data/house_tiny.csv` gespeichert ist. Daten, die in anderen Formaten gespeichert werden, können auf ähnliche Weise verarbeitet werden. Die folgende Funktion `mkdir_if_not_exist` stellt sicher, dass das Verzeichnis `../data` vorhanden ist. Beachten Sie, dass der Kommentar `# @save `eine spezielle Markierung ist, bei der die folgende Funktion, Klasse oder Anweisungen im Paket `d2l` gespeichert werden, sodass sie später direkt aufgerufen werden können (z. B. `d2l.mkdir_if_not_exist(path)`), ohne neu definiert zu werden.

```{.python .input}
#@tab all
import os

def mkdir_if_not_exist(path):  #@save
    """Make a directory if it does not exist."""
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)
```

Im Folgenden schreiben wir den Datensatz Zeile für Zeile in eine CSV-Datei.

```{.python .input}
#@tab all
data_file = '../data/house_tiny.csv'
mkdir_if_not_exist('../data')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data point
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

Um das Raw-Dataset aus der erstellten CSV-Datei zu laden, importieren wir das Paket `pandas` und rufen die Funktion `read_csv` auf. Dieses Dataset verfügt über vier Zeilen und drei Spalten, wobei jede Zeile die Anzahl der Räume („NumRooms“), den Gasetyp („Alley“) und den Preis („Preis“) eines Hauses beschreibt.

```{.python .input}
#@tab all
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Umgang mit fehlenden Daten

Beachten Sie, dass „NaN“ -Einträge Werte fehlen. Um fehlende Daten zu verarbeiten, umfassen typische Methoden *imputation* und *deletion*, wobei die Imputation fehlende Werte durch substituierte ersetzt, während beim Löschen fehlende Werte ignoriert werden. Hier werden wir die Zuordnung betrachten.

Durch die Integer-Location-basierte Indizierung (`iloc`) teilen wir `data` in `inputs` und `outputs` auf, wobei erstere die ersten beiden Spalten übernimmt, während letztere nur die letzte Spalte behält. Für numerische Werte in `inputs`, die fehlen, ersetzen wir die „NaN“ -Einträge durch den Mittelwert derselben Spalte.

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

Für kategoriale oder diskrete Werte in `inputs` betrachten wir „NaN“ als Kategorie. Da die Spalte „Alley“ nur zwei Arten von kategorischen Werten „Pave“ und „NaN“ akzeptiert, kann `pandas` diese Spalte automatisch in zwei Spalten „Alley_Pave“ und „Alley_NAN“ konvertieren. Eine Zeile, deren Alley-Typ „Pave“ ist, setzt die Werte von „Alley_Pave“ und „Alley_NAN“ auf 1 und 0. Eine Zeile mit einem fehlenden Gasetyp legt ihre Werte auf 0 und 1 fest.

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## Konvertierung in das Tensor-Format

Jetzt, da alle Einträge in `inputs` und `outputs` numerisch sind, können sie in das Tensor-Format konvertiert werden. Sobald Daten in diesem Format vorliegen, können sie weiter mit den Tensor-Funktionalitäten manipuliert werden, die wir in:numref:`sec_ndarray` eingeführt haben.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## Zusammenfassung

* Wie viele andere Erweiterungspakete im riesigen Ökosystem von Python kann `pandas` mit Tensoren zusammenarbeiten. * Imputation und Löschung können verwendet werden, um fehlende Daten zu verarbeiten.

## Übungen

Erstellen Sie ein Raw-Dataset mit mehr Zeilen und Spalten.

1. Löschen Sie die Spalte mit den meisten fehlenden Werten. 2. Konvertieren Sie das vorverarbeitete Dataset in das Tensor-Format.

:begin_tab:`mxnet`
[Diskussionen](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Diskussionen](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Diskussionen](https://discuss.d2l.ai/t/195)
:end_tab:
