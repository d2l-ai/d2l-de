# Installation
:label:`chap_installation`

Um Ihnen praktische Lernerfahrungen zur Verfügung zu stellen, müssen Sie eine Umgebung für die Ausführung von Python, Jupyter-Notizbüchern, den relevanten Bibliotheken und dem Code einrichten, der zum Ausführen des Buches selbst benötigt wird.

## Installieren von Miniconda

Der einfachste Weg, um loszukommen, ist die Installation von [Miniconda](https://conda.io/en/latest/miniconda.html). Die Python 3.x-Version ist erforderlich. Sie können die folgenden Schritte überspringen, wenn conda bereits installiert ist. Laden Sie die entsprechende Miniconda sh-Datei von der Website herunter und führen Sie die Installation über die Befehlszeile mit `sh <FILENAME> -b` aus. Für macOS-Benutzer:

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

Für Linux-Benutzer:

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Als nächstes initialisieren Sie die Shell, damit wir `conda` direkt ausführen können.

```bash
~/miniconda3/bin/conda init
```

Schließen Sie nun Ihre aktuelle Shell und öffnen Sie sie erneut. Sie sollten in der Lage sein, eine neue Umgebung wie folgt zu erstellen:

```bash
conda create --name d2l -y
```

## Herunterladen der D2L-Notebooks

Als nächstes müssen wir den Code dieses Buches herunterladen. Sie können auf die Registerkarte „Alle Notizbücher“ oben auf jeder HTML-Seite klicken, um den Code herunterzuladen und zu entpacken. Alternativ, wenn Sie `unzip` haben (ansonsten `sudo apt install unzip` ausführen):

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Jetzt möchten wir die `d2l` Umgebung aktivieren und `pip` installieren. Geben Sie `y` für die Abfragen ein, die diesem Befehl folgen.

```bash
conda activate d2l
conda install python=3.7 pip -y
```

## Installieren des Frameworks und des `d2l`-Pakets

:begin_tab:`mxnet,pytorch`
Bevor Sie das Deep Learning-Framework installieren, überprüfen Sie bitte zuerst, ob Sie die richtigen GPUs auf Ihrem Computer haben (die GPUs, die das Display auf einem Standard-Laptop betreiben, zählen nicht für unsere Zwecke). Wenn Sie auf einem GPU-Server installieren, fahren Sie mit :ref:`subsec_gpu` fort, um Anweisungen zum Installieren einer GPU-unterstützten Version zu erhalten.

Andernfalls können Sie die CPU-Version installieren. Das wird mehr als genug PS sein, um Sie durch die ersten Kapitel zu bekommen, aber Sie möchten auf GPUs zugreifen, bevor Sie größere Modelle ausführen.
:end_tab:

:begin_tab:`mxnet`
```bash
pip install mxnet==1.6.0
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch==1.5.1 torchvision -f https://download.pytorch.org/whl/torch_stable.html
```
:end_tab:

:begin_tab:`tensorflow`
Sie können TensorFlow mit CPU- und GPU-Unterstützung über Folgendes installieren:

```bash
pip install tensorflow==2.2.0 tensorflow-probability==0.10.0
```
:end_tab:

Wir installieren auch das Paket `d2l`, das häufig verwendete Funktionen und Klassen in diesem Buch kapselt.

```bash
pip install -U d2l
```

Sobald sie installiert sind, öffnen wir nun das Jupyter Notebook, indem Sie Folgendes ausführen:

```bash
jupyter notebook
```

An dieser Stelle können Sie http://localhost:8888 (es wird normalerweise automatisch geöffnet) in Ihrem Webbrowser öffnen. Dann können wir den Code für jeden Abschnitt des Buches ausführen. Führen Sie immer `conda activate d2l` aus, um die Laufzeitumgebung zu aktivieren, bevor Sie den Code des Buches ausführen oder das Deep Learning-Framework oder das Paket `d2l` aktualisieren. Führen Sie `conda deactivate` aus, um die Umgebung zu beenden.

## GPU-Unterstützung
:label:`subsec_gpu`

:begin_tab:`mxnet,pytorch`
Standardmäßig wird das Deep Learning-Framework ohne GPU-Unterstützung installiert, um sicherzustellen, dass es auf jedem Computer (einschließlich der meisten Laptops) ausgeführt wird. Ein Teil dieses Buches erfordert oder empfiehlt, mit GPU zu arbeiten. Wenn Ihr Computer über NVIDIA-Grafikkarten verfügt und [CUDA](https://developer.nvidia.com/cuda-downloads) installiert ist, sollten Sie eine GPU-fähige Version installieren. Wenn Sie die Nur-CPU-Version installiert haben, müssen Sie sie möglicherweise zuerst entfernen, indem Sie Folgendes ausführen:
:end_tab:

:begin_tab:`tensorflow`
Standardmäßig wird TensorFlow mit GPU-Unterstützung installiert. Wenn Ihr Computer über NVIDIA-Grafikkarten verfügt und [CUDA](https://developer.nvidia.com/cuda-downloads) installiert ist, sind Sie alle eingestellt.
:end_tab:

:begin_tab:`mxnet`
```bash
pip uninstall mxnet
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip uninstall torch
```
:end_tab:

:begin_tab:`mxnet,pytorch`
Dann müssen wir die CUDA-Version finden, die Sie installiert haben. Sie können es durch `nvcc --version` oder `cat /usr/local/cuda/version.txt` überprüfen. Angenommen Sie, Sie haben CUDA 10.1 installiert, dann können Sie mit dem folgenden Befehl installieren:
:end_tab:

:begin_tab:`mxnet`
```bash
# For Windows users
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
pip install mxnet-cu101==1.6.0
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
:end_tab:

:begin_tab:`mxnet,pytorch`
Sie können die letzten Ziffern entsprechend Ihrer CUDA-Version ändern, z.B. `cu100` für CUDA 10.0 und `cu90` für CUDA 9.0.
:end_tab:

## Übungen

1. Laden Sie den Code für das Buch herunter und installieren Sie die Laufzeitumgebung.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:
