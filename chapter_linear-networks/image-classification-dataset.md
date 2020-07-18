# Das Dataset zur Bildklassifizierung
:label:`sec_fashion_mnist`

Einer der weit verbreiteten Datasets für die Bildklassifizierung ist der MNIST-Datensatz :cite:`LeCun.Bottou.Bengio.ea.1998`. Obwohl es einen guten Lauf als Benchmark-Dataset hatte, erreichen selbst einfache Modelle nach heutigen Standards eine Klassifikationsgenauigkeit von über 95%, was es für die Unterscheidung zwischen stärkeren und schwächeren Modellen ungeeignet macht. Heute dient MNIST eher als Vernunftigkeitsprüfung als als als Benchmark. Um die Ante nur ein wenig zu heben, werden wir unsere Diskussion in den kommenden Abschnitten auf den qualitativ ähnlichen, aber vergleichsweise komplexen Fashion-MNIST-Datensatz :cite:`Xiao.Rasul.Vollgraf.2017` konzentrieren, der 2017 veröffentlicht wurde.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## Lesen des Datensatzes

Wir können den Fashion-MNIST-Datensatz über die eingebauten Funktionen im Framework herunterladen und in den Speicher lesen.

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST-Bilder besteht aus 10 Kategorien, die jeweils durch 6000 Bilder im Trainingsdatensatz und durch 1000 im Testdatensatz dargestellt werden. Ein *Testdataset* (oder *Testsatz*) wird für die Bewertung der Modellleistung und nicht für Schulungen verwendet. Folglich enthalten das Trainingset und das Testset 60000 bzw. 10000 Bilder.

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

Die Höhe und Breite jedes Eingabebildes betragen beide 28 Pixel. Beachten Sie, dass das Dataset aus Graustufenbildern besteht, deren Anzahl der Kanäle 1 beträgt. Aus Kürze, in diesem Buch speichern wir die Form eines beliebigen Bildes mit der Höhe $h$ Breite $w$ Pixel als $h \times w$ oder ($h$, $w$).

```{.python .input}
#@tab all
mnist_train[0][0].shape
```

Die Bilder in Fashion-MNist sind mit folgenden Kategorien verknüpft: T-Shirt, Hose, Pullover, Kleid, Mantel, Sandale, Hemd, Sneaker, Tasche und Stiefelette. Die folgende Funktion konvertiert zwischen numerischen Beschriftungsindizes und ihren Namen in Text.

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

Wir können nun eine Funktion erstellen, um diese Beispiele zu visualisieren.

```{.python .input}
#@tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

Hier sind die Bilder und die entsprechenden Beschriftungen (im Text) für die ersten Beispiele im Trainingsdatensatz.

```{.python .input}
X, y = mnist_train[:18]
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

## Minibatch lesen

Um unser Leben beim Lesen aus den Trainings- und Testsätzen zu erleichtern, verwenden wir den integrierten Dateniterator, anstatt einen von Grund auf neu zu erstellen. Daran erinnern, dass bei jeder Iteration ein Datenlader jedes Mal eine Minibatch von Daten mit der Größe `batch_size` liest. Wir mischen auch zufällig die Beispiele für den Trainingsdaten-Iterator.

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data expect for Windows."""
    return 0 if sys.platform.startswith('win') else 4

# `ToTensor` converts the image data from uint8 to 32-bit floating point. It
# divides all numbers by 255 so that all pixel values are between 0 and 1
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data."""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

Schauen wir uns die Zeit an, die es braucht, um die Trainingsdaten zu lesen.

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## Alles zusammenstellen

Jetzt definieren wir die Funktion `load_data_fashion_mnist`, die den Fashion-MNIST-Datensatz abruft und liest. Es gibt die Dateniteratoren für den Trainingssatz und Validierungssatz zurück. Darüber hinaus akzeptiert es ein optionales Argument, um die Größe der Bilder auf eine andere Form zu ändern.

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

Im Folgenden testen wir die Bildgrößenfunktion der Funktion `load_data_fashion_mnist`, indem wir das Argument `resize` angeben.

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

Wir sind nun bereit, mit dem Fashion-MNIST-Dataset in den folgenden Abschnitten zu arbeiten.

## Zusammenfassung

* Fashion-MNist ist ein Bekleidungsklassifizierungsdatensatz, der aus Bildern besteht, die 10 Kategorien darstellen. Wir werden diesen Datensatz in den nachfolgenden Abschnitten und Kapiteln verwenden, um verschiedene Klassifizierungsalgorithmen auszuwerten.
* Wir speichern die Form eines Bildes mit der Höhe $h$ Breite $w$ Pixel als $h \times w$ oder ($h$, $w$).
* Dateniteratoren sind eine Schlüsselkomponente für eine effiziente Leistung. Verlassen Sie sich auf gut implementierte Dateniteratoren, die High-Performance-Computing nutzen, um eine Verlangsamung Ihres Trainingskreises zu vermeiden.

## Übungen

1. Beeinflusst die Reduzierung des `batch_size` (z. B. auf 1) die Leseleistung?
1. Die Dateniterator-Performance ist wichtig. Glauben Sie, dass die aktuelle Implementierung schnell genug ist? Entdecken Sie verschiedene Optionen, um es zu verbessern.
1. Schauen Sie sich die Online-API-Dokumentation des Frameworks an. Welche anderen Datensätze stehen zur Verfügung?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:
