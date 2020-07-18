# Mehrschichtige Perzeptrons
:label:`chap_perceptrons`

In diesem Kapitel stellen wir Ihnen Ihr erstes wirklich *deep* Netzwerk vor. Die einfachsten tiefen Netzwerke werden mehrschichtige Wahrnehmungen genannt, und sie bestehen aus mehreren Schichten von Neuronen, die jeweils vollständig mit denen in der unteren Schicht verbunden sind (von denen sie Eingang erhalten) und denen oben (die sie wiederum beeinflussen). Wenn wir Modelle mit hoher Kapazität trainieren, besteht die Gefahr einer Überrüstung. Daher müssen wir Ihre erste strikte Einführung in die Begriffe Überanpassung, Underfitting und Modellauswahl geben. Um Ihnen zu helfen, diese Probleme zu bekämpfen, werden wir Regularisierungstechniken wie Gewichtszerfall und Dropout einführen. Wir werden auch Fragen im Zusammenhang mit numerischer Stabilität und Parameter-Initialisierung diskutieren, die für eine erfolgreiche Schulung tiefer Netzwerke von entscheidender Bedeutung sind. Unser Ziel ist es, Ihnen nicht nur die Konzepte, sondern auch die Praxis der Nutzung von Deep Networks festzuhalten. Am Ende dieses Kapitels wenden wir das, was wir bisher eingeführt haben, auf einen realen Fall an: die Vorhersage der Hauspreise. Wir stellen Fragen in Bezug auf die Rechenleistung, Skalierbarkeit und Effizienz unserer Modelle zu den nachfolgenden Kapiteln.

```toc
:maxdepth: 2

mlp
mlp-scratch
mlp-concise
underfit-overfit
weight-decay
dropout
backprop
numerical-stability-and-init
environment
kaggle-house-price
```
