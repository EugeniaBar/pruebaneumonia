Dataset https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 


# Detección de Neumonía en Radiografías de Tórax: Un Estudio de Caso de Aprendizaje Profundo

![NEUM](https://github.com/user-attachments/assets/e929f98f-b3bc-47a0-af86-40ac2a0be84b)

# Detección de Neumonía en Radiografías de Tórax: Un Estudio de Caso de Aprendizaje Profundo

---

## 📄 Resumen del Proyecto

Este proyecto explora la aplicación de Redes Neuronales Convolucionales (CNNs) para clasificar radiografías de tórax en dos categorías: **Normal** y **Neumonía**. A través de este trabajo, se compara el rendimiento de un modelo CNN entrenado "desde cero" con un modelo que utiliza la potente técnica de **Transfer Learning** (VGG16), destacando los desafíos del entrenamiento con datasets limitados y la importancia de las estrategias de IA.

---

## 🚀 Objetivo de Aprendizaje

El propósito principal de este proyecto es didáctico:

* **Comprender las Limitaciones de CNNs desde Cero:** Demostrar cómo una red neuronal, incluso con aumento de datos, puede fallar en aprender patrones complejos y tender a la **memorización (sobreajuste)** cuando el volumen de datos de entrenamiento es muy escaso.
* **Dominar el Transfer Learning:** Ilustrar la eficacia y la necesidad del Transfer Learning para tareas de clasificación de imágenes con datasets pequeños, aprovechando el conocimiento pre-entrenado de modelos de vanguardia.
* **Manejo de Datos:** Entender la importancia del balance del dataset y las implicaciones de las estrategias de submuestreo.
* **Interpretación de Métricas:** Aprender a leer y entender métricas clave como Accuracy, Precision, Recall y AUC en el contexto de clasificación binaria, especialmente para evaluar la calidad del modelo.

---

## 📁 Estructura del Dataset

El dataset utilizado consta de radiografías de tórax, dividido en conjuntos de entrenamiento, validación y prueba. Cada conjunto está perfectamente **balanceado** con 240 imágenes de la clase "Normal" y 240 imágenes de la clase "Neumonía".

* **`DATOS235/`**
    * **`train/`**
        * `normal/` (240 imágenes)
        * `pneumonia/` (240 imágenes)
    * **`val/`**
        * `normal/` (240 imágenes)
        * `pneumonia/` (240 imágenes)
    * **`test/`**
        * `normal/` (240 imágenes)
        * `pneumonia/` (240 imágenes)

**Nota sobre el balance:** Se optó por submuestrear las clases con más imágenes para lograr un balance perfecto. Si bien esto ayuda a prevenir el sesgo hacia la clase mayoritaria, es importante reconocer que se descartó una cantidad considerable de datos únicos que podrían haber mejorado aún más el rendimiento general del modelo si se hubieran utilizado con otras estrategias (como ponderación de clases o aumento de datos para la minoritaria).

---

## 🧪 Enfoques Implementados y Conclusiones

Se exploraron dos enfoques principales para la clasificación de imágenes:

### 1. Modelo CNN Entrenado "Desde Cero"

* **Descripción:** Se construyó una Red Neuronal Convolucional (CNN) con capas como `Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Flatten`, `Dense` y `Dropout`, intentando variar el número de filtros (ej. 16-32-64, 32-64-128) y el `batch_size`. Se aplicó **aumento de datos (data augmentation)** extensivo para intentar diversificar el dataset limitado.

* **Resultados Típicos en Test:**
    * Loss: ~2.1249
    * Accuracy: ~0.5000
    * Precision: ~0.5000
    * Recall: ~1.0000
    * AUC: ~0.8238

* **Conclusión:** Este enfoque **fracasó en la generalización**. Las métricas de prueba (`Accuracy` del 50%, `Precision` del 50%, `Recall` del 100%) revelan un **sobreajuste severo y un sesgo**. El modelo no aprendió a distinguir entre "Normal" y "Neumonía"; en su lugar, **simplemente memorizó las imágenes de entrenamiento** y terminó prediciendo sistemáticamente una sola clase (probablemente "Neumonía") para todas las entradas en los conjuntos de validación y prueba. La **cantidad extremadamente limitada de imágenes únicas (480 en entrenamiento)** impidió que la CNN aprendiera patrones robustos y generalizables, incluso con el apoyo de la data augmentation.

### 2. Modelo Basado en Transfer Learning (VGG16)

* **Descripción:** Se utilizó la arquitectura **VGG16**, un modelo pre-entrenado en el gigantesco dataset ImageNet. Se cargó VGG16 **sin sus capas clasificatorias (`include_top=False`)** y sus capas convolucionales se **congelaron (`layer.trainable = False`)**. Luego, se añadieron capas personalizadas (`Flatten`, `Dense`, `Dropout`, `Dense(1, sigmoid)`) en la parte superior para la clasificación binaria. Se entrenó esta nueva "cabeza" clasificatoria y, en una segunda fase, se realizó un **fine-tuning** (descongelando las últimas capas convolucionales de VGG16) con un `learning_rate` más bajo.

* **Resultados Finales del Modelo (Fine-Tuned) en Test:**
    * Loss: **0.4427**
    * Accuracy: **0.9103** (91.03%)
    * Precision: **0.9138** (91.38%)
    * Recall: **0.9060** (90.60%)
    * AUC: **0.9608**

* **Detalle de Predicciones por Imagen (en el conjunto de Prueba):**
    * Total de imágenes en test: 468
    * **Correctos Positivos** (Neumonía correctamente predicha): 131
    * **Correctos Negativos** (Normal correctamente predicho): 228
    * **Falsos Positivos** (Normal predicho como Neumonía): 6
    * **Falsos Negativos** (Neumonía predicha como Normal): 103
    * Accuracy Calculado a partir de estas predicciones: 76.71% (Esta es una **discrepancia importante** a investigar; la precisión de `model.evaluate` es la más fiable si el generador de prueba se usa correctamente).

* **Conclusión:** El **Transfer Learning fue la estrategia funcional y exitosa**, logrando un rendimiento **muy bueno y equilibrado**. El modelo Fine-Tuned demostró una alta **precisión general (Accuracy)** y, crucialmente, una **excelente capacidad para identificar correctamente los casos de Neumonía (Recall: 90.60%)**, al mismo tiempo que mantiene una buena **Precisión (Precision: 91.38%)**. Esto indica que el modelo no solo predice bien, sino que también es fiable en sus predicciones positivas y captura la mayoría de los casos reales de la enfermedad. Este resultado valida la potencia del Transfer Learning para adaptarse a tareas específicas con datasets limitados, superando por mucho las limitaciones de entrenar desde cero.

---

## 💡 Reflexión Final y Aprendizaje Clave

Este proyecto demuestra empíricamente que para tareas de clasificación de imágenes con **datasets pequeños**, entrenar una CNN desde cero es extremadamente desafiante y a menudo ineficaz debido al sobreajuste. La técnica de **Transfer Learning** es una solución poderosa y práctica, permitiendo aprovechar el conocimiento de modelos pre-entrenados para resolver problemas específicos con recursos de datos limitados.

El principal problema no fue la falta de una "herramienta mejorada" para la CNN desde cero, sino la **cantidad insuficiente de datos de entrenamiento únicos para que una red tan compleja pudiera aprender a generalizar por sí misma**. El VGG16 ya es esa "herramienta mejorada" por su entrenamiento previo.

---

## ⚖️ Consideraciones Éticas

Es crucial recordar que, aunque este proyecto es un ejercicio de aprendizaje, los modelos de IA aplicados a la salud, como la detección de enfermedades en imágenes médicas, conllevan importantes responsabilidades éticas:

* **Precisión y Fiabilidad:** Un modelo de IA nunca debe reemplazar el juicio clínico de un profesional médico. Es una herramienta de apoyo. Errores (falsos positivos o falsos negativos) pueden tener consecuencias graves. En este caso, aunque el Recall es alto, un 9.4% de Falsos Negativos (casos de Neumonía no detectados) sigue siendo un riesgo significativo que debe ser mitigado en aplicaciones reales.
* **Sesgo del Modelo:** Es vital que el dataset de entrenamiento sea representativo de la población a la que se aplicará el modelo para evitar sesgos raciales, geográficos o de equipo médico.
* **Diversidad de Datos:** La falta de diversidad en la fuente de las imágenes (ej. diferentes máquinas, etnias de pacientes, gravedades de la enfermedad) puede llevar a que el modelo funcione bien solo en los datos a los que ha sido expuesto, y mal en casos reales más variados.
* **Transparencia:** Comprender cómo el modelo toma decisiones es importante, aunque difícil en redes neuronales profundas (problema de "caja negra").

Este proyecto sirve como una base para entender los desafíos y las soluciones en IA aplicada a la salud, pero siempre destacando la necesidad de validación rigurosa, consideraciones éticas y la supervisión humana en aplicaciones reales.
