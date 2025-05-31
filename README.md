Dataset https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 


# Detección de Neumonía en Radiografías de Tórax: Un Estudio de Caso de Aprendizaje Profundo

---

## 📄 Resumen del Proyecto

Este proyecto explora la aplicación de Redes Neuronales Convolucionales (CNNs) para clasificar radiografías de tórax en dos categorías: **Normal** y **Neumonía**. A través de este trabajo, se compara el rendimiento de un modelo CNN entrenado "desde cero" (manual) con un modelo que utiliza la potente técnica de **Transfer Learning** (VGG16), destacando los desafíos del entrenamiento con datasets limitados y la importancia de las estrategias de IA.

---

## 🚀 Objetivo de Aprendizaje

El propósito principal de este proyecto es didáctico:

* **Comprender las Limitaciones de CNNs desde Cero:** Demostrar cómo una red neuronal, incluso con aumento de datos, puede fallar en aprender patrones complejos y tender a la **memorización (sobreajuste)** cuando el volumen de datos de entrenamiento es muy escaso.
* **Dominar el Transfer Learning:** Ilustrar la eficacia y la necesidad del Transfer Learning para tareas de clasificación de imágenes con datasets pequeños, aprovechando el conocimiento pre-entrenado de modelos de vanguardia.
* **Manejo de Datos:** Entender la importancia del balance del dataset y las implicaciones de las estrategias de submuestreo vs. sobremuestreo/ponderación de clases.
* **Interpretación de Métricas:** Aprender a leer y entender métricas clave como Accuracy, Precision, Recall y AUC en el contexto de clasificación binaria, especialmente para detectar problemas como el sesgo del modelo.

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

* **Resultados Reportados (Ejemplo de éxito):**
    * Accuracy: ~76.71%
    * Precision: ~95.62%
    * Recall: ~55.98%

* **Conclusión:** El **Transfer Learning fue la estrategia funcional y exitosa**. A diferencia del modelo desde cero, VGG16 ya posee un vasto "conocimiento" de características visuales generales (bordes, texturas, formas) aprendidas de millones de imágenes. Al congelar sus capas, lo utilizamos como un potente extractor de características, y solo entrenamos las pocas capas nuevas para nuestra tarea específica. Esto **evitó el sobreajuste severo** y permitió al modelo aprender patrones discriminativos de las radiografías de tórax, obteniendo un rendimiento significativamente superior y útil.

---

## 💡 Reflexión Final y Aprendizaje Clave

Este proyecto demuestra empíricamente que para tareas de clasificación de imágenes con **datasets pequeños**, entrenar una CNN desde cero es extremadamente desafiante y a menudo ineficaz debido al sobreajuste. La técnica de **Transfer Learning** es una solución poderosa y práctica, permitiendo aprovechar el conocimiento de modelos pre-entrenados para resolver problemas específicos con recursos de datos limitados.

El principal problema no fue la falta de una "herramienta mejorada" para la CNN desde cero, sino la **cantidad insuficiente de datos de entrenamiento únicos para que una red tan compleja pudiera aprender a generalizar por sí misma**. El VGG16 ya es esa "herramienta mejorada" por su entrenamiento previo.

---

## ⚖️ Consideraciones Éticas

Es crucial recordar que, aunque este proyecto es un ejercicio de aprendizaje, los modelos de IA aplicados a la salud, como la detección de enfermedades en imágenes médicas, conllevan importantes responsabilidades éticas:

* **Precisión y Fiabilidad:** Un modelo de IA nunca debe reemplazar el juicio clínico de un profesional médico. Es una herramienta de apoyo. Errores (falsos positivos o falsos negativos) pueden tener consecuencias graves.
* **Sesgo del Modelo:** El rendimiento del `Recall` (capacidad de detectar todos los casos positivos) es especialmente crítico en diagnóstico médico. Un `Recall` bajo (como el ~56% en este ejemplo) significa que el modelo falla en identificar casi la mitad de los casos de neumonía, lo cual es inaceptable en un escenario clínico. Es vital optimizar esta métrica.
* **Diversidad de Datos:** Si bien se balanceó el dataset, la falta de diversidad en la fuente de las imágenes (ej. diferentes máquinas, etnias de pacientes, gravedades de la enfermedad) puede llevar a que el modelo funcione bien solo en los datos a los que ha sido expuesto, y mal en casos reales más variados.
* **Transparencia:** Comprender cómo el modelo toma decisiones es importante, aunque difícil en redes neuronales profundas (problema de "caja negra").

Este proyecto sirve como una base para entender los desafíos y las soluciones en IA aplicada a la salud, pero siempre destacando la necesidad de validación rigurosa, consideraciones éticas y la supervisión humana en aplicaciones reales.
