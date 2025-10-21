# 🌸 Clasificador de Flores (Rosas, Margaritas, Girasoles)

## Descripción del Proyecto

Este proyecto implementa un clasificador de imágenes de flores utilizando técnicas de **Deep Learning** y **Transfer Learning**. El modelo utiliza una arquitectura CNN basada en **MobileNetV2** pre-entrenada en ImageNet para clasificar automáticamente tres tipos de flores: rosas, margaritas y girasoles.

El sistema incluye un pipeline completo de Machine Learning que va desde la organización de datos hasta el despliegue de una aplicación web interactiva usando **Streamlit**.

## Resultados Obtenidos

El modelo final alcanzó una **precisión (accuracy) del 93%** en el conjunto de prueba, demostrando excelente capacidad de clasificación para las tres clases de flores.

## Tecnologías Utilizadas

- **Python** - Lenguaje de programación principal
- **TensorFlow / Keras** - Framework de Deep Learning
- **Streamlit** - Framework para aplicaciones web interactivas
- **Scikit-learn** - Herramientas de evaluación y métricas
- **Numpy & Pillow** - Procesamiento de imágenes y arrays
- **Matplotlib & Seaborn** - Visualización de datos

## Estructura del Proyecto

### Scripts Principales

- **`organize_dataset.py`** - Script para dividir y organizar el dataset en carpetas de entrenamiento, validación y prueba
- **`train.py`** - Script principal de entrenamiento que implementa Transfer Learning con MobileNetV2 y fine-tuning
- **`evaluate.py`** - Script de evaluación que genera reportes de clasificación y matrices de confusión
- **`app.py`** - Aplicación web interactiva desarrollada con Streamlit para clasificación en tiempo real

### Archivos de Configuración

- **`requirements.txt`** - Lista de dependencias del proyecto
- **`.gitignore`** - Archivos y carpetas excluidos del control de versiones

## Cómo Ejecutarlo

### Prerrequisitos

- Python 3.8 o superior
- Dataset de flores (debe estar en la carpeta `dataset_original/`)

### Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <url-del-repositorio>
   cd ClasificacionFloresIA
   ```

2. **Crear y activar un entorno virtual**
   ```bash
   # Crear entorno virtual
   python -m venv venv
   
   # Activar entorno virtual
   # En Windows:
   .\venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Instalar las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Preparar el dataset**
   - Descargar el dataset de flores de Kaggle
   - Colocar los archivos en la carpeta `dataset_original/`
   - La estructura debe ser:
     ```
     dataset_original/
     ├── rosas/
     ├── margaritas/
     └── girasoles/
     ```

### Ejecución del Pipeline

1. **Organizar el dataset**
   ```bash
   python organize_dataset.py
   ```
   Esto creará la estructura `data/train/`, `data/val/`, `data/test/` con las divisiones apropiadas.

2. **Entrenar el modelo**
   ```bash
   python train.py
   ```
   El entrenamiento incluye:
   - Transfer Learning con MobileNetV2 congelado (15 épocas)
   - Fine-tuning de las últimas 30 capas (10 épocas)
   - Guardado automático del modelo como `flower_model.keras`

3. **Evaluar el modelo**
   ```bash
   python evaluate.py
   ```
   Genera:
   - Reporte de clasificación detallado
   - Matriz de confusión visual (`confusion_matrix.png`)

4. **Ejecutar la aplicación web**
   ```bash
   streamlit run app.py
   ```
   La aplicación estará disponible en `http://localhost:8501`

## Características del Modelo

- **Arquitectura**: MobileNetV2 pre-entrenada en ImageNet
- **Técnica**: Transfer Learning + Fine-tuning
- **Data Augmentation**: Rotación, flip horizontal, zoom, color jitter
- **Optimización**: Adam optimizer con learning rate adaptativo
- **Precisión**: 93% en conjunto de prueba

## Uso de la Aplicación Web

1. Abre la aplicación en tu navegador
2. Sube una imagen de rosa, margarita o girasol
3. El modelo procesará la imagen automáticamente
4. Recibe la predicción con nivel de confianza
5. Explora las probabilidades detalladas para cada clase

## Estructura de Datos

```
ClasificacionFloresIA/
├── dataset_original/          # Dataset original (ignorado por Git)
│   ├── rosas/
│   ├── margaritas/
│   └── girasoles/
├── data/                      # Dataset organizado (ignorado por Git)
│   ├── train/
│   ├── val/
│   └── test/
├── organize_dataset.py        # Script de organización
├── train.py                   # Script de entrenamiento
├── evaluate.py                # Script de evaluación
├── app.py                     # Aplicación Streamlit
├── requirements.txt           # Dependencias
├── .gitignore                # Archivos ignorados
└── README.md                 # Este archivo
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Desarrollado con ❤️ usando TensorFlow, Streamlit y Python**
