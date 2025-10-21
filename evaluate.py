import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def load_model():
    """
    Carga el modelo entrenado.
    
    Returns:
        tf.keras.Model: Modelo cargado
    """
    print("📥 Cargando modelo entrenado...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo cargado exitosamente")
    return model

def load_test_data():
    """
    Carga los datos de prueba.
    
    Returns:
        tuple: (test_ds, class_names)
    """
    print("📂 Cargando datos de prueba...")
    
    # Cargar dataset de prueba
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )
    
    # Obtener nombres de las clases
    class_names = test_ds.class_names
    
    
    print(f"✅ Datos de prueba cargados exitosamente")
    print(f"   🌸 Clases: {class_names}")
    print(f"   📊 Test batches: {len(test_ds)}")
    
    return test_ds, class_names

def get_predictions(model, test_ds):
    """
    Obtiene las predicciones del modelo en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado
        test_ds: Dataset de prueba
        
    Returns:
        tuple: (y_true, y_pred, y_pred_probs)
    """
    print("🔮 Obteniendo predicciones...")
    
    # Listas para almacenar datos
    all_images = []
    all_labels = []
    
    # Extraer todas las imágenes y etiquetas
    for images, labels in test_ds:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())
    
    # Concatenar en arrays de NumPy
    images = np.concatenate(all_images, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    print(f"   📊 Total de imágenes procesadas: {len(images)}")
    
    # Obtener predicciones
    y_pred_probs = model.predict(images, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("✅ Predicciones obtenidas exitosamente")
    
    return y_true, y_pred, y_pred_probs

def show_classification_report(y_true, y_pred, class_names):
    """
    Muestra el reporte de clasificación.
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones
        class_names: Nombres de las clases
    """
    print("\n" + "="*60)
    print("--- Reporte de Clasificación ---")
    print("="*60)
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Genera y guarda la matriz de confusión.
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones
        class_names: Nombres de las clases
    """
    print("📊 Generando matriz de confusión...")
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Crear figura
    plt.figure(figsize=(10, 8))
    
    # Crear heatmap con seaborn
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Número de muestras'}
    )
    
    # Configurar el gráfico
    plt.title('Matriz de Confusión - Clasificación de Flores', fontsize=16, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar la imagen
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Matriz de confusión guardada como 'confusion_matrix.png'")
    
    # Mostrar el gráfico
    plt.show()

def calculate_accuracy(y_true, y_pred):
    """
    Calcula la precisión general del modelo.
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones
        
    Returns:
        float: Precisión del modelo
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total * 100
    
    print(f"\n🎯 PRECISIÓN GENERAL: {accuracy:.2f}%")
    print(f"   ✅ Correctas: {correct}/{total}")
    
    return accuracy

if __name__ == "__main__":
    print("🌻 EVALUACIÓN DEL MODELO DE CLASIFICACIÓN DE FLORES")
    print("="*60)
    
    # Parámetros
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    MODEL_PATH = 'flower_model.keras'
    TEST_DIR = 'data/test'
    
    # Cargar modelo
    model = load_model()
    
    # Cargar datos de prueba
    test_ds, class_names = load_test_data()
    
    # Obtener predicciones
    y_true, y_pred, y_pred_probs = get_predictions(model, test_ds)
    
    # Calcular y mostrar precisión
    accuracy = calculate_accuracy(y_true, y_pred)
    
    # Mostrar reporte de clasificación
    show_classification_report(y_true, y_pred, class_names)
    
    # Generar matriz de confusión
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    print("\n🎉 ¡Evaluación completada exitosamente!")
    print("📁 Archivos generados:")
    print("   - confusion_matrix.png")
