import tensorflow as tf
import matplotlib.pyplot as plt

def load_data():
    """
    Carga los datasets de entrenamiento, validación y prueba.
    
    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    print("📂 Cargando datasets...")
    
    # Parámetros para cargar datos
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    DATA_DIR = 'data'
    
    # Cargar dataset de entrenamiento
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'{DATA_DIR}/train',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # Cargar dataset de validación
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f'{DATA_DIR}/val',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # Cargar dataset de prueba
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f'{DATA_DIR}/test',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # Obtener nombres de las clases
    class_names = train_ds.class_names
    
    print(f"✅ Datasets cargados exitosamente")
    print(f"   🌸 Clases: {class_names}")
    print(f"   📊 Train batches: {len(train_ds)}")
    print(f"   📊 Val batches: {len(val_ds)}")
    print(f"   📊 Test batches: {len(test_ds)}")
    
    return train_ds, val_ds, test_ds, class_names

def create_augmentation_layer():
    """
    Crea una capa de aumento de datos.
    
    Returns:
        tf.keras.Sequential: Capa de aumento de datos
    """
    print("🔄 Creando capa de aumento de datos...")
    
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2)
    ])
    
    print("✅ Capa de aumento de datos creada")
    return data_augmentation

def build_model(num_classes):
    """
    Construye el modelo de clasificación usando Transfer Learning con MobileNetV2.
    
    Args:
        num_classes (int): Número de clases a clasificar
        
    Returns:
        tf.keras.Model: Modelo compilado
    """
    print(f"🏗️  Construyendo modelo para {num_classes} clases...")
    
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    
    # Cargar MobileNetV2 pre-entrenado
    print("   📥 Cargando MobileNetV2 pre-entrenado...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar el modelo base
    base_model.trainable = False
    print("   ❄️  Modelo base congelado")
    
    # Crear el modelo completo
    print("   🔧 Construyendo arquitectura del modelo...")
    model = tf.keras.Sequential([
        create_augmentation_layer(),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar el modelo
    print("   ⚙️  Compilando modelo...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modelo construido y compilado exitosamente")
    return model, base_model

if __name__ == "__main__":
    print("🌻 INICIANDO ENTRENAMIENTO DEL MODELO DE CLASIFICACIÓN DE FLORES")
    print("="*70)
    
    # Parámetros de entrenamiento
    EPOCHS = 15
    FINE_TUNE_EPOCHS = 10
    
    # Cargar datos
    train_ds, val_ds, test_ds, class_names = load_data()
    
    # Construir modelo
    num_classes = len(class_names)
    model, base_model = build_model(num_classes)
    
    # Mostrar resumen del modelo
    print("\n📋 RESUMEN DEL MODELO:")
    print("-" * 50)
    model.summary()
    
    # Entrenar el modelo
    print(f"\n🚀 INICIANDO ENTRENAMIENTO ({EPOCHS} épocas)...")
    print("-" * 50)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Fine-tuning
    print("\n--- Iniciando Fine-Tuning ---")
    
    # Descongelar el modelo base
    base_model.trainable = True
    
    # Congelar la mayoría de las capas (excepto las últimas 30)
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Re-compilar el modelo con learning rate muy bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar de nuevo con fine-tuning
    print(f"\n🔧 INICIANDO FINE-TUNING ({FINE_TUNE_EPOCHS} épocas)...")
    print("-" * 50)
    
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        verbose=1
    )
    
    # Guardar el modelo
    print("\n💾 Guardando modelo...")
    model.save('flower_model.keras')
    print("✅ Modelo base entrenado y guardado como flower_model.keras")
    
    # Mostrar estadísticas finales
    print("\n📊 ESTADÍSTICAS FINALES:")
    print("-" * 30)
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"🎯 Precisión final - Train: {final_train_acc:.4f}, Val: {final_val_acc:.4f}")
    print(f"📉 Pérdida final - Train: {final_train_loss:.4f}, Val: {final_val_loss:.4f}")
    
    print("\n🎉 ¡Entrenamiento completado exitosamente!")
