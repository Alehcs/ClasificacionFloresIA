import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset():
    """
    Organiza el dataset de flores en carpetas de entrenamiento, validación y prueba.
    """
    # Variables de configuración
    SOURCE_DIR = 'dataset_original'
    DEST_DIR = 'data'
    CLASES = ['rosas', 'margaritas', 'girasoles']
    SPLIT_RATIO = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    
    print("🌻 Iniciando organización del dataset de flores...")
    print(f"📁 Directorio origen: {SOURCE_DIR}")
    print(f"📁 Directorio destino: {DEST_DIR}")
    print(f"🌸 Clases: {', '.join(CLASES)}")
    print(f"📊 Proporciones: Train={SPLIT_RATIO['train']}, Val={SPLIT_RATIO['val']}, Test={SPLIT_RATIO['test']}")
    print("-" * 50)
    
    # Crear estructura de carpetas de destino
    print("📂 Creando estructura de carpetas...")
    for split in ['train', 'val', 'test']:
        for clase in CLASES:
            folder_path = os.path.join(DEST_DIR, split, clase)
            os.makedirs(folder_path, exist_ok=True)
            print(f"   ✅ Creada: {folder_path}")
    
    # Diccionario para almacenar estadísticas
    stats = {}
    
    # Procesar cada clase
    for clase in CLASES:
        print(f"\n🌸 Procesando clase: {clase}")
        
        # Ruta de la clase en el directorio origen
        source_class_dir = os.path.join(SOURCE_DIR, clase)
        
        # Verificar que existe el directorio origen
        if not os.path.exists(source_class_dir):
            print(f"   ❌ Error: No se encontró el directorio {source_class_dir}")
            continue
        
        # Listar todos los archivos de imagen
        image_files = []
        for file in os.listdir(source_class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_files.append(file)
        
        print(f"   📊 Total de imágenes encontradas: {len(image_files)}")
        
        if len(image_files) == 0:
            print(f"   ⚠️  No se encontraron imágenes en {source_class_dir}")
            continue
        
        # Primera división: 80% train, 20% temp (val + test)
        train_files, temp_files = train_test_split(
            image_files, 
            test_size=0.2, 
            random_state=42
        )
        
        # Segunda división: dividir temp en val (50%) y test (50%)
        val_files, test_files = train_test_split(
            temp_files, 
            test_size=0.5, 
            random_state=42
        )
        
        print(f"   📈 División: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
        
        # Copiar archivos a sus carpetas correspondientes
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            dest_class_dir = os.path.join(DEST_DIR, split_name, clase)
            
            for file in files:
                source_path = os.path.join(source_class_dir, file)
                dest_path = os.path.join(dest_class_dir, file)
                shutil.copy2(source_path, dest_path)
            
            # Almacenar estadísticas
            if split_name not in stats:
                stats[split_name] = {}
            stats[split_name][clase] = len(files)
        
        print(f"   ✅ Archivos copiados exitosamente para {clase}")
    
    # Verificación final y resumen
    print("\n" + "="*60)
    print("📊 RESUMEN FINAL DEL DATASET ORGANIZADO")
    print("="*60)
    
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        print(f"\n📁 {split.upper()}:")
        split_total = 0
        
        for clase in CLASES:
            if split in stats and clase in stats[split]:
                count = stats[split][clase]
                print(f"   🌸 {clase}: {count} imágenes")
                split_total += count
            else:
                print(f"   🌸 {clase}: 0 imágenes")
        
        print(f"   📊 Total {split}: {split_total} imágenes")
        total_images += split_total
    
    print(f"\n🎯 TOTAL GENERAL: {total_images} imágenes")
    print("\n✅ ¡Dataset organizado exitosamente!")
    print(f"📁 Estructura creada en: {DEST_DIR}/")

if __name__ == "__main__":
    organize_dataset()
