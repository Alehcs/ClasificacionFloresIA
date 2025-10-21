import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Parámetros
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Nombres de clases (orden alfabético)
CLASS_NAMES = ['girasoles', 'margaritas', 'rosas']

@st.cache_resource
def load_model():
    """
    Carga el modelo entrenado con caché de Streamlit.
    
    Returns:
        tf.keras.Model: Modelo cargado
    """
    try:
        model = tf.keras.models.load_model('flower_model.keras')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def preprocess_image(image):
    """
    Preprocesa una imagen para la predicción.
    
    Args:
        image: Imagen de PIL
        
    Returns:
        np.array: Imagen procesada
    """
    # Redimensionar la imagen
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    
    # Convertir a array de NumPy
    image_array = np.array(image)
    
    # Expandir dimensiones para que tenga forma (1, 224, 224, 3)
    processed_image = np.expand_dims(image_array, axis=0)
    
    return processed_image

def main():
    """
    Función principal de la aplicación Streamlit.
    """
    # Configurar la página
    st.set_page_config(
        page_title="Clasificador de Flores",
        page_icon="🌸",
        layout="wide"
    )
    
    # Título principal
    st.title("Clasificador de Flores 🌸")
    
    # Subtítulo
    st.write("Sube una imagen de una rosa, margarita o girasol para ver la predicción.")
    
    # Crear dos columnas para el layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Subir Imagen")
        
        # Widget para subir archivos
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de flor:",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos soportados: JPG, JPEG, PNG"
        )
        
        # Mostrar información sobre las clases
        st.subheader("🌺 Clases Soportadas")
        for i, class_name in enumerate(CLASS_NAMES, 1):
            st.write(f"{i}. **{class_name.title()}**")
    
    with col2:
        st.subheader("🔍 Resultado de la Predicción")
        
        # Verificar si se subió un archivo
        if uploaded_file is not None:
            try:
                # Abrir la imagen
                image = Image.open(uploaded_file)
                
                # Mostrar la imagen subida
                st.image(
                    image, 
                    caption='Imagen Subida', 
                    use_column_width=True
                )
                
                # Mostrar spinner mientras se procesa
                with st.spinner('Clasificando...'):
                    # Cargar el modelo
                    model = load_model()
                    
                    if model is not None:
                        # Preprocesar la imagen
                        processed_image = preprocess_image(image)
                        
                        # Realizar la predicción
                        prediction = model.predict(processed_image, verbose=0)
                        
                        # Obtener la clase predicha y el score de confianza
                        predicted_class = np.argmax(prediction[0])
                        confidence = np.max(prediction[0]) * 100
                        
                        # Mostrar el resultado
                        st.success(f"**Predicción:** {CLASS_NAMES[predicted_class].title()}")
                        st.success(f"**Confianza:** {confidence:.2f}%")
                        
                        # Mostrar todas las probabilidades
                        st.subheader("📊 Probabilidades por Clase")
                        for i, class_name in enumerate(CLASS_NAMES):
                            prob = prediction[0][i] * 100
                            st.write(f"**{class_name.title()}:** {prob:.2f}%")
                            
                            # Barra de progreso para visualizar
                            st.progress(prob / 100)
                        
                        # Mostrar información adicional
                        if confidence > 80:
                            st.info("🎉 ¡Excelente! El modelo está muy seguro de esta predicción.")
                        elif confidence > 60:
                            st.warning("⚠️ El modelo está moderadamente seguro de esta predicción.")
                        else:
                            st.error("❓ El modelo no está muy seguro de esta predicción. Intenta con otra imagen.")
                    
            except Exception as e:
                st.error(f"Error al procesar la imagen: {e}")
                st.write("Por favor, asegúrate de que la imagen sea válida y esté en uno de los formatos soportados.")
        
        else:
            # Mostrar mensaje cuando no hay imagen
            st.info("👆 Por favor, sube una imagen usando el widget de arriba.")
            
            # Mostrar ejemplo de uso
            st.subheader("💡 Cómo usar la aplicación")
            st.write("""
            1. **Sube una imagen** de una rosa, margarita o girasol
            2. **Espera** a que el modelo procese la imagen
            3. **Ve el resultado** con la predicción y nivel de confianza
            4. **Explora** las probabilidades de cada clase
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Clasificador de Flores** - Desarrollado con TensorFlow y Streamlit | "
        "Modelo entrenado con Transfer Learning usando MobileNetV2"
    )

if __name__ == "__main__":
    main()
