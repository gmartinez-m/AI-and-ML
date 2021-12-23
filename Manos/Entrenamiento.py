from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

datos_entrenamiento = "C:/Users/gonma/PycharmProjects/Manos/Fotos/Entrenamiento"
datos_validacion = "C:/Users/gonma/PycharmProjects/Manos/Fotos/Validacion"

# Parámetros
iteraciones = 20  # Número de iteraciones para ajustar nuestro modelo.
altura, longitud = 200, 200  # Tamaño de las imágenes de entrenamiento.
batch_size = 1  # Número de imágenes que vamos a procesar secuencialmente.
pasos_entrenamiento = 300 / 1  # Número de veces que se va a procesar la información en cada iteración. (300 imágenes)
pasos_validacion = 200 / 1  # Después de cada iteración, validamos lo anterior. (Número items para validar)
filtrosconv1 = 32
filtrosconv2 = 64  # Número de filtros que vamos a aplicar en cada convolución.
tam_filtro1 = (3, 3)
tam_filtro2 = (2, 2)  # Tamaño de los filtros 1 y 2.
tam_pool = (2, 2)  # Tamaño del filtro en max pooling.
clases = 2  # Mano izquierda y mano derecha.
lr = 0.0005  # Ajustes de la red neuronal para acercarse a una solución óptima.

# Preprocesamiento de las imágenes
preprocesamiento_entre = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True
)

preprocesamiento_vali = ImageDataGenerator(
    rescale = 1./255
)

imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
)


# Creación de la red neuronal convolucional (CNN)
cnn = Sequential()

# Se agregan filtros con el fin de volver nuestra imagen muy profunda pero pequeña
cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))

cnn.add(Dense(clases, activation='softmax'))

# Se agregan parámetros para optimizar el modelo
cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Entrenamos la red
cnn.fit(imagen_entreno, steps_per_epoch=pasos_entrenamiento, epochs=iteraciones, validation_data=imagen_validacion, validation_steps=pasos_validacion)

# Guardo el modelo
cnn.save('ModeloManos.h5')
cnn.save_weights('PesosManos.h5')