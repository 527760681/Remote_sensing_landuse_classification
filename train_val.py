import keras
from PIL import Image
from keras import  optimizers
from keras.preprocessing.image import ImageDataGenerator

import utils
import numpy as np
import vgg_keras

classes = 4
image_size = 128
log_filepath = '.\log'
csv_path = r'class.txt'
csv_test_path = r'.\testRegion\class.txt'

def train():  # finetuning
    # input tensor
    X_train, X_test, y_train, y_test = utils.load_data(csv_path=csv_path,num_classes=classes)

    # Data Augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # load model
    model = vgg_keras.vgg()

    # Compilation
    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    tb_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
                                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    # Train
    # model.fit(X_train,y_train,epochs=50,batch_size=32)

    # Fits the model on batches with real-time data augmentation
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        steps_per_epoch=len(X_train) / 32, epochs=500, callbacks=[tb_cb])
    # Evaluate
    score = model.evaluate(X_test, y_test, batch_size=32)
    print(score)

    # Save Model
    model.save('vgg16.h5')
    from keras.utils import plot_model
    plot_model(model, to_file=r'model.png')


def val(model_path):  # val
    model = vgg_keras.vgg()

    model.load_weights(model_path)

    image_path = r'b.png'

    # Load img and resize
    img = Image.open(image_path)
    img = img.resize((128, 128))
    x = np.array(img)
    # keep the shape same as domel.output (256,256,3,1)
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    print(preds)
    print(np.argmax(preds))

def model_test(model_path):
    # input tensor
    X_test, y_test = utils.load_test_data(csv_path=csv_test_path, num_classes=classes)

    # Data Augmentation
    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # horizontal_flip=True
    )
    datagen.fit(X_test)

    # load model
    model = vgg_keras.vgg()

    #load weights
    model.load_weights(model_path)

    # Compilation
    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    tb_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
                                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    # Evaluate
    score = model.evaluate(X_test, y_test, batch_size=32)
    print(score)


# val('vgg16.h5')
# train()
model_test('vgg16.h5')
