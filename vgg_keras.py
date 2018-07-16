from keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout

classes = 4
image_size=128

def vgg(): #finetuning
    # load weight without 3 FC
    base_model = VGG16(include_top=False, weights='imagenet',classes=classes,input_shape=(image_size, image_size, 3))
    model=base_model.output

    #retrain 3 fc layers
    model = Flatten(name='flatten')(model)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dropout(rate=0.5)(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(rate=0.5)(model)
    predictions = Dense(classes, activation='softmax', name='predictions')(model)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model