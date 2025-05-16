import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19 
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import shutil
from tqdm import tqdm

# Data Path
train_dir = r'C:\Users\liangming.jiang21\FYP\dataset01\train'
valid_dir = r'C:\Users\liangming.jiang21\FYP\dataset01\valid'
test_dir = r'C:\Users\liangming.jiang21\FYP\dataset01\test'
augmented_dir = r'C:\Users\liangming.jiang21\FYP\dataset01\augmented_dataset'

# Data Augmentation
def augment_dataset(source_dir, target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    for class_name in os.listdir(source_dir):
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        for img_name in tqdm(os.listdir(source_class_dir), desc=f"Augmenting {class_name}"):
            img_path = os.path.join(source_class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            i = 0
            for batch in data_gen.flow(img_array, batch_size=1, 
                                       save_to_dir=target_class_dir, 
                                       save_format="jpg"):
                i += 1
                if i >= 5:  # Generate 5 enhanced images for each image
                    break

augment_dataset(train_dir, augmented_dir)

# data preprocessing
def preprocess_data(data_dir, batch_size=16):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=(224, 224), batch_size=batch_size, label_mode='categorical', shuffle=True
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)  
    return img_array

# loading data
batch_size = 32  
train_data = preprocess_data(augmented_dir, batch_size=batch_size)
val_data = preprocess_data(valid_dir, batch_size=batch_size)
test_data = preprocess_data(test_dir, batch_size=batch_size)

class_names = [item.name for item in os.scandir(augmented_dir) if item.is_dir()]
num_classes = len(class_names)

# create model
def create_model():
    input_shape = (224, 224, 3)
    base_model = VGG19(include_top=False, weights=None, input_shape=input_shape, pooling='max') 

    x = base_model.output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=AdamW(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# learning rate adjustment
def scheduler(epoch, lr):
    return lr * 0.95

# Train and evaluate
def train_and_evaluate_model(model, train_data, val_data, test_data, model_name, num_epoch):
    checkpoint_path = f"/kaggle/working/{model_name}_augmented_best_model.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
    lr_scheduler = LearningRateScheduler(scheduler)

    print("Training the model...")
    history = model.fit(train_data, epochs=num_epoch, validation_data=val_data, callbacks=[checkpoint, lr_scheduler])

    final_model_path = f"/kaggle/working/{model_name}_augmented_final_model.keras"
    model.save(final_model_path)
    print(f"Final model saved at: {final_model_path}")

    print("Evaluating on the test set...")
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {test_accuracy}")

    y_true, y_pred = [], []
    for images, labels in test_data:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='crest', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.show()

    return checkpoint_path

# model training
model = create_model()
checkpoint_path = train_and_evaluate_model(model, train_data, val_data, test_data, "VGG19", num_epoch=20)

