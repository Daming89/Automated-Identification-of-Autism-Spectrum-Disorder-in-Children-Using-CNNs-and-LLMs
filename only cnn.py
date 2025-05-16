import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns

# test image path
test_directory = r"C:\Users\liangming.jiang21\FYP\dataset01\test"

# Load the model
keras_fname = 'VGG19_augmented_best_model'
# keras_fname = 'VGG19_autism_ep5'
saved_model = r"C:\Users\liangming.jiang21\FYP\VGG19_augmented_best_model.keras"
model = tf.keras.models.load_model(saved_model)
input_size = model.input_shape[1:3] # check the input size from the loaded model

# Summarize model.
model.summary()


def load_images_from_directory(directory, target_size):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])  # Sort the classes
    label_map = {name: index for index, name in enumerate(class_names)}

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(class_dir, file)
                img = load_img(file_path, target_size=target_size, color_mode='rgb')
                img = img_to_array(img)
                img = np.expand_dims(img, axis=0)
                images.append(img)
                labels.append(label_map[class_name])

    images = np.vstack(images)  # Stack images into a single numpy array
    labels = np.array(labels)

    # Shuffle images and labels in unison
    images, labels = shuffle(images, labels, random_state=42)

    return images, labels, class_names

# Load images
images, true_labels, class_names = load_images_from_directory(test_directory, target_size=input_size)  # Change target size as per your model's input layer

# predict test images
predictions = model.predict(images)
predicted_labels = np.argmax(predictions, axis=1)

# analyze the predicted result
conf_matrix = confusion_matrix(true_labels, predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, target_names=class_names)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()