from collections import OrderedDict
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.experimental.preprocessing import Resizing

img_height = 180
img_width = 180

current_directory = os.path.join(os.path.dirname(__file__))
json_path = os.path.join(current_directory, "panoramas.json")
with open(json_path, 'r') as f:
  data = json.load(f)

def list_to_ordered_set(input_list):
    ordered_dict = OrderedDict.fromkeys(input_list)
    ordered_set = list(ordered_dict.keys())
    return ordered_set

def load_and_preprocess_image(data, degrees_increment=30):
    images = []
    labels = []
    label = data['label']
    url = data['url']
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = image.resize((180, 180))  # Optional: Resize the image to a desired size
    image_array = np.array(image)
    # Determine the number of segments based on the degrees increment
    segments = int(360 / degrees_increment)

    # Calculate the width of each segment
    segment_width = int(image_array.shape[1] / segments)

    # Split the panoramic image into segments and save them
    for i in range(segments):
        start_col = i * segment_width
        end_col = (i + 1) * segment_width
        segment_image_array = image_array[:, start_col:end_col, :]
        images.append(segment_image_array)
        labels.append(label)
    images = [tf.image.resize(segment, [img_height, img_width]) for segment in images]
    return images, labels

def train():
    images = []
    labels = []
    for d in data:
        label = d['label']
        url = d['url']
        image_list, label_list = load_and_preprocess_image(d)
        images.extend(image_list) 
        labels.extend(label_list) 
        
    label_set = list_to_ordered_set(labels)
    print(label_set)
    label_to_index = {label: idx for idx, label in enumerate(label_set)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    labels = [label_to_index[label] for label in labels]

    images = tf.convert_to_tensor(images, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    train_size = int(0.8 * len(data))
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    batch_size = 10  # Adjust this as needed

    train_dataset = train_dataset.shuffle(buffer_size=train_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    num_classes = len(label_set)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Resizing(img_height, img_width, interpolation="bilinear"),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(
        train_dataset,
        validation_data=test_dataset.batch(batch_size),  # Batch the test dataset
        epochs=5
    )

    test_dataset = test_dataset.batch(batch_size)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("Test accuracy:", test_accuracy)

    return model


### TESTING ###
model_dir = os.path.join(current_directory, "saved_model")
if not os.path.exists(model_dir):
  model = train()
  model.save('saved_model')
else:
  model = tf.keras.models.load_model('saved_model')

label_list = ['living room', 'balcony', 'bathroom', 'bedroom', 'entrance', 'kitchen', 'art gallery', 'Uncategorized', 'kids room']
def test(path):
    img = tf.keras.utils.load_img(
    path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(label_list[np.argmax(score)], 100 * np.max(score))
    )

path = os.path.join(current_directory, "bedroom_test.jpeg")
test(path)

path = os.path.join(current_directory, "bathroom_test.jpeg")
test(path)

def predict_category(image):
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_label = label_list[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return predicted_label, confidence