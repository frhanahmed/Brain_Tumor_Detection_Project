import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# Step 1: Load and preprocess the data
def load_data(data_dir, img_size):
    data = []
    labels = []
    categories = os.listdir(data_dir)  # Ensure only 'tumor' and 'no_tumor'
    print(f"Categories found: {categories}")
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                # Apply CLAHE
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                img = cv2.resize(img, (img_size, img_size))
                img = preprocess_input(img)  # Important for ResNet
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image: {e}")
    return np.array(data), np.array(labels)

# Parameters
data_dir = r'C:\Users\frhan\Desktop\03_Farhan\Projects\Python\Project_College\Brain_Tumor_Datasets copy\train'
img_size = 224  # ResNet50 expects 224x224

# Load dataset
X, y = load_data(data_dir, img_size)
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
datagen.fit(X_train)

# Step 3: Load ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_size, img_size, 3)))
base_model.trainable = False  # Freeze base model

# Step 4: Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=5,
    validation_data=(X_test, y_test)
)

# Save model
#model.save('brain_tumor_model_resnet50.h5')

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Visualization
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Step 6: Prediction
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img = cv2.resize(img, (img_size, img_size))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return "Tumor" if np.argmax(prediction) == 1 else "No Tumor"

# Test prediction
image_path = r'C:\Users\frhan\Desktop\03_Farhan\Projects\Python\Project_College\Brain_Tumor_Datasets copy\test\yes\Te-gl_0018.jpg'
result = predict_image(image_path, model)
print(f"Prediction: {result}")