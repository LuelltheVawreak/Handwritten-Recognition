import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical, load_model
from tensorflow.keras.models import Sequential, 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
import gradio as gr

#Load the data: 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

#Check 
print("=== MNIST Dataset Info (simulated header) ===")
print(f"Number of Training Images: {x_train.shape[0]}")
print(f"Number of Test Images: {x_test.shape[0]}")
print(f"Image Dimensions: {x_train.shape[1]} x {x_train.shape[2]}")
print(f"Total Pixels per Image: {x_train.shape[1] * x_train.shape[2]}")
print("=============================================")

# Show some example of the MNIST dataset
plt.figure(figsize=(10,4))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(x_train[i].reshape(28,28), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
#
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Build CNN model
print('Building CNN model...')
model = Sequential([
    Conv2D(20, (5,5), padding="same", activation="relu", input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2), strides=(2,2)),Dropout(0.25),

    Conv2D(50, (5,5), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2,2), strides=(2,2)),Dropout(0.25),

    Conv2D(100, (5,5), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    Dropout(0.25),


    Flatten(),
    Dense(500, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

#Compile model 
opt = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#Train
print('Training the model...')
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Accuracy: {acc*100:.2f}%")

model.save("mnist_model.h5")
print("Model saved to mnist_model.h5")

'''history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.1)'''

#Kiểm tra overfitting
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Validation Acc')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Cacullate confusion matrix

y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Gradio interface

# Load pretrain model
model = load_model("mnist_model.h5")

# input and predict
def predict_digit(img):
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    pred = model.predict(img)
    return {str(i): float(pred[0][i]) for i in range(10)}

# create intereface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(width=200, height=200, image_mode='RGB', label=" Vẽ hoặc tải ảnh số viết tay"),
    outputs=gr.Label(num_top_classes=3, label="Kết quả dự đoán"),
    live=False,
    title="MNIST Digit Recognizer",
    description="Vẽ hoặc upload ảnh số viết tay (0–9) để mô hình dự đoán."
)

demo.launch()
