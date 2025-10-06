import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Model banana
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# GUI banana
root = tk.Tk()
root.title("Draw Digit")
canvas = tk.Canvas(root, width=280, height=280, bg="white")
canvas.pack()

image = Image.new("L", (280, 280), color=255)
draw = ImageDraw.Draw(image)

def paint(event):
    x1 = event.x - 10
    y1 = event.y - 10
    x2 = event.x + 10
    y2 = event.y + 10
    canvas.create_oval(x1, y1, x2, y2, fill="black")
    draw.ellipse([x1, y1, x2, y2], fill=0)

def predict():
    img = image.resize((28, 28))
    img = ImageOps.invert(img)
    data = np.array(img)
    data = data / 255
    data = data.reshape(1, 28, 28)
    result = model.predict(data)
    digit = np.argmax(result)
    label.config(text=f"Prediction: {digit}")

def clear():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill=255)
    label.config(text="Draw and Predict")

canvas.bind("<B1-Motion>", paint)

btn1 = tk.Button(root, text="Predict", command=predict)
btn1.pack()

btn2 = tk.Button(root, text="Clear", command=clear)
btn2.pack()

label = tk.Label(root, text="Draw and Predict")
label.pack()

root.mainloop()# Hand-Digit-Recognization-using-python-
