import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

model = MobileNetV2(weights='imagenet')

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

def classify_image(img_path):
    """Classify the image and return the result."""
    img, img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_classes = decode_predictions(predictions, top=3)[0]
    
    results = []
    for (_, label, score) in predicted_classes:
        results.append(f"Label: {label}, Score: {score:.4f}")
    return results, img

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        results, img = classify_image(file_path)
        result_text = "\n".join(results)

        # Clear previous image and result
        for widget in frame.winfo_children():
            widget.destroy()

        # Display the image
        img_display = Image.open(file_path)
        img_display.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img_display)
        img_label = tk.Label(frame, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack(pady=10)

        # Display the results
        result_label = tk.Label(frame, text="Predictions:")
        result_label.pack()
        result_text_label = tk.Label(frame, text=result_text, justify=tk.LEFT)
        result_text_label.pack()

def main():
    if len(sys.argv) > 1:
        # Command-line mode
        image_path = sys.argv[1]
        if not image_path:
            print("Usage: python image_classifier.py <image_path>")
            sys.exit(1)
        results, _ = classify_image(image_path)
        for result in results:
            print(result)
    else:
        # GUI mode
        global frame
        root = tk.Tk()
        root.title("Image Classifier")

        frame = tk.Frame(root)
        frame.pack(pady=20)

        tk.Button(root, text="Open Image", command=open_file).pack(pady=10)

        root.mainloop()

if __name__ == '__main__':
    main()
