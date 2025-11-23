import tkinter as tk
from tkinter import ttk
import numpy as np
import pickle

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Create a function for Prediction
def diabetes_prediction(input_data):
    # Change the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def on_button_click():
    input_data = [variables[label.lower().replace(" ", "_")].get() for label in labels]
    diagnosis.set(diabetes_prediction(input_data))

# Create the main window
root = tk.Tk()
root.title('Isaac Diabetes prediction')

variables = {}

labels = ['Number of Pregnancies', 'Glucose Level', 'Blood Pressure value',
          'Skin Thickness value', 'Insulin Level', 'BMI value',
          'Diabetes Pedigree Function value', 'Age of the Person']

for i, label_text in enumerate(labels):
    # Check if label_text is not None before processing
    if label_text is not None:
        label = ttk.Label(root, text=label_text)
        label.grid(row=i, column=0, padx=10, pady=10)

        var = tk.StringVar()
        entry = ttk.Entry(root, textvariable=var)
        entry.grid(row=i, column=1, padx=10, pady=10)
        variables[label_text.lower().replace(" ", "_")] = var

# Create result label
diagnosis = tk.StringVar()
result_label = ttk.Label(root, textvariable=diagnosis)
result_label.grid(row=len(labels), columnspan=2, pady=20)

# Create the prediction button
predict_button = ttk.Button(root, text='Diabetes Test Result', command=on_button_click)
predict_button.grid(row=len(labels) + 1, columnspan=2, pady=10)

# Start the Tkinter event loop
root.mainloop()
