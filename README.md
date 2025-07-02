# codealpha_tasks2
# ✍️ Handwritten Character Recognition using CNN

This project uses a **Convolutional Neural Network (CNN)** to recognize **handwritten English letters (A-Z)** from the **EMNIST Letters** dataset.

---

## 🎯 Objective

Build a deep learning model that can classify **handwritten alphabets** from grayscale images using **image processing and CNNs**.

---

## 📂 Dataset

We use the [**EMNIST Letters**](https://www.nist.gov/itl/products-and-services/emnist-dataset) dataset via `tensorflow_datasets`. It contains:
- 145,600 training images
- 26 balanced classes (letters A–Z, lowercase labeled as 1–26)

---

## 🛠️ Tools & Libraries

- Python 3.x
- TensorFlow 2.x
- TensorFlow Datasets
- Matplotlib

Install dependencies:

bash
pip install tensorflow tensorflow-datasets matplotlib
Model Architecture
Input: 28x28 grayscale image

2 Convolutional Layers with MaxPooling

Flatten + Dense Layer

Dropout for regularization

Output Layer: 26 neurons with softmax activation (one per letter)

🏃 How to Run
▶️ Training
bash
python your_script_name.py

 Evaluation
Final test accuracy ~85-90% (may vary with epochs)

Accuracy vs Epochs graph will be shown after training
Results
Metric	Value
Accuracy	~88%
Loss	~0.4
Model Type	CNN
Epochs	10

 Accuracy Plot
<p align="center"> <img src="https://user-images.githubusercontent.com/your-username/your-plot.png" alt="Accuracy Plot" width="500"/> </p>

Future Enhancements
Use EMNIST-ByClass for digits + letters

Build CRNN (CNN + RNN) for word-level recognition

Train with your own handwritten data (using OpenCV)

📚 References
EMNIST Dataset

TensorFlow Datasets

CNN in TensorFlow

🧑‍💻 Author
Ritu Raj – GitHub

📜 License
This project is licensed under the MIT License.
