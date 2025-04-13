# CNN Transfer Learning Experiments: Stanford Dogs → Cats vs Dogs

This project explores **transfer learning** using Convolutional Neural Networks (CNNs) trained on the **Stanford Dogs dataset**, then adapted and evaluated on the **Cats vs Dogs dataset**.

The experiments demonstrate how reusing different parts of a trained model can impact performance when fine-tuning for a similar task.

---


---

## 🧪 Experiments Overview

### 🔹 **Experiment 1** – Train and Save Model
- **Files:** `test_exp1_1.py`, `test_exp1_2.py`
- Trains a CNN from scratch on the **Cat and Dog set**
- Trains a CNN from scratch on the **Stanford Dogs dataset**
- Uses a learning rate of **0.0001**
- Saves the model to `model_saved/`

---

### 🔹 **Experiment 2** – Replace Output Layer
- **File:** `test_exp2.py`
- Loads the trained model from Exp 1
- Replaces only the **output layer** to classify **Cats vs Dogs**
- Freezes all other layers
- Trains the new model for **50 epochs**

---

### 🔹 **Experiment 3** – Replace Output + First 2 Conv Layers
- **File:** `test_exp3.py`
- Loads model from Exp 1
- Replaces:
  - Output layer
  - First **two convolutional layers**
- Keeps remaining layers frozen
- Trains for **50 epochs**

---

### 🔹 **Experiment 4** – Replace Output + Last 2 Conv Layers
- **File:** `test_exp4.py`
- Loads model from Exp 1
- Replaces:
  - Output layer
  - Last **two convolutional layers**
- Keeps remaining layers frozen
- Trains for **50 epochs**

---

## ⚙️ Requirements

- Python 3.7+
- TensorFlow 2.x
- `tensorflow_datasets`
- `numpy`, `os`, etc.

Install with:
```bash
pip install tensorflow tensorflow_datasets numpy


