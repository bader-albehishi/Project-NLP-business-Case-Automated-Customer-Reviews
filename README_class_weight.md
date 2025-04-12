
# 📊 Sentiment Classification Using RoBERTa with Class Weights

## 📖 Overview
This project implements a sentiment classification model using the `roberta-base` transformer from Hugging Face. The model is fine-tuned on a product reviews dataset and addresses class imbalance using computed class weights. It classifies reviews into three categories: **positive**, **neutral**, and **negative**.

## 🧠 Model Summary
- **Base Model:** `roberta-base`
- **Task:** 3-class sentiment classification
- **Approach:** Class imbalance is addressed by computing weights for each class and applying them to the classifier.
- **Tokenizer:** `RobertaTokenizerFast`
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score

## 📁 Files
| File Name                        | Description                                      |
|----------------------------------|--------------------------------------------------|
| `classfiction class weight.ipynb` | Main notebook for training and evaluation using class weighting |
| `train.csv`                      | Preprocessed training data                      |
| `val.csv`                        | Validation data used for model evaluation       |

## ⚙️ Requirements

Install the required libraries:

```bash
pip install transformers
pip install pandas scikit-learn torch
```

You also need Python 3.7+ and PyTorch (with GPU support recommended for speed).

## 🚀 How to Run

1. Open the notebook `classfiction class weight.ipynb` in Jupyter or Google Colab.
2. Upload your `train.csv` and `val.csv` files.
3. Run all cells sequentially:
   - Data is loaded and labels are encoded.
   - Class weights are calculated and applied.
   - The RoBERTa model is initialized and fine-tuned.
   - The model is evaluated on the validation set.

## 📈 Evaluation

After training, the model's performance is assessed using:

- **Classification Report** – Precision, Recall, F1-score per class.
- **Confusion Matrix** – To visualize misclassifications.

## ⚠️ Limitations

- The default loss function (`CrossEntropyLoss`) is used without explicitly applying class weights. While class weights were calculated, they were not integrated directly into the loss, which may affect learning for minority classes.
- The neutral class continues to show lower performance, indicating that further refinement (e.g., custom weighted loss or data augmentation) may be needed.

## 📝 License

This project is provided for educational and research purposes.
