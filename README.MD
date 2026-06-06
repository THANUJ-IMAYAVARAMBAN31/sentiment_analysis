# IMDb Sentiment Analysis: Neural Networks vs Traditional Machine Learning

## Project Overview

This project explores sentiment analysis on the IMDb Movie Reviews dataset by comparing multiple machine learning and deep learning approaches.

The goal was not only to build a sentiment classifier but also to understand how different feature extraction techniques and model architectures affect performance and generalization.

The dataset contains 50,000 movie reviews labeled as either:

* Positive (1)
* Negative (0)

---

## Models Compared

### Model 1

**CountVectorizer + Linear Neural Network**

Architecture:

```text
1000 → 1
```

Accuracy: **~86.5%**

---

### Model 2

**CountVectorizer + Non-Linear Neural Network**

Architecture:

```text
1000 → 128 → ReLU → Dropout → 1
```

Accuracy: **~85-86%**

---

### Model 3

**TF-IDF + Linear Neural Network**

Architecture:

```text
1000 → 1
```

Accuracy: **~86.5%**

---

### Model 4

**TF-IDF + Non-Linear Neural Network**

Architecture:

```text
1000 → 128 → ReLU → Dropout → 1
```

Accuracy: **~85-86%**

---

### Model 5 (Best Model)

**TF-IDF + Logistic Regression**

Accuracy: **86.57%**

This model achieved the best balance between performance, stability, and generalization.

---

## Results Summary

| Model   | Feature Extraction | Architecture        | Accuracy   |
| ------- | ------------------ | ------------------- | ---------- |
| Model 1 | CountVectorizer    | Linear NN           | ~86.5%     |
| Model 2 | CountVectorizer    | Non-Linear NN       | ~85-86%    |
| Model 3 | TF-IDF             | Linear NN           | ~86.5%     |
| Model 4 | TF-IDF             | Non-Linear NN       | ~85-86%    |
| Model 5 | TF-IDF             | Logistic Regression | **86.57%** |

---

## Why Did the Neural Networks Overfit?

One of the most interesting findings from this project was that the neural networks frequently overfit despite having higher capacity.

### Reason 1: The Dataset Is Almost Linearly Separable

After applying CountVectorizer or TF-IDF, each review becomes a numerical feature vector.

Words such as:

```text
excellent
amazing
great
```

strongly indicate positive sentiment, while words such as:

```text
terrible
awful
boring
```

strongly indicate negative sentiment.

Because of this, the transformed feature space becomes highly linearly separable.

A simple linear decision boundary is already sufficient to classify most reviews correctly.

This is one of the reasons Logistic Regression has remained a strong baseline for sentiment analysis tasks for many years.

---

### Reason 2: The Best Performance Appears Very Early

During training, test accuracy reached its peak within the first few epochs.

After that:

* Training accuracy continued increasing
* Training loss continued decreasing
* Test accuracy stopped improving
* Test loss began increasing

This is a classic sign of overfitting.

The models started memorizing training reviews rather than learning patterns that generalized to unseen reviews.

---

### Reason 3: Neural Networks Were Unnecessarily Complex

The non-linear models attempted to learn complex relationships:

```text
1000 → 128 → ReLU → Dropout → 1
```

However, the dataset did not require such complexity.

Since sentiment information was already captured effectively by TF-IDF and CountVectorizer, additional layers provided little benefit and mainly increased the risk of memorization.

---

## Conclusion

Although neural networks are often considered more powerful, this experiment demonstrates that simpler models can outperform more complex architectures when the feature representation is already strong.

For the IMDb sentiment analysis task:

* TF-IDF produced better feature representations than raw word counts.
* Logistic Regression provided the most stable performance.
* Neural networks frequently overfit without improving test accuracy.
* Simpler models generalized better than deeper architectures.

The final deployed model uses:

```text
TF-IDF + Logistic Regression
```

which achieved the best overall results in this study.

---

## Live Demo

Live Application:

```text
[Add Render Deployment Link Here]
```

---

## Future Improvements

Possible future improvements include:

* Word Embeddings
* LSTM Networks
* GRU Networks
* BERT
* DistilBERT
* Transformer-based Sentiment Analysis

These approaches can learn semantic relationships between words instead of relying solely on bag-of-words representations.
