# Social Media Bot Detection

## A Machine Learning and Deep Learning Approach to Detecting Inauthentic Accounts and Content on Instagram and Twitter/X

This project develops an integrated **Social Media Bot and Fake Content Detection Platform** using Machine Learning, Deep Learning, Natural Language Processing (NLP), Graph Neural Networks (GNNs), and Transformer-based models.

The system focuses on three main detection tasks:

1. Instagram Fake Account Detection
2. Twitter/X Bot Detection
3. Fake Comment / Spam Detection

The project covers data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, model serialization, and interactive Streamlit dashboards.

---

## Project Overview

Social media platforms contain increasing amounts of automated accounts, fake profiles, spam, and artificial engagement. These activities can distort genuine popularity, spread misleading information, and reduce trust in online platforms.

This project investigates different approaches for automatically identifying suspicious accounts and comments.

The system compares:

* Traditional Machine Learning
* Artificial Neural Networks
* Natural Language Processing
* Graph Neural Networks
* Transformer-based Deep Learning

The objective is not only to achieve high predictive performance, but also to investigate how different approaches perform on:

* Structured profile data
* Text data
* Graph-based data

---

## Objectives

The main objectives of this project are:

1. Detect fake or spam accounts on Instagram.
2. Detect bot accounts on Twitter/X.
3. Detect bot or spam comments using NLP.
4. Compare traditional Machine Learning and Deep Learning approaches.
5. Investigate profile-based, text-based, and graph-based signals.
6. Develop interactive dashboards for model analysis and prediction.
7. Evaluate models using multiple performance metrics.
8. Provide a reusable prototype for social-media authenticity analysis.

---

# System Components

## 1. Instagram Fake Account Detection

The Instagram component performs binary classification.

```text
Real Account = 0
Fake/Bot Account = 1
```

### Features

The model uses profile-level features including:

* Profile picture
* Username characteristics
* Full-name characteristics
* Description length
* External URL
* Private account status
* Number of posts
* Number of followers
* Number of following accounts

### Models

The following models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost
* Artificial Neural Network (ANN)

### Dataset

After duplicate removal, the cleaned dataset contains **692 Instagram profiles**.

The dataset was deliberately resampled to approximately:

* 60% Real
* 40% Fake

The original dataset contained 696 profiles before duplicate removal.

---

# 2. Twitter/X Bot Detection

The Twitter/X component classifies accounts as:

```text
Human / Authentic = 0
Bot / Automated = 1
```

The final merged dataset contains **2,465 Twitter/X accounts**:

| Class | Samples |
| ----- | ------: |
| Human |   1,394 |
| Bot   |   1,071 |

### Profile Features

The dataset contains information such as:

* Followers count
* Friends count
* Listed count
* Favourites count
* Status count
* Verified status
* Protected status
* Default profile
* Default profile image
* Username
* Name
* Description

### Detection Approaches

The Twitter/X component evaluates three main approaches.

#### Profile-Based Machine Learning

* Logistic Regression
* Random Forest
* XGBoost

#### NLP

The profile text is processed using:

```text
Profile Text
     |
     v
Text Cleaning
     |
     v
TF-IDF
     |
     v
Logistic Regression
```

#### Graph-Based Learning

A Graph Convolutional Network (GCN) is constructed using a **K-Nearest Neighbour (KNN) profile-similarity graph**.

> The graph does not represent actual Twitter/X follower or friend relationships because the dataset does not contain a real social-network edge list.

---

# 3. Fake Comment Detection

The third component performs binary text classification.

```text
Genuine = 0
Bot / Spam = 1
```

The corpus combines two sources.

### UCI YouTube Spam Collection

* 1,956 original comments
* 951 Ham
* 1,005 Spam

### UCI Sentiment Labelled Sentences

* 3,000 human-written sentences
* Used to diversify genuine examples

After preprocessing and downsampling, the final dataset contains:

**4,481 text examples**

### Class Distribution

| Class      | Samples | Percentage |
| ---------- | ------: | ---------: |
| Genuine    |   3,476 |      77.6% |
| Bot / Spam |   1,005 |      22.4% |

The dataset was split using an **85/15 stratified train-validation split**.

---

# Transformer-Based Fake Comment Detection

The fake comment detector fine-tunes:

```text
distilroberta-base
```

for binary text classification.

### Training Configuration

| Parameter             | Value                |
| --------------------- | -------------------- |
| Base Model            | `distilroberta-base` |
| Learning Rate         | `5e-6`               |
| Warm-up Ratio         | `0.1`                |
| Batch Size            | 16                   |
| Epochs                | 3                    |
| Weight Decay          | 0.01                 |
| Maximum Gradient Norm | 1.0                  |
| Model Selection       | Best Validation F1   |
| Maximum Tokens        | 128                  |

The tokenizer uses byte-level BPE with dynamic padding.

---

# Overall System Pipeline

```text
                         SOCIAL MEDIA DATA
                                |
                +---------------+---------------+
                |               |               |
                v               v               v
           Instagram        Twitter/X       Comments
                |               |               |
                v               v               v
         Data Cleaning    Data Cleaning    Text Cleaning
                |               |               |
                v               v               v
       Feature Engineering      |               |
                |               |               |
                |       +-------+-------+       |
                |       |       |       |       |
                |       v       v       v       |
                |    Profile   NLP     GNN      |
                |       |       |       |       |
                +-------+-------+-------+-------+
                                |
                                v
                         Model Evaluation
                                |
                                v
                       Model Serialization
                                |
                                v
                     Streamlit Applications
                                |
                                v
                     Prediction and Analysis
```

---

# Technologies Used

## Programming

* Python

## Machine Learning

* Scikit-learn
* XGBoost

## Deep Learning

* TensorFlow / Keras
* PyTorch

## Natural Language Processing

* Hugging Face Transformers
* TF-IDF
* DistilRoBERTa

## Graph Learning

* PyTorch
* Graph Convolutional Network (GCN)
* K-Nearest Neighbour similarity graph

## Data Processing

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn
* Plotly

## Dashboard

* Streamlit

## Model and Data Tools

* Joblib
* Hugging Face
* CSV
* JSON
* TSV

---

# Exploratory Data Analysis

Exploratory Data Analysis was performed independently for Instagram and Twitter/X.

The analysis included:

* Target class distribution
* Follower distributions
* Following/friend distributions
* Status/post counts
* Profile-picture analysis
* Correlation analysis
* Outlier analysis
* Numerical feature distributions

Twitter/X numerical variables showed strong right-skew, particularly:

* Followers
* Friends
* Statuses

Log-scale visualisation was therefore used for several EDA plots.

---

# Feature Engineering

## Instagram Features

Important engineered features include:

* Numeric username ratio
* Full-name word count
* Numeric full-name ratio
* Name/username equality
* Description length
* External URL
* Private account
* Number of posts
* Number of followers
* Number of following accounts

## Twitter/X Features

Engineered features include:

* Followers/Friends ratio
* Favourites/Statuses ratio
* Statuses/Followers ratio
* Description length
* Username length
* Full-name length
* Username digit count
* Username digit ratio
* Full-name word count
* Name/username equality

---

# Model Evaluation

The project evaluates models using multiple performance metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* Feature Importance

These metrics provide a broader evaluation than accuracy alone, particularly for classification problems with different class distributions.

---

# Results

## Instagram Results

| Model               |   Accuracy |  Precision |     Recall |   F1-score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| **XGBoost**         | **94.96%** | **1.0000** | **0.8750** | **93.33%** |
| Random Forest       |     94.24% |     1.0000 |     0.8571 |     92.31% |
| ANN                 |     87.05% |     0.8800 |     0.7857 |     83.02% |
| Logistic Regression |     87.05% |     0.8958 |     0.7679 |     82.69% |

### Best Instagram Model

**XGBoost** achieved the highest test accuracy and F1-score.

Random Forest achieved the highest ROC-AUC of approximately **0.9960**.

---

# Twitter/X Results

| Model                        | Category   |   Accuracy | Precision |  Recall |   F1-score |
| ---------------------------- | ---------- | ---------: | --------: | ------: | ---------: |
| **Random Forest**            | Profile ML | **75.86%** |    0.7654 |  0.6402 | **69.72%** |
| XGBoost                      | Profile ML |     75.25% |    0.7706 |  0.6121 |     68.23% |
| Logistic Regression          | Profile ML |     72.21% |    0.7391 |  0.5561 |     63.47% |
| GCN                          | GNN        |    72.21%* |   0.7391* | 0.5561* |    63.47%* |
| TF-IDF + Logistic Regression | NLP        |     60.04% |    0.5739 |  0.3084 |     40.12% |

### Best Twitter/X Model

**Random Forest** achieved the highest reported Twitter/X test F1-score of **69.72%**.

The TF-IDF NLP model performed substantially worse, particularly in recall.

### GCN Evaluation Note

The GCN accuracy, precision, recall, and F1 values are considered unreliable because the report identifies a likely variable-reuse issue in the evaluation cell.

The separately calculated GCN ROC-AUC was approximately **0.4857**.

Therefore, the GCN results should not be interpreted as a reliable comparison against the other Twitter/X models.

---

# Fake Comment Detection Results

| Metric              |        Result |
| ------------------- | ------------: |
| Validation Accuracy |    **98.22%** |
| Precision           |    **96.03%** |
| Recall              |    **96.03%** |
| F1-score            |    **96.03%** |
| Validation Loss     |        0.0850 |
| Probe Set           | 12/12 correct |

The fine-tuned DistilRoBERTa model achieved a validation accuracy of **98.22%** and an F1-score of **96.03%**.

The model also passed the built-in non-collapse checks and correctly classified all 12 hand-labelled probe comments.

> The fake-comment result should not be directly compared with the Instagram and Twitter/X results because it is a different classification task using a different text corpus.

---

# Best Models Summary

| Detection Task          | Best Model        |   Accuracy |   F1-score |
| ----------------------- | ----------------- | ---------: | ---------: |
| Instagram Fake Account  | **XGBoost**       | **94.96%** | **93.33%** |
| Twitter/X Bot Detection | **Random Forest** | **75.86%** | **69.72%** |
| Fake Comment Detection  | **DistilRoBERTa** | **98.22%** | **96.03%** |

---

# Streamlit Dashboard

The project includes locally hosted Streamlit dashboards for interactive analysis and prediction.

## Instagram Dashboard

The Instagram dashboard provides four main sections:

1. Executive Summary
2. Exploratory Data Analysis
3. Machine Learning Benchmarks
4. Real-Time Bot Predictor

The predictor allows users to enter Instagram profile characteristics and obtain a prediction using the trained XGBoost model.

---

## Twitter/X Dashboard

The Twitter/X dashboard provides:

1. Executive Summary
2. Exploratory Data Analysis
3. Machine Learning Benchmarks
4. Real-Time Bot Predictor

Users can enter profile information such as:

* Username
* Bio
* Followers
* Following
* Verification status
* Account metadata

The dashboard then generates a bot prediction using the trained model.

---

## Fake Comment Detection Dashboard

The NLP dashboard provides:

1. Model Overview
2. Single Comment Checker
3. Batch CSV Checker

Users can:

* Enter an individual comment
* Predict whether the comment is Genuine or Bot/Spam
* Upload a CSV file for batch classification

---

# Project Structure

```text
social-media-bot-detection/
│
├── app.py
│
├── data/
│   ├── instagram/
│   ├── twitter/
│   └── comments/
│
├── notebooks/
│   ├── Instagram_Analysis.ipynb
│   ├── Twitter_Analysis.ipynb
│   └── Fake_Comment_Detection.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── models/
│   └── visualization/
│
├── models/
│   ├── instagram/
│   ├── twitter/
│   └── fake-comment-detector/
│
├── outputs/
│   ├── EDA_Plots/
│   ├── Evaluation_Results/
│   └── Feature_Importance/
│
├── requirements.txt
│
└── README.md
```

> The exact folder and file names may differ depending on the final repository structure.

---

# Installation

## 1. Clone the Repository

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd social-media-bot-detection
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Streamlit Dashboard

From the project root directory:

```bash
streamlit run app.py
```

After starting the application, Streamlit will display a local URL in the terminal.

The default address is:

```text
http://localhost:8501
```

Open the displayed address in your web browser.

---

# Reproducing the Experiments

The general workflow for reproducing the experiments is:

```text
1. Load Dataset
        |
        v
2. Data Quality Audit
        |
        v
3. Remove Duplicates / Conflicts
        |
        v
4. Train/Test or Train/Validation Split
        |
        v
5. Preprocessing
        |
        v
6. Feature Engineering
        |
        v
7. Model Training
        |
        v
8. Model Evaluation
        |
        v
9. Save Model Artefacts
        |
        v
10. Streamlit Prediction
```

Preprocessing statistics such as scalers, vectorizers, and tokenizer-related statistics were fitted only after splitting the datasets to reduce the risk of data leakage.

---

# Training Safeguards

The Fake Comment Detection model includes several safeguards.

## Pre-training Check

Model parameters are checked for non-finite values before training.

## NaN Guard

A custom callback detects non-finite training loss and stops training.

## Post-training Check

Model weights are checked again after training.

## Probe Testing

A hand-labelled probe set is used to check whether the model has collapsed into predicting only one class.

All three safeguards passed during the reported training run.

---

# Limitations

## Dataset Size

The Instagram dataset contains only 692 cleaned profiles, which is relatively small compared with modern social-media bot networks.

## Limited Platform Coverage

The implemented system focuses on Instagram, Twitter/X, and generic social-media comments rather than platforms such as TikTok or Facebook.

## Limited Hyperparameter Tuning

The models were evaluated using single train/test or train/validation splits rather than extensive K-Fold cross-validation and hyperparameter search.

## Twitter/X Graph Limitation

The GCN uses a KNN profile-similarity graph rather than real follower/friend relationships.

## Fake Comment Dataset

The comment detector uses UCI datasets as proxy data rather than real Instagram or Twitter/X comments.

Therefore, the reported **98.22% validation accuracy** should not be interpreted as guaranteed performance on real-world platform comments.

---

# Future Improvements

Potential future improvements include:

* Real social-network follower/friend graphs
* Larger and more recent datasets
* Real Instagram/Twitter/X comments
* TikTok and Facebook detection
* K-Fold cross-validation
* Hyperparameter optimization
* More advanced GNN architectures
* More advanced transformer models
* Larger-scale real-time deployment
* Model monitoring and retraining
* Cloud or web-based deployment
* Improved multilingual social-media text detection

Important future research directions include real follower relationships, authentic comment validation, improved model serialization, cross-validation, and hyperparameter optimization.

---

# Group Members

| Member                  | Student ID     |
| ----------------------- | -------------- |
| **Tharanya Pushparaj**  | CIT-23-02-0176 |
| **Amintha Jayasooriya** | CIT-23-02-0335 |
| **Damsara Dissanayaka** | CIT-23-02-0163 |
| **Thamindu Kavinda**    | CIT-23-02-0356 |

**Module:** CCS4310 – Deep Learning
**Faculty:** Faculty of Computing
**Module Owner:** Mr. Indeera Weerasinghe

---

# Project Highlights

| Component                        | Best Model    | Accuracy | F1-score |
| -------------------------------- | ------------- | -------: | -------: |
| Instagram Fake Account Detection | XGBoost       |   94.96% |   93.33% |
| Twitter/X Bot Detection          | Random Forest |   75.86% |   69.72% |
| Fake Comment Detection           | DistilRoBERTa |   98.22% |   96.03% |

The project combines structured profile analysis, NLP, graph-based learning, and transformer-based classification within an interactive Streamlit platform.

---

# Disclaimer

This project is an academic research and prototype implementation.

The models are trained and evaluated using the datasets described in the project documentation. Performance on real-world social-media accounts or comments may differ significantly from the reported experimental results.

The system should therefore be considered a research prototype rather than a production-level social-media moderation system.
