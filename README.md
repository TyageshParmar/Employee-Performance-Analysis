# Employee-Performance-Analysis
# Employee Performance Prediction

## 📌 **Project Overview**
This project aims to predict employee performance ratings at **INX Future Inc.** using **machine learning and deep learning models**. The dataset includes various features such as **work experience, job satisfaction, training history, and work-life balance** to analyze and improve employee performance.

## 📂 **Dataset Information**
- **Source:** INX Future Inc Employee Performance Dataset
- **Features:** 27 columns (Employee Demographics, Job Role, Work Environment, etc.)
- **Target Variable:** `PerformanceRating` (Categorical: 2, 3, or 4)
- **Challenges:** The dataset is **imbalanced**, requiring techniques like **SMOTE** and **class weighting**.

## 🚀 **Technologies Used**
- **Programming Language:** Python
- **Libraries & Tools:** Pandas, NumPy, Scikit-Learn, TensorFlow, Keras, Matplotlib, Seaborn
- **Machine Learning Models:** Logistic Regression, Random Forest, SVM, Naïve Bayes, Gradient Boosting, XGBoost and many more...
- **Deep Learning:** Artificial Neural Network (ANN) with Hyperparameter Tuning
- **Hyperparameter Tuning:** GridSearchCV, RandomizedSearchCV

## 🏆 **Best Performing Model**
| Model | HPT Training Accuracy | HPT Testing Accuracy | Overfitting Risk |
|--------|----------------------|----------------------|------------------|
| **Gradient Boosting** | **100.00%** | **95.73%** | ✅ Low (Minimal overfitting) |
| **Ensemble Model** | 99.59% | 96.49% | ✅ Low (Balanced performance) |
| **Random Forest** | 100.00% | 96.18% | ✅ Low (Minimal overfitting) |
| **Support Vector Machine (SVM)** | 99.54% | 94.66% | ✅ Low (Good improvement after tuning) |
| **Ada Boosting** | 91.04% | 91.00% | ✅ Low (Stable model) |
|	**Xtreme Gradient Boosting** |	99.69% |	96.18% | ✅ Low (Minimal overfitting) |
|	**Bagging Algorithm** |	99.33% |	94.96% | ✅ Low (Minimal overfitting) |
|	**Neural Network** |	98.21% |	91.46% | ✅ Low (Minimal overfitting) |

## ⚠️ **Challenges Faced**
### **1️⃣ Data Quality and Preparation**
✔ Handling missing values and categorical encoding
✔ Addressing class imbalance with SMOTE & class weighting
✔ Feature selection using correlation analysis and mutual information

### **2️⃣ Model Selection and Hyperparameter Tuning**
✔ Avoiding overfitting in complex models (ANN, Decision Tree)
✔ Reducing computational costs of ensemble methods
✔ Finding the right balance between accuracy and generalization

### **3️⃣ Performance Evaluation & Overfitting Prevention**
✔ Used **confusion matrix, classification report, and accuracy curves**
✔ Applied **dropout, early stopping, and L2 regularization** for deep learning
✔ Compared models based on **generalization ability** (train vs. test accuracy)

## 🔧 **Installation & Setup**
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/employee-performance-prediction.git
   cd employee-performance-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the project:
   ```sh
   python main.py
   ```

## 📊 **Results & Insights**
✔ **Job satisfaction & work-life balance strongly influence performance**
✔ **Recent promotions & salary hikes impact retention & productivity**
✔ **Overtime & training frequency correlate with higher performance ratings**
✔ **Feature engineering & balancing techniques significantly improve model accuracy**
✔ **Neural Networks showed potential but required extensive tuning to generalize well**

## 📌 **Future Enhancements**
🔹 Implement **Explainable AI (XAI)** to interpret model decisions
🔹 Deploy the model using **Flask/Django API** for real-time predictions
🔹 Expand dataset with **employee feedback & additional performance metrics**
🔹 Further optimize deep learning models with **transfer learning & advanced architectures**

## 👨‍💻 **Contributors**
- [Tyagesh Parmar]([https://github.com/your-username](https://github.com/TyageshParmar/Employee-Performance-Analysis))
