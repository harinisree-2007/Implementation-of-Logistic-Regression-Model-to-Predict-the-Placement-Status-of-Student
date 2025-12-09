# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Define the Problem
- Goal: Predict whether a student gets placed (Yes/No).
- Dependent variable (Y): Placement status (binary: 1 = Placed, 0 = Not Placed).
- Independent variables (X): Features such as CGPA, number of projects, internships, communication skills score, etc.

Step 2: Collect and Prepare Data
- Gather a dataset of students with:
- Features (X): academic and skill-related attributes.
- Target (Y): placement status.
- - Perform preprocessing:
- Handle missing values.
- Encode categorical variables (e.g., gender, stream).
- Normalize/scale numerical features.

Step 3: Split Dataset
- Divide data into:
- Training set (e.g., 70–80%)
- Testing set (e.g., 20–30%)

Step 4: Initialize Logistic Regression Model
- Logistic regression uses the sigmoid function:
P(Y=1|X)=\frac{1}{1+e^{-(b0+b1X1+b2X2+...+bnXn)}}
Where:
- b0 = intercept
- b1,b2,...,bn = coefficients for features

Step 5: Train the Model
- Use Maximum Likelihood Estimation (MLE) to find the best coefficients.
- Iteratively update weights using optimization methods (e.g., Gradient Descent).

Step 6: Prediction
- For a new student record:
- Compute probability P.
- If P\geq 0.5 → Predict Placed (1).
- If P<0.5 → Predict Not Placed (0).

Step 7: Evaluate Model
- Use metrics like:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve & AUC




## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
```

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
    'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
    'Internship': [0, 0, 1, 0, 1, 1, 1, 1],  # 0 = No, 1 = Yes
    'Placement': [0, 0, 0, 1, 1, 1, 1, 1]    # Target: 0 = Not Placed, 1 = Placed
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split into features and target
# ------------------------------
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']

# ------------------------------
# Step 3: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ------------------------------
# Step 4: Feature scaling
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Step 5: Create and train Logistic Regression model
# ------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Predict placement for a new student
# ------------------------------
new_student = np.array([[6, 68, 1]])  # Example: 6 hours studied, 68 prev score, Internship yes
new_student_scaled = scaler.transform(new_student)
placement_pred = model.predict(new_student_scaled)
placement_prob = model.predict_proba(new_student_scaled)

print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 else 'Not Placed'}")
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")

## Output:
the Logistic Regression Model to Predict the Placement Status of Student

![WhatsApp Image 2025-12-08 at 22 42 12_81d8807f](https://github.com/user-attachments/assets/63ba855e-ed41-4735-b0f7-d190f40d71a1)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
