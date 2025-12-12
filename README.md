# 🚢 Titanic Survival Prediction: From Baseline to Ensemble

**A data science journey achieving top-tier accuracy (~87% Training / ~78%+ Leaderboard) by uncovering hidden social groups.**

## 📖 The Story

In data science, the "final model" is often the least interesting part. The real value lies in the **evolution of thought**—understanding *why* early attempts failed and how specific insights led to the solution.

This project documents my journey from a naive baseline to a sophisticated Voting Ensemble. It wasn't a straight line to success; it was an iterative process of hypothesis, failure, investigation, and refinement.

---

## 📉 Chapter 1: The "Naive" Baseline
**Hypothesis:** Basic demographic features like Age, Sex, and Class are sufficient to predict survival.

* **Approach:** I started with standard cleaning (median imputation) and fed the raw data into a Logistic Regression model.
* **Result:** The model hit a "glass ceiling" with an accuracy of around **79-80%**. It treated every passenger as an isolated individual, missing the complex social dynamics of the tragedy.

## 🔍 Chapter 2: The Detective Work (Feature Engineering)
**Insight:** Survival on the Titanic wasn't just individual luck—it was a group effort. Families lived or died together.

To capture this signal, I engineered three critical features:
1.  **`Title`**: Extracted from passenger names (Mr, Mrs, Master, Dr, etc.) to capture social status and age nuances better than the raw "Age" column.
2.  **`Ticket_Freq`**: A count of how many passengers shared the exact same ticket number. This revealed groups of friends or extended families traveling together who weren't captured by the "SibSp" (Siblings/Spouse) or "Parch" (Parents/Children) columns.
3.  **The "Magic Loop" (`Family_Survival`)**:
    * I grouped passengers by their **Last Name** and **Fare**.
    * I searched the training data to see if *any* member of that specific group survived.
    * If one lived, I marked the whole group as "Likely to Survive." If one died, I marked them as "Likely to Die."
    * *This single feature drastically boosted the model's predictive power.*

## 🛠 Chapter 3: Advanced Preprocessing
With valuable features in hand, I prepared the data for tree-based models:
* **Smart Imputation:** Filled missing ages based on the median age of their specific `Title` group (e.g., a "Master" is much younger than a "Mr").
* **Binning:** Converted continuous variables (`Age`, `Fare`) into quantiles/bins to reduce noise and prevent overfitting.
* **Encoding:** Transformed all categorical text data into numerical format for the algorithms.

## 🤝 Chapter 4: The Final Ensemble
**Strategy:** Instead of relying on a single algorithm, I combined the strengths of three diverse models using a **Voting Classifier**:

1.  **Logistic Regression:** For stability and linear baselines.
2.  **Random Forest (Tuned):** For handling non-linear interactions and hierarchies.
3.  **Gradient Boosting (XGBoost):** For precision and correcting errors made by the other models.

**The Solution:** A **Soft Voting Ensemble** that averages the probabilities of all three models, resulting in a stable, high-accuracy prediction that generalizes well to unseen data.

---

## 📊 Visualizations
The project includes visualizations to verify the feature engineering before training:
* **Survival by Family_Survival:** Confirmed a massive separation in survival rates (0 vs 1).
* **Survival by Ticket Frequency:** Revealed that small-to-medium groups had better survival odds than solo travelers or massive groups.
* **Correlation Heatmaps:** To identify multicollinearity and strong predictors.

---

## 🚀 Key Results
* **Baseline Score:** ~78-79%
* **Final Ensemble Training Score:** ~87%
* **Kaggle Leaderboard Score:** **0.787+** (Top Tier)

---

## 💻 How to Run
1.  Clone this repository.
2.  Install the required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
    ```
3.  Open `Titanic_Survival_Ensemble.ipynb` in Jupyter Notebook or VS Code.
4.  Run the cells sequentially to see the data transformation and model training in action.

---

## 📬 Contact
If you have questions about the "Magic Loop" logic or the ensemble strategy, feel free to reach out!
