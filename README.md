🫁 Asthma Risk Prediction using Logistic Regression
A binary classification project that predicts asthma diagnosis in patients using clinical, environmental, and lifestyle risk factors — built on a synthetic medical dataset of 7,000 patient records.

📌 Problem Statement
Can a machine learning model identify patients at risk of asthma using non-invasive features like smoking history, air pollution exposure, family history, and clinical biomarkers? This project builds a logistic regression classifier to answer that question and explores which risk factors contribute most to asthma diagnosis.

📂 Dataset

Source: Synthetic Asthma Medical Dataset
Records: 7,000 synthetic patient records
Target: Has_Asthma (binary: 0 = No, 1 = Yes)
Note: This is a synthetic dataset — generated to mimic real clinical patterns. Models trained on synthetic data tend to perform optimistically; results should be validated on real-world data before any clinical use.


🔍 Features Used
FeatureTypeDescriptionSmoking_StatusCategoricalNever / Former / Current smokerFamily_HistoryBinaryFamily history of asthmaAllergiesCategoricalType of allergies presentAir_Pollution_LevelCategoricalLow / Medium / High exposurePeak_Expiratory_FlowContinuousMeasures airway obstructionFeNO_LevelContinuousExhaled nitric oxide — airway inflammation marker

🔍 Exploratory Data Analysis
Visualizations were created to understand risk factor relationships with asthma diagnosis:

Age distribution (histplot): Asthma peaks in children under 10, young adults in mid-20s, and elderly patients in mid-60s and 90+
Smoking status (lineplot): Current smokers show approximately 2x higher asthma prevalence vs never/former smokers
Air pollution (barplot): High pollution environments significantly increase asthma risk
Occupation type (histplot): Outdoor occupations correlate with higher diagnosis rates
Peak Expiratory Flow (jointplot KDE): Reduced PEF strongly associated with asthma presence
FeNO Level (lineplot): Elevated FeNO levels clearly indicate higher asthma risk
Family history (lineplot): Emerged as the strongest single predictor
Allergies (kdeplot): Weaker signal due to categorical encoding — see known limitations


⚙️ Data Preprocessing

Dropped rows with missing values in Allergies column
Applied pd.get_dummies() with drop_first=True on categorical columns: Smoking_Status, Allergies, Air_Pollution_Level
Train/test split: 80% train, 20% test (random_state=42)


🤖 Model
Algorithm: Logistic Regression (sklearn default settings)
Logistic regression was chosen as it is interpretable, appropriate for binary classification, and works well as a strong baseline before exploring more complex models.

📊 Results
MetricScoreAccuracy:80.5%
Key Findings

Family history is the strongest single predictor of asthma diagnosis
Current smokers show ~2x higher asthma prevalence compared to never/former smokers
FeNO > 40 ppb serves as a reliable diagnostic threshold (clinical standard is >50 ppb)
Reduced Peak Expiratory Flow confirms airway obstruction and strongly indicates asthma
High air pollution + outdoor occupation significantly increases risk
Allergies underperformed as a predictor — the categorical encoding diluted its signal; a binary Has_Allergies feature would likely perform better


⚠️ Known Limitations

Synthetic data: The dataset was artificially generated. Real clinical data introduces noise, missing values, and class imbalance that this model has not been tested against.
Evaluation gap: Only accuracy and confusion matrix reported. Precision, recall, F1, and ROC-AUC need to be added.
No hyperparameter tuning: Default logistic regression settings used — regularization strength (C) not optimized.
No cross-validation: A single train/test split may not reflect true generalization performance.


🛠️ Tech Stack

Python 3
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn (LogisticRegression, train_test_split, accuracy_score, confusion_matrix)
