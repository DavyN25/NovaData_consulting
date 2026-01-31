# Retail Banking Simulator — UCI + Loan Classification Project
NovaData Consulting — Final Bootcamp Project

## Team Member


## Project Overview
The objective of this project is to determine how a retail bank can combine marketing behaviour data and loan risk indicators to build more accurate customer segments, recommend suitable financial products, and improve the performance of marketing campaigns using data analysis and visualization.

- **Objective:** provide a clear, structured, and reproducible workflow from raw data to final recommendations. 
- **Approach:** We followed a dual‑dataset, unified‑framework approach designed to combine marketing behavior analysis (UCI dataset) with credit risk assessment (Loan dataset). The goal was to understand customers holistically and support smarter product targeting.  
- **Dataset:**
- UCI dataset : https://www.kaggle.com/datasets/adityamhaske/bank-marketing-dataset?resource=download
- Loan default : https://www.kaggle.com/datasets/nikhil1e9/loan-default

## Final project presentation
The results from the analysis can be found in the presentation slides of the project:
https://www.canva.com/design/DAG_ydo4qRI/O39eMMxw85LqMw5gMe61yQ/edit


## 2. Tools & Libraries
#### 2.1.Programming & Data Analysis
- Python
- pandas
- numpy
- scikit‑learn
- xgboost
- seaborn
- matplotlib
#### 2.2 Machine Learning Techniques
- Logistic Regression
- Random Forest
- XGBoost
- Threshold tuning
- SMOTE for class imbalance
- One‑hot encoding
- Feature scaling
#### 2.3. Dashboarding
- Tableau (interactive dashboards, segmentation, KPIs)
  
#### 2.4. Collaboration & Project Management
- GitHub 
- Trello 
- Jupyter Notebook
- Canvas


## Day 1 - 2 - Project Initiation & Data Selection
On Day 1 we focused on dataset selection and cleaning. 

#### 1.1 Introduction of the Project 

We are a data and analytics advisory firm specializing in financial services. 
- Analyze customer behaviour
- Evaluate loan risk
- Build unified customer segments
- Recommend product strategies
- Design a decision‑making dashboard

#### 1.2 The company objectives:
- Marketing : maximize recall (don’t miss potential subscribers).​
- Risk : balance recall and precision (avoid approving risky loans).​

#### 1.3. Created a Kanban Board for Project Management Purposes on Trello.
We organized the tasks here:
https://trello.com/b/MOUTGF8n/novadata-consulting

#### 1.4. Created a Github Repository for the Project:
https://github.com/DavyN25/NovaData_consulting

We created the repository so we can share and all files are always updated.


### 2. Dataset Selection

Our project integrates two complementary datasets to give a unified view of customer behaviour and financial risk in a retail banking context.
 

- UCI dataset : https://www.kaggle.com/datasets/adityamhaske/bank-marketing-dataset?resource=download
- Loan default : https://www.kaggle.com/datasets/nikhil1e9/loan-default

#### 2.1. UCI Bank Marketing Dataset (Marketing Behaviour)
This dataset contains information about customers contacted during a marketing campaign for a term deposit:

Purpose in the project
- Understand customer behaviour
- Identify segments most likely to subscribe
- Analyse contact strategy effectiveness
  
**Key variables:**
- Demographics: age, job, marital status, education
- Financial indicators: housing loan, personal loan, default flag
- Campaign behaviour: contact type, duration, number of contacts, previous outcomes
- Target: y (Subscribed: yes/no)

It reveals who responds to marketing, and under which conditions.

#### 2.2. Loan Default Dataset (Credit Risk)
This dataset contains borrower information used to predict whether a customer will default on a loan.
Purpose in the project
- Evaluate financial capacity
- Predict probability of default
- Identify high‑risk vs low‑risk customer profiles
  
**Key variables:**
- Income level
- Loan amount & term
- Credit score
- Employment length
- Debt‑to‑income indicators
- Target: default (yes/no)

It reveals who is financially reliable, and who represents a credit risk.

#### 2.3 Melodi INSEE API Dataset (Real‑Life Enrichment)
Due to time constraints, we could not fully integrate this dataset into the modeling pipeline, but we successfully retrieved it to demonstrate how real demographic data could enhance segmentation.
It provides:
- Population density
- Income distribution
- Household structure
- Regional socio‑economic indicators
This dataset shows how the simulator could be extended in future iterations.

Together, they allow the bank to answer:
- Who is likely to subscribe and financially safe?
- Which segments should receive which product?
- How to maximize conversion while minimizing risk?
- How real demographic context can refine segmentation?


## Day 3-7  

### 1.  Data Examination & Hypothesis Procedure

once the project goal was defined, we moved into a structured exploration phase to understand each dataset, identify patterns, and formulate hypotheses that would guide our modeling strategy.
#### 1.1. Initial Data Profiling
We began by examining both datasets to understand their structure and quality:
- Checked dataset dimensions
- Inspected column types (categorical, numerical, boolean)
- Identified missing values, duplicates, and outliers
- Reviewed distributions of key variables (age, income, duration, loan amount, etc.)
- Looked for class imbalance in the target variables (subscription and default)
This step helped us understand the shape, limitations, and opportunities within each dataset.

#### 1.2. Exploratory Data Analysis (EDA)
We performed a detailed EDA to uncover behavioural and financial patterns:
UCI Dataset (Marketing Behaviour)
- Analyzed subscription rates across age groups, job types, marital status
- Examined how contact frequency and duration influenced conversion
- Compared performance of contact channels (cellular vs telephone)
- Identified segments with the highest subscription probability
- Visualized correlations between features and the target
Loan Dataset (Credit Risk)
- Explored default rates by income level, loan amount, credit score
- Identified high‑risk vs low‑risk borrower profiles
- Analyzed how financial capacity indicators relate to default
- Visualized distributions and outliers using boxplots and histograms
- Checked correlations between numerical variables
This phase allowed us to see the story inside the data before building any models.

#### 1.3. Hypothesis Formulation
Based on EDA, we formulated hypotheses to test with machine learning models.
Marketing Hypotheses (UCI)
- Customers contacted 1–3 times are more likely to subscribe
- Younger and senior customers may be more responsive than mid‑age groups
- Contact duration is a strong predictor of subscription
- Certain job categories (e.g., management, technician) may convert better
- Previous campaign success increases the probability of subscription
Risk Hypotheses (Loan)
- Low‑income borrowers have a significantly higher default rate
- Higher loan amounts increase default probability
- Credit score is a strong negative predictor of default
- Employment length correlates with financial stability
- Combining income + loan amount + credit score improves risk segmentation
These hypotheses guided our feature engineering, model selection, and business interpretation.

#### 1.4. Cross‑Dataset Hypothesis
Because the project integrates marketing and risk analytics, we also formulated a combined hypothesis:
High‑value customers are those who are both likely to subscribe and financially safe.

This led us to explore:
- Whether high‑income customers also show high subscription probability
- Whether risky borrowers appear in segments that marketing teams typically target
- How to align product recommendations with both behaviour and risk

#### 1.5. Preparing for Modeling
The insights and hypotheses from EDA shaped:
- Which features to keep or engineer
- How to handle imbalance (SMOTE)
- Which models to test
- How to evaluate them (recall for marketing, precision/recall balance for risk)


## 2. Modeling Approach
Our modeling strategy was designed to compare multiple supervised learning algorithms across two different business problems — marketing conversion prediction and loan default risk prediction — while aligning model selection with the bank’s operational priorities.

#### 2.1. Define the Modeling Objective
Because the project combines two datasets, we defined two prediction tasks:
UCI Dataset — Marketing Conversion
Goal: Predict whether a customer will subscribe to a financial product.
Business priority: Maximize recall (don’t miss potential subscribers).
Loan Dataset — Credit Risk
Goal: Predict whether a borrower will default on a loan.
Business priority: Balance recall and precision (avoid approving risky loans while not rejecting too many safe customers).
This dual‑objective setup guided all modeling decisions.

#### 2.2. Data Preparation for Modeling
Before training models, we applied a consistent preprocessing pipeline:
- One‑hot encoding for categorical variables
- Scaling of numerical features
- Train/test split with stratification
- Handling class imbalance using SMOTE
- Removal of outliers (Loan dataset)
- Feature selection based on EDA insights
This ensured that all models were trained on clean, comparable data.

#### 2.3. Models Trained and Compared
We selected models commonly used in retail banking and marketing analytics:
Baseline Model
- Logistic Regression
- Interpretable
- Fast to train
- Provides clear coefficients for business teams
Tree‑Based Models
- Random Forest
- Handles non‑linear relationships
- Robust to noise and outliers
- XGBoost
- State‑of‑the‑art performance on tabular data
- Excellent for imbalanced classification
- Tunable thresholds for business needs
Each model was trained twice:
- Standard version
- SMOTE‑balanced version
This allowed us to evaluate the impact of class imbalance correction.

#### 2.4. Evaluation Metrics
We evaluated models using metrics aligned with business priorities:
UCI (Marketing)
- Recall — most important
- Precision
- F1 Score
- AUC (ROC)
- 
Loan (Risk)
- Precision & Recall trade‑off
- F1 Score
- AUC (ROC)
- Confusion Matrix
This ensured that model selection was not only technical but also operationally relevant.

#### 2.5. Final Model Selection
After comparing all models and thresholds:
UCI Dataset
- XGBoost (threshold 0.5) delivered the best recall and AUC
- Logistic Regression (threshold 0.3) offered a balanced alternative
Loan Dataset
- XGBoost + SMOTE provided the best separation between risky and safe borrowers
- Logistic Regression remained useful for interpretability

#### 2.6. Integration into the Retail Banking Simulator
Model outputs were integrated into a unified segmentation framework:
- High‑value, low‑risk customers
- High‑value, high‑risk customers
- Low‑value, low‑risk customers
- Low‑value, high‑risk customers
This allowed the bank to tailor product recommendations and marketing strategies based on both behaviour and risk



## Day 8-9– Conclusions & Assembling Report
On Day 8 we focused on insights and visualization

## 1. Results & Insights
Our analysis produced a unified view of customer behaviour (UCI dataset) and financial risk (Loan dataset). Together, these insights form the foundation of the Retail Banking Simulator, enabling smarter segmentation, better targeting, and safer credit decisions.

#### 1.1. Marketing Insights (UCI Dataset)
Contact Strategy Drives Conversion
- Customers contacted 1–3 times showed the highest subscription rates.
- Conversion dropped sharply after 4+ contacts, indicating diminishing returns.
-> Banks should prioritize short, efficient campaigns.
High‑Value Segments Identified
- Young High‑Balance and Senior High‑Balance customers had the strongest subscription performance.
->Combining behavioural and financial segmentation improves targeting precision.
Mid‑Age Customers Underperform
- The 26–45 age group showed lower responsiveness compared to younger and senior customers.
-> Marketing resources should be reallocated toward more receptive segments.
Contact Duration Is a Strong Predictor
- Longer call durations correlated with higher subscription probability.
->Duration acts as a behavioural signal of interest.

#### 1.2. Credit Risk Insights (Loan Dataset)
Income Is the Strongest Predictor of Default
- Low‑income borrowers had a 67% default rate, nearly double that of higher‑income groups.
->Income segmentation is essential for risk modeling.
XGBoost + SMOTE Provided the Best Risk Detection
- This model delivered the highest recall and AUC, effectively separating risky from safe borrowers.
->Ideal for credit teams aiming to reduce losses.
Financial Capacity Indicators Matter
- Loan amount, credit score, and employment length were strong predictors of default.
->Combining these features improves risk stratification.

#### 1.3. Model Performance Insights
UCI (Marketing Conversion)
- XGBoost (threshold 0.5) achieved the best recall and AUC.
- Logistic Regression (threshold 0.3) provided a balanced alternative with strong interpretability.
->Marketing teams can choose between aggressive targeting or balanced targeting.
Loan (Default Prediction)
- XGBoost + SMOTE delivered the best class separation.
- Logistic Regression remained valuable for explainability.
->Risk teams can combine performance with transparency.

#### 1.4. Cross‑Dataset Insights (Unified Customer View)
High‑Value ≠ Low‑Risk
Some segments with high subscription probability also showed elevated default risk.
->Marketing and credit teams must coordinate to avoid promoting products to risky customers.
Best Opportunity Segment
Customers who are:
- High‑income
- High‑balance
- Responsive to campaigns
- Low default risk
represent the ideal target group for cross‑selling and product bundling.
Behaviour + Risk = Better Product Strategy
Combining both datasets allowed us to:
- Match the right product to the right customer
- Avoid risky approvals
- Improve campaign ROI
- Reduce credit losses

#### 1.5. Dashboard Insights (Tableau)
UCI Dashboard
- Clear segmentation by age, income, contact frequency
- KPI boxes showing conversion ratess
- Filters for job, marital status, and campaign behaviour
  
Loan Dashboard
- Risk heatmaps by income and loan amount
- Default probability distributions
- Feature importance visualizations
Final Retail Banking Simulator
- Unified segmentation combining behaviour + risk
- Product recommendation logic
- Executive‑ready storytelling


### 2.Business Recommendations

#### 2.1. Align Marketing & Risk Teams
Create shared customer segments combining behaviour + risk.
#### 2.2. Prioritize High‑Value, Low‑Risk Segments
Ideal for cross‑selling and premium products.
#### 2.3. Optimize Contact Strategy
Short, efficient campaigns (1–3 contacts) perform best.
#### 2.4. Strengthen Credit Controls
Use XGBoost + SMOTE for loan approvals and risk thresholds.
#### 2.5. Tailor Product Recommendations
Match product type to customer segment:
High‑income, low‑risk → premium credit & investment
High‑income, high‑risk → savings products
Low‑income, low‑risk → micro‑loans
Low‑income, high‑risk → avoid credit products
#### 2.6. Deploy the Retail Banking Simulator
Use it across marketing, product, and risk teams.
#### 2.7. Move Toward Real‑Time Decisioning
Deploy models via API and connect dashboards to live data.


## 4.Conclusion

This project demonstrates how integrating marketing analytics with credit risk modeling creates a more complete and actionable understanding of retail banking customers. By combining the UCI and Loan datasets (and preparing for future integration with real demographic data via the Melodi INSEE API) we built a unified framework that improves segmentation, enhances product targeting, and reduces credit risk.
The Retail Banking Simulator transforms complex analytics into practical strategies that help banks grow responsibly increasing customer engagement while protecting against financial losses.



### 3. Challenges in the Project
Throughout the project, the team encountered several technical and organizational challenges that shaped our final approach and highlighted opportunities for future improvements.

#### 3.1. Integrating Two Very Different Datasets
The UCI and Loan datasets differed in structure, cleanliness, and business purpose.
We had to design two separate cleaning pipelines and then merge insights into a unified segmentation framework.

#### 3.2. Class Imbalance
Both targets were imbalanced:
- Low subscription rate in UCI
- Concentrated default rate in Loan
This required:
- SMOTE
- Threshold tuning
- Careful metric selection

#### 3.3. Outliers and Data Quality
The Loan dataset contained extreme values and inconsistent financial indicators.
The UCI dataset had categorical complexity and duration‑related leakage risks.
We addressed this with:
- Outlier removal
- Feature engineering
- Careful interpretation

#### 3.4. Marketing vs Risk Priorities
Marketing wants high recall, risk wants high precision.
Balancing these conflicting goals required:
- Multiple models
- Threshold tuning
- A unified segmentation logic

#### 3.5. Model Interpretability
XGBoost performed best but is less interpretable.
Logistic Regression was easier to explain but less powerful.
We had to balance performance with transparency.

#### 3.6. Dashboard Integration
Merging insights from two datasets into a single simulator required:
- Consistent KPIs
- Clear segmentation
- Strong visual storytelling

#### 3.7. SQL Limitations
We attempted to  import Loan dataset to run SQL queries, but:
- The dataset was too heavy
- after one hour of unsucced importation of one of the 3 tables we gave up and return on Python.


#### 3.8. Time Constraints & External Data Integration
We retrieved the Melodi INSEE API dataset to enrich the project with real demographic data.
However, due to limited time:
- We could not fully integrate it into the modeling pipeline
- We used it as a proof of concept for future expansion
- It demonstrated how real‑world data could enhance segmentation and product strategy


## Day 10 – Project Presentation
On Day 10 we presented our findings and pitched the business plan.  

### 1. Relationship charts

We draw a Relationship chart in Miro from our database, for practise purposes. 
https://miro.com/app/board/uXjVGKbwHLo=/
https://miro.com/app/board/uXjVGKSNh5o=/

Then we translate the chart to DrawDB to get code for SQL, for practise purposes. 
Drawdb : into the "sql_scripts" folder

We had the chart images in "sql_scripts" folder. 

### 2. Final project presentation
The results from the analysis can be found in the presentation slides of the project:
https://www.canva.com/design/DAG_ydo4qRI/O39eMMxw85LqMw5gMe61yQ/edit


