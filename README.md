ML health insurance prediction project

# Predictive Health Insurance Model

## Project Overview
This repository contains the development and deployment of a predictive health insurance premium model for a client, created by a service provider. The project aims to estimate health insurance premiums based on key factors such as age, smoking habits, BMI, and medical history. It is structured in two phases, with this repository focusing on 

**Phase 1: Minimum Viable Product (MVP)** development.

### Objectives
- Develop a high-accuracy predictive model with a minimum accuracy of 97%, ensuring that at least 95% of prediction errors have a percentage difference of less than 10% from actual values.
- Deploy the model on a cloud platform for accessibility by insurance underwriters from any location.
- Create an interactive Streamlit application to facilitate easy input of factors and display premium predictions.

## Scope of Work
### Phase 1: MVP Development
1. **Data Collection and Preprocessing**
   - Gather and clean labeled datasets.
   - Conduct exploratory data analysis (EDA) to understand data patterns and quality.

2. **Model Development**
   - Train and evaluate multiple machine learning models.
   - Optimize the best-performing model to meet accuracy targets.

3. **Model Deployment**
   - Deploy the model on a cloud platform with a focus on security and scalability.

4. **Streamlit Application Development**
   - Build an intuitive application for underwriters to input data and receive predictions.

5. **Testing and Validation**
   - Perform rigorous testing with real-world data to ensure reliability.

6. **Documentation and Training**
   - Provide comprehensive documentation and training materials for underwriters.

7. **Artifacts**
artifact folder contains the joblib files for the trained model and the Streamlit app. The model is trained on a dataset of 50000 samples with 13 features, split into yount er and older age groups. The model achieves an accuracy of 98.5% on the test

### Deliverables
- Trained and deployed predictive model.
- Fully functional Streamlit application.
- Detailed documentation and training resources.

## Acceptance Criteria
- Successful deployment of the model and Streamlit application.
- Model achieves a minimum accuracy of 97%.
- Application is user-friendly and operational for underwriters.

## Getting Started
1. Clone the repository: `git clone https://github.com/shekhus/ml-project-premium-prediction.git

install required libraries
pip install -r requirements.txt 

run the following command in your terminal to run the project

streamlit run .\main.py
