# codebasics ML course: codebasics.io, all rights reserved

import pandas as pd
import joblib
import os

# Load models and scalers safely using raw strings
model_young = joblib.load(r"artifacts/model_young.joblib")
model_rest = joblib.load(r"artifacts/model_rest.joblib")
scaler_young = joblib.load(r"artifacts/scaler_young.joblib")
scaler_rest = joblib.load(r"artifacts/scaler_rest.joblib")


# Function to calculate normalized risk score based on medical history
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    # Split by "&" and lowercase
    diseases = medical_history.lower().split(" & ")

    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)

    max_score = 14  # Max combined score of heart disease + second highest
    min_score = 0

    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score


# Preprocess input dictionary into model-ready DataFrame
def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    # Initialize DataFrame with zeros
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    # Fill in values
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    # Calculate normalized risk
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', 'none'))

    # Handle scaling
    df = handle_scaling(input_dict['Age'], df)

    return df


# Scaling function
def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    # Provide a default column if scaler expects it
    df['income_level'] = 0
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis=1, inplace=True)

    return df


# Prediction function
def predict(input_dict):
    input_df = preprocess_input(input_dict)

    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction[0])

'''✅ Key Takeaways / Theory to Remember

File paths

Use raw strings r"artifacts\model.joblib" or forward slashes "artifacts/model.joblib" in Python, especially on Windows.

Key access in dictionaries

Python is case-sensitive. 'Age' ≠ 'age'. Always match exactly.

Data preprocessing for ML

Initialize DataFrame with all expected columns.

Fill zeros first, then assign values for categorical features.

Scale only numeric columns using saved scalers.

Normalized risk score

Convert qualitative medical history into numeric risk score.

Normalize between min and max for uniform input to models.

Model selection

Use model_young if Age <= 25, otherwise model_rest.

Safe scaling

If a scaler expects extra columns, provide defaults (0s) to avoid errors.'''