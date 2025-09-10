from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import pickle
import pandas as pd
import re
import os

model_dir = "."

try:
    with open(os.path.join(model_dir, 'obesity_rf.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)
    with open(os.path.join(model_dir, 'mapping.pkl'), 'rb') as f:
        data_mappings = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Required model file not found: {e}. "
        "Make sure 'obesity_rf.pkl', 'scaler.pkl', 'encoder.pkl', and 'mapping.pkl' "
        f"are in the '{model_dir}' directory."
    )
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor files: {e}")

# Labels for obesity
lb = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

# Define the exact order of features that the model expects after all preprocessing.
feature_order = [
    'Gender', 'Age', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP',
    'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'BMI',
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike',
    'MTRANS_Public_Transportation', 'MTRANS_Walking'
]

# Define numerical columns to scale with RobustScaler
scale_col = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']

class ObesityPredictRequest(BaseModel):
    Gender: str = Field(..., description="Gender (Male/Female)")
    Age: float = Field(..., description="Age in years (e.g., 31)")
    Height: float = Field(..., gt=0, description="Height in meters (e.g., 1.70)")
    Weight: float = Field(..., gt=0, description="Weight in kilograms (e.g., 80.0)")
    family_history_with_overweight: str = Field(..., description="Family history with overweight (yes/no)")
    FAVC: str = Field(..., description="Frequent consumption of high caloric food (yes/no)")
    FCVC: float = Field(..., ge=1.0, le=3.0, description="Frequency of consumption of vegetables (1.0 to 3.0)")
    NCP: float = Field(..., ge=1.0, le=4.0, description="Number of main meals (1.0 to 4.0)")
    CAEC: str = Field(..., description="Consumption of food between meals (no, Sometimes, Frequently, Always)")
    SMOKE: str = Field(..., description="Smoker (yes/no)")
    CH2O: float = Field(..., ge=1.0, le=3.0, description="Consumption of water daily (1.0 to 3.0)")
    SCC: str = Field(..., description="Calories consumption monitoring (yes/no)")
    FAF: float = Field(..., ge=0.0, le=3.0, description="Physical activity frequency (0.0 to 3.0)")
    TUE: float = Field(..., ge=0.0, le=3.0, description="Time using technology devices (0.0 to 3.0)")
    CALC: str = Field(..., description="Consumption of alcohol (no, Sometimes, Frequently)")
    MTRANS: str = Field(..., description="Transportation mode (Automobile, Public_Transportation, Walking, Bike, Motorbike)")

class ObesityPredictResponse(BaseModel):
    predicted_obesity_level: str

app = FastAPI(
    title="Obesity Level Prediction API",
    description="Predicts obesity levels (e.g., Normal_Weight, Obesity_Type_I) "
                "based on various lifestyle and demographic factors using a pre-trained Random Forest model.",
    version="1.0.0"
)

def _apply_gender_map(gender_str: str) -> int:
    """Maps 'Male'/'Female' to 0/1."""
    mapping = data_mappings['person_gender']
    if gender_str in mapping:
        return mapping[gender_str]
    raise ValueError(f"Invalid Gender value: '{gender_str}'. Expected 'Male' or 'Female'.")

def _apply_yes_no_map(value_str: str) -> int:
    """Maps 'yes'/'no' to 1/0."""
    mapping = data_mappings['yes_no']
    if value_str in mapping:
        return mapping[value_str]
    raise ValueError(f"Invalid yes/no value: '{value_str}'. Expected 'yes' or 'no'.")

def _apply_caec_calc_map(value_str: str) -> int:
    """Maps CAEC/CALC categories to numerical based on 'caec' mapping."""
    mapping = data_mappings['caec']
    if value_str in mapping:
        return mapping[value_str]
    raise ValueError(f"Invalid CAEC/CALC value: '{value_str}'. Expected 'no', 'Sometimes', 'Frequently', or 'Always'.")

@app.get("/")
async def root():
    return {"message": "Welcome to the Obesity Level Prediction API! Navigate to /docs for interactive documentation."}

@app.post("/predict", response_model=ObesityPredictResponse)
async def predict_obesity(request: ObesityPredictRequest):
    try:
        input_data_dict = request.dict()

        # verify height, derive BMI and drop height & weight
        if input_data_dict['Height'] <= 0:
            raise HTTPException(status_code=400, detail="Height must be positive for BMI calculation.")
        input_data_dict['BMI'] = input_data_dict['Weight'] / (input_data_dict['Height']**2)

        # mapping
        input_data_dict['Gender'] = _apply_gender_map(input_data_dict['Gender'])
        input_data_dict['family_history_with_overweight'] = _apply_yes_no_map(input_data_dict['family_history_with_overweight'])
        input_data_dict['FAVC'] = _apply_yes_no_map(input_data_dict['FAVC'])
        input_data_dict['SMOKE'] = _apply_yes_no_map(input_data_dict['SMOKE'])
        input_data_dict['SCC'] = _apply_yes_no_map(input_data_dict['SCC'])
        input_data_dict['CAEC'] = _apply_caec_calc_map(input_data_dict['CAEC'])
        input_data_dict['CALC'] = _apply_caec_calc_map(input_data_dict['CALC'])

        # create df for features excluding MTRANS because we will apply one hot encoding
        features_for_df = {k: input_data_dict[k] for k in input_data_dict if k != 'MTRANS'}
        df_processed = pd.DataFrame([features_for_df])

        # one hot encoding for MTRANS
        mtrans_temp_df = pd.DataFrame([input_data_dict['MTRANS']], columns=['MTRANS'])
        encoded_mtrans_array = encoder.transform(mtrans_temp_df)
        encoded_mtrans_df = pd.DataFrame(
            encoded_mtrans_array,
            columns=encoder.get_feature_names_out(['MTRANS']),
            index=df_processed.index
        )
        df_final_features = pd.concat([df_processed, encoded_mtrans_df], axis=1)

        df_final_features = df_final_features[feature_order]

        # scaling
        df_final_features[scale_col] = scaler.transform(df_final_features[scale_col])

        # prediction
        prediction_numeric = model.predict(df_final_features)[0]
        prediction_label = lb.get(prediction_numeric, "Unknown")

        return ObesityPredictResponse(predicted_obesity_level=prediction_label)

    except (ValueError, KeyError) as e:
        # error handling from preprocessing
        raise HTTPException(status_code=400, detail=f"Data validation or transformation error: {e}")
    except ValidationError as e:
        # error handling from wrong datatype/input
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        # overall error handling
        print(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")