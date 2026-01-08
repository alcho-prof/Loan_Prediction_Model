import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from schemas import LoanApplication
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load model
model_path = os.path.join("models", "loan_model.pkl")
model = None

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: LoanApplication):
    if not model:
        return {"prediction": "Error: Model not loaded"}

    # Preprocessing to match training features
    # Feature Order: ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
    # 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
    # 'Credit_History', 'Semiurban', 'Urban']
    
    # 1. Gender (Male=1, Female=0 - Common assumption, or LabelEncoded F=0, M=1)
    # Let's assume Male=1, Female=0 based on typical 1-hot or manual mapping
    # If LabelEncoder: Female(0) < Male(1).
    gender_val = 1 if data.Gender == "Male" else 0
    
    # 2. Married (Yes=1, No=0)
    married_val = 1 if data.Married == "Yes" else 0
    
    # 3. Dependents (0, 1, 2, 3)
    dep_map = {'0': 0, '1': 1, '2': 2, '3': 3}
    dep_val = dep_map[data.Dependents]
    
    # 4. Education (Graduate=0, Not Graduate=1) - Alphabetical
    edu_val = 0 if data.Education == "Graduate" else 1
    
    # 5. Self_Employed (Yes=1, No=0)
    self_emp_val = 1 if data.Self_Employed == "Yes" else 0
    
    # 6. Property Area (One-Hot Encoded into Semiurban, Urban)
    # If Rural: semi=0, urban=0
    semiurban = 1 if data.Property_Area == "Semiurban" else 0
    urban = 1 if data.Property_Area == "Urban" else 0
    
    # Construct feature vector
    features = [
        gender_val,
        married_val,
        dep_val,
        edu_val,
        self_emp_val,
        data.ApplicantIncome,
        data.CoapplicantIncome,
        data.LoanAmount,
        data.Loan_Amount_Term,
        data.Credit_History,
        semiurban,
        urban
    ]
    
    # --- SMART SCALING WRAPPER ---
    BASELINE_INCOME = 5000.0
    total_input_income = data.ApplicantIncome + data.CoapplicantIncome
    
    # Safety Check: If Loan is outrageously high (e.g. > 5000 which implies 5 Million), reject immediately.
    # The model expects "150" for 150k. So 5000 means 5 Million.
    # 3000000 means 3 Billion. That's definitely a user typo or joke.
    if data.LoanAmount > 10000:
         # Hard Reject for unrealistic numbers
         return {
             "prediction": "Not Eligible", 
             "reason": [f"Loan Amount ({data.LoanAmount}) is realistically too high (likely typo?). Please check units."]
         }
    
    # Use scaling only if income is significantly higher than baseline
    if total_input_income > 15000:
        scale_factor = total_input_income / BASELINE_INCOME
        
        # Scale fields
        model_applicant_income = data.ApplicantIncome / scale_factor
        model_coapplicant_income = data.CoapplicantIncome / scale_factor
        model_loan_amount = data.LoanAmount / scale_factor
        
        # Second Safety Check:
        # If the Scaled Loan Amount is still huge (like 300,000), it means the original input was MASSIVE relative to income.
        # e.g Income 50k, Loan 3M. Factor=10. Scaled Income=5k. Scaled Loan=300k.
        # 300k is huge for 5k income.
        
        input_ratio = data.LoanAmount / (total_input_income + 1)
        # Normal ratio is ~0.03.
        # Your case: 3,000,000 / 50,000 = 60.0 !
        
        if input_ratio > 0.5: # Generous upper bound (Loan is 50% of monthly income? No, 0.5 implies Loan amount (k) is half of income (units))
             # Wait, units. 150 (k) / 5000 = 0.03.
             # 0.5 means Loan is 16x larger than typical ratio.
             return {
                 "prediction": "Not Eligible",
                 "reason": ["Loan Amount is exceptionally high compared to Income."]
             }
    else:
        # Use raw values
        model_applicant_income = data.ApplicantIncome
        model_coapplicant_income = data.CoapplicantIncome
        model_loan_amount = data.LoanAmount

    # Conversion: Term (Years) -> Term (Months)
    # The dataset uses '360' for 30 years. So 1 Year = 12 Months.
    # User inputs Years (e.g. 30). We convert to 360.
    model_term = data.Loan_Amount_Term * 12

    # Construct feature vector
    features = [
        gender_val,
        married_val,
        dep_val,
        edu_val,
        self_emp_val,
        model_applicant_income,
        model_coapplicant_income,
        model_loan_amount,
        model_term, # Converted to months
        data.Credit_History,
        semiurban,
        urban
    ]
    
    # Predict
    try:
        # Reshape and Predict
        final_features = np.array([features])
        prob_array = model.predict_proba(final_features)[0]
        prob_score = prob_array[1] # Probability of Approval
        
        # Decision Threshold
        if prob_score >= 0.45:
            result = "Eligible"
            reason = []
        else:
            result = "Not Eligible"
            reason = []
            
            # Reasons...
            if data.Credit_History == 0.0:
                reason.append("Credit History is marked as poor (0.0).")
            
            # Scaled Ratio Check
            scaled_total_income = model_applicant_income + model_coapplicant_income
            ratio = model_loan_amount / (scaled_total_income + 1)
            
            if ratio > 0.06: 
                reason.append("Loan amount is too high relative to the provided income.")
                
            if data.Loan_Amount_Term < 300:
                 reason.append("Short loan term increases monthly obligation.")
                 
            if not reason:
                 reason.append(f"The calculated probability ({prob_score:.2f}) is below the approval threshold.")
            
        return {"prediction": result, "reason": reason}
    except Exception as e:
        return {"prediction": f"Error during prediction: {str(e)}", "reason": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
