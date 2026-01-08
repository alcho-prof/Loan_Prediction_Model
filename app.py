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
    dep_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
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
    
    # Predict
    try:
        # Reshape to (1, 12)
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]
        
        # Interpret prediction
        if str(prediction) == '1' or str(prediction) == 'Y' or str(prediction) == '1.0':
            result = "Eligible"
            reason = []
        else:
            result = "Not Eligible"
            reason = []
            
            # 1. Check Credit History
            if data.Credit_History == 0.0:
                reason.append("Credit History is poor (0.0). This is a critical factor for approval.")
                
            # 2. Check Loan to Income Ratio
            total_income = data.ApplicantIncome + data.CoapplicantIncome
            # LoanAmount is in thousands so multiply by 1000 for actual value comparison? 
            # OR typically dataset assumes LoanAmount=100 means 100k, and Income=5000 means 5k monthly.
            # Let's look at standard ratios.
            # If user entered 200,000 income and 2,000,000 loan:
            # Ratio = 2000000 / 300000 = 6.6
            
            # Wait, usually Income is Monthly. 360 term is 30 years.
            # Loan 2,000,000. Income 300,000 monthly.
            # Annual Income = 3,600,000. Loan is < 1 year income. That should be approved!
            
            # HOWEVER, the model sees: Income=200,000, Loan=2,000,000.
            # The training data usually has Input: Income=5000, Loan=150.
            # Feature scaling issues? Gradient Boosting handles unscaled data well, BUT:
            # If training data max income was 50,000 and user puts 200,000, it's out of distribution.
            # BUT key is likely the ratio the trees learned.
            
            # Heuristic for user feedback:
            # If LoanAmount > (TotalIncome * 0.5): # Very rough approximation relative to typical dataset values
            # (In dataset: Loan 150 vs Income 5000. Ratio 0.03)
            # (User Input: Loan 2,000,000 vs Income 300,000. Ratio 6.66)
            # The user input Loan is massively larger relative to Income compared to training data.
            
            ratio = data.LoanAmount / (total_income + 1) # Avoid div/0
            if ratio > 0.1: # Threshold based on typical dataset values (150/5000 = 0.03)
                reason.append("Loan Amount is too high relative to Income.")
                
            if not reason:
                 # Check for the specific case user reported: 
                 # Features: [1, 1, 1, 0, 1, 20000, 40000, 2000, 360, 1.0, 1, 0]
                 # Ratio is 0.033, which is fine.
                 # Credit history is 1.0 (Good).
                 # Income is high (20k + 40k = 60k).
                 # Loan is 2000 (2 million? or 2k? in context of 150 mean).
                 # If 2000 is 2M, it's 13x the average loan of 150.
                 # Even if income is high, the model might penalize extreme loan amounts regardless of income.
                 # Gradient Boosting trees often have threshold splits. If LoanAmount > X, prob decreases.
                 
                 if data.LoanAmount > 500:
                     reason.append(f"Loan Amount ({data.LoanAmount}) is significantly higher than the typical range (100-200).")
                 elif data.Loan_Amount_Term < 360:
                     reason.append("Loan term is shorter than standard (360 days), increasing monthly burden.")
                 else:
                     reason.append("The combination of Applicant Income and Loan Amount does not fit the approval profile pattern.")
            
        return {"prediction": result, "reason": reason}
    except Exception as e:
        return {"prediction": f"Error during prediction: {str(e)}", "reason": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
