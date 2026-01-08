from pydantic import BaseModel
from typing import Literal

class LoanApplication(BaseModel):
    Gender: Literal['Male', 'Female']
    Married: Literal['Yes', 'No']
    Dependents: Literal['0', '1', '2', '3+']
    Education: Literal['Graduate', 'Not Graduate']
    Self_Employed: Literal['Yes', 'No']
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: Literal['Urban', 'Semiurban', 'Rural']
