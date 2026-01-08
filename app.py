from fastapi import FastAPI,Form 
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
import pickle

app=FastAPI() # Creating a FastAPI Object 

#loading the training model 
