from importlib import import_module
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import uvicorn
from dotenv import load_dotenv
import os

import torch
from mongo_connect import mongo_config, import2mongo, export2mongo
from prediction import inference
load_dotenv()

app = FastAPI(debug=True)
app.mount("/static", StaticFiles(directory="apps/static"), name="static")
templates = Jinja2Templates(directory="apps/templates")
class Item(BaseModel):
          language = "english"

class Checkcode(BaseModel):
    hash: str
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse(name="index.html", context={"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request, hash: str):
    info = import2mongo(hash)
    return templates.TemplateResponse(
        name="dashboard.html", context={"request": info}
    )

@app.get("/predict", response_class=Checkcode)
async def get_dashboard(request: Request):
    return inference(request)

if __name__ == "__main__":
    uvicorn.run("app:app", host=os.getenv("HOST_ADDRESS"), port=int(os.getenv("PORT")))
