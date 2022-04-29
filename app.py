from importlib import import_module
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from dotenv import load_dotenv
import os

import torch
from mongo_connect import mongo_config, import2mongo, export2mongo

load_dotenv()

app = FastAPI(debug=True)
app.mount("/static", StaticFiles(directory="apps/static"), name="static")
templates = Jinja2Templates(directory="apps/templates")


@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(
        name="dashboard.html", context={"request": request}
    )

    return app

@app.get("/predict", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse(
        name="dashboard.html", context={"request": request}
    )

    return app

if __name__ == "__main__":
    uvicorn.run("app:app", host=os.getenv("HOST_ADDRESS"), port=int(os.getenv("PORT")))
