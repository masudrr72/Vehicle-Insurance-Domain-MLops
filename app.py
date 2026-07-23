import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fastapi.staticfiles import StaticFiles

from src.pipeline.prediction_pipeline import (
    VehicleData,
    PredictionPipeline,
)

app = FastAPI(
    title="Vehicle Insurance Prediction",
    description="End-to-End MLOps Project by MASUDUR RAHMAN",
    version="1.0"
)

app.mount(
    "/static",
    StaticFiles(directory="src/static"),
    name="static",
)

templates = Jinja2Templates(directory="src/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "prediction": None,
            "probability": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,

    Gender: str = Form(...),
    Age: int = Form(...),
    Driving_License: int = Form(...),
    Region_Code: float = Form(...),
    Previously_Insured: int = Form(...),
    Vehicle_Age: str = Form(...),
    Vehicle_Damage: str = Form(...),
    Annual_Premium: float = Form(...),
    Policy_Sales_Channel: float = Form(...),
    Vintage: int = Form(...),
):

    vehicle_data = VehicleData(
        Gender=Gender,
        Age=Age,
        Driving_License=Driving_License,
        Region_Code=Region_Code,
        Previously_Insured=Previously_Insured,
        Vehicle_Age=Vehicle_Age,
        Vehicle_Damage=Vehicle_Damage,
        Annual_Premium=Annual_Premium,
        Policy_Sales_Channel=Policy_Sales_Channel,
        Vintage=Vintage,
    )

    dataframe = vehicle_data.get_data_as_dataframe()

    prediction_pipeline = PredictionPipeline()

    prediction = prediction_pipeline.predict(dataframe)

    prediction_value = int(prediction)

    probability = prediction_pipeline.predict_proba(dataframe)

    prediction_label = (
        "Interested in Vehicle Insurance"
        if prediction_value == 1
        else "Not Interested in Vehicle Insurance"
    )

    probability = round(probability * 100, 2)
    progress_degree = (probability / 100) * 360

    return templates.TemplateResponse(
        request=request,
        name="home.html",
            context={
                "prediction": prediction_label,
                "prediction_value":prediction_value,
                "probability": probability,
                "progress_degree": progress_degree,
            },
    )


@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }


if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )



