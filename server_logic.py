from fastapi import FastAPI, UploadFile, File
import pandas as pd
import model_usage as model 
import uvicorn
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/prediction")
async def predict_file(file: UploadFile = File(...)):
    # Read the uploaded file into a DataFrame
    df = pd.read_excel(file.file)
    df["Gender"] = df["Name"].apply(model.safe_predict)

    # Save results to a new file
    output_path = "result.xlsx"
    df.to_excel(output_path, index=False)

    # Return file response
    return FileResponse(output_path, filename="result.xlsx")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
