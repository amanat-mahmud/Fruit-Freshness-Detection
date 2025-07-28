from fastapi import FastAPI, File, UploadFile
from model_helper import predict
app = FastAPI()


@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)): 
    try:
        image_bytes = await file.read()
        ### in real file generate a unique name for the file
        ### for simplicity, we are saving it as temp_file.jpg
        image_path ="temp_file.jpg"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        prediction = predict(image_path)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


