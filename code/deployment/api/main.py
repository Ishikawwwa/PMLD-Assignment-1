from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from get_pred import ModelRunner

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Upload</title>
    </head>
    <body>
        <h2>Upload an Image</h2>
        <form action="/upload/" method="post" enctype="multipart/form-data">
            <label for="imageUpload">Select an image:</label>
            <input type="file" id="imageUpload" name="file" accept="image/*">
            <br><br>
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """

@app.post("/upload/")
async def upload_file(file: UploadFile):
    
    with open("image.jpg", "wb+") as file_object:
        file_object.write(file.file.read())

    runner = ModelRunner()
    return {"filename": file.filename, "Prediction": runner.get_prediction("image.jpg", "api/classes.txt")}