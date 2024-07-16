from fastapi import FastAPI, UploadFile, File, HTTPException, status
from typing import Annotated
from keras import models  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import base64
from pydantic import BaseModel
import uuid
import io

app = FastAPI()


class ImageBase64(BaseModel):
    base64_str: str


def teachable_machine_model(image_file):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = models.load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_file).convert("RGB")

    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return {'class_name': class_name, 'confidence_score': confidence_score}



# This function predicts image taking uploaded picture as a parameter
@app.post("/picture-prediction", tags=['image_recognition'])
async def predict_picture(uploaded_image: Annotated[UploadFile, File(...)]):
    # Image in binary format
    binary_image = await uploaded_image.read()

    parameters = teachable_machine_model(uploaded_image.file)

    # converting image to base64 format
    base64_image = base64.b64encode(binary_image).decode()

    begin = 2
    end = len(parameters['class_name']) - 1

    confidence_score_in_percentage = parameters['confidence_score'] * 100

    return {'name': parameters['class_name'][begin:end], 'confidence_score': f"{confidence_score_in_percentage:.3f}%", "image_encoded": base64_image}



# This function predicts image taking base64 as a parameter
@app.post("/base64-prediction", tags=['image_recognition'])
async def predict_base64(image_data: ImageBase64):
    image_in_binary = base64.b64decode(image_data.base64_str)

    parameters = teachable_machine_model(io.BytesIO(image_in_binary))

    begin = 2
    end = len(parameters['class_name']) - 1

    confidence_score_in_percentage = parameters['confidence_score'] * 100

    return {'name': parameters['class_name'][begin:end], 'confidence_score': f"{confidence_score_in_percentage:.3f}%"}



# This function decodes base64 and saves it inside img/decoded_images folder.
@app.post("/decode-base64", tags=['save_base64_image'])
async def decode_base64_encoded_image(image_data: ImageBase64):
    try:
        decoded_image = base64.b64decode(image_data.base64_str)

        image_name = f"img/decoded_images/{uuid.uuid4()}.jpeg"

        with open(image_name, 'wb') as f:
            f.write(decoded_image)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="you entered incorrect base64 string") from e


    return {"message": "Image saved successfully", 'filename': image_name}


