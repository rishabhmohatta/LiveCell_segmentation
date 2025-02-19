import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response,JSONResponse
# from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet 
# from PIL import Image
# import io
# Initialize FastAPI App
app = FastAPI()

#  Load Model at Startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None # Global model variable

@app.on_event("startup")
def load_model():
    """
    Load the trained U-Net model only once when FastAPI starts.
    """
    global model
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load("unet_livecell_best.pth", map_location=device))
    model.eval()
    print(" Model loaded successfully!")
# Image Preprocessing
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

# Overlay Function

def overlay_mask(image, mask, alpha=0.5):
    """
    Overlay predicted mask on the test image.
    - `alpha` controls transparency (0 = only image, 1 = only mask).
    """
    image= cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    color_mask = np.zeros_like(image)
    color_mask[:, :, 2] = mask # Red channel for mask visualization
    overlay = cv2.addWeighted(image, 1, color_mask, alpha, 0)
    return overlay

# API Endpoint: Upload Image for Segmentation# ------------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """  Accepts an image, runs segmentation, and returns overlayed result.  """  
    try:    
        global model # Use preloaded model
        if model is None:
            return JSONResponse(content={"error": "Model not loaded yet"}, status_code=500)
        # Read Image
        # image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        file_bytes = np.frombuffer(await file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        # image = np.array(image)

        # Store Original Size
        original_size = image.shape[:2]

        # Preprocess Image
        transformed = transform(image=image)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        # Run Model Inference
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output).cpu().numpy().squeeze()

        # Convert Prediction to Mask
        mask = (output > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (original_size[1], original_size[0])) # Resize to original size

        # Overlay Mask on Image
        overlayed_image = overlay_mask(image, mask)

        # Encode to Bytes
        _, img_encoded = cv2.imencode(".png", overlayed_image)
        return Response(content=img_encoded.tobytes(),media_type="image/png")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run API Locally (For Testing)# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)