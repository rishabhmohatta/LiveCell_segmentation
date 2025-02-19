# LiveCell_segmentation
  This project used to segment the LiveCell dataset
```
```

##  Setup & Installation

### **1️ Clone the Repository**
```sh
git clone https://github.com/rishabhmohatta/LiveCell_segmentation.git
cd LiveCell_segmentation
```

### **2️ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️ Start FastAPI Server**
```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## API Usage

### **1️ Make a Prediction**
#### **(Get Output Image Directly)**
```sh
curl -X POST -F "file=@test.jpg" "http://127.0.0.1:8000/predict/" --output result.png
```



---

##  Docker Deployment

### **1️ Build Docker Image**
```sh
docker build -t unet-api .
```

### **2️ Run the Container**
```sh
docker run -p 8000:8000 unet-api
```

---

##  Training the Model
The U-Net model was trained on the **LIVECell dataset** using:
- **Model:** Unet
- **Input Image Size:** 256x256

### ** mAP Score**
To evaluate the model performance, **mAP (Mean Average Precision) was calculated** on the test dataset.

#### **Compute mAP on Test Dataset:**
```sh
python evaluate.py --model unet_livecell_best.pth --images test/ --labels test.json [--num_images]
```

Example output:
```sh
For 100 test images: mAP: 0.84, IoU: 0.81
For total(1500) test images : mAP: 0.83, IoU: 0.78
```

---
## 📄 References
- **U-Net Architecture Paper:** [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- **LIVECell Dataset Paper:** [Edlund et al., 2021](https://www.nature.com/articles/s41592-021-01249-6)
- **PyTorch U-Net Implementation:** [GitHub - milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

---

##  To-Do
- Model trained on single class as Dataset has 8 type of cell can create a multiclass model
- Optimizing model for real time inference (like converting it into onnx ,trt,etc)
- Training the model with an pretrained resnet architecture as a backbone for better accuracy
- Using other different model like deeplab for segmentation



