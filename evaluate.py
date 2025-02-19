import torch
import numpy as np
import cv2
from sklearn.metrics import average_precision_score
import os
import json
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet import UNet 
import argparse
# import matplotlib.pyplot as plt

class LIVECellDataset(Dataset):
    def __init__(self, img_dir, annotation_file,num_images=None,transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.num_images = num_images
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        if self.num_images:
            self.images = self.data['images'][:self.num_images]
        else:
            self.images = self.data['images']
        self.annotations = self.data['annotations']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # Initialize empty mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Get annotations for this image
        for ann in self.annotations:
            if ann["image_id"] == img_info["id"]:
                for seg in ann["segmentation"]:
                    pts = np.array(seg, np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts], 1) # Binary mask (1 = cell, 0 = background)

        # Apply transforms
        if self.transform:
            augment = self.transform(image = image,mask = mask)
            image,mask = augment['image'],augment['mask']
            # mask = mask.unsqueeze(0)
        return image, mask

# Define transformations
transform = A.Compose([
    A.Resize(256,256),
    A.Normalize((0.5,), (0.5,)),
    ToTensorV2()
])
#  IoU Calculation for Mask Annotations
def calculate_iou(pred_mask, gt_mask):
    """
    param: 
        pred_mask-The ouput from unet model
        gt_mask- the actual mask of an image
    return Intersection over Union (IoU) between predicted and ground truth binary masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-6)  
    return iou
def evaluate_map(model, test_loader,device,iou_threshold=0.5):
    model.eval()
    iou_scores = []
    ap_scores = []
 
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
 
            outputs = model(images)
            outputs = torch.sigmoid(outputs).cpu().numpy()  
            masks = masks.cpu().numpy()
 
            for pred_mask, gt_mask in zip(outputs, masks):
                pred_mask = (pred_mask > 0.5).astype(np.uint8)  
                gt_mask = (gt_mask > 0.5).astype(np.uint8)  
 
                # Compute IoU
                iou = calculate_iou(pred_mask, gt_mask)
                iou_scores.append(iou)
 
                # Compute Average Precision
                ap = average_precision_score(gt_mask.flatten(), pred_mask.flatten())
                ap_scores.append(ap)
 
    mean_iou = np.mean(iou_scores)
    mean_ap = np.mean(ap_scores)
 
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"mAP: {mean_ap:.4f}")
    return mean_ap
def main(model_path,testimages,testlabels,num_images=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_test = UNet(n_channels=1, n_classes=1).to(device)
    model_test.load_state_dict(torch.load(model_path, map_location=device))
    loader_args = dict(batch_size=2,num_workers=2)
    test_dataset = LIVECellDataset(img_dir=testimages, 
                                annotation_file=testlabels,num_images=num_images,transform=transform)
    test_loader = DataLoader(test_dataset,shuffle=False,**loader_args)
    mAP = evaluate_map(model_test, test_loader,device)
    print(f"mAP:{mAP*100}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate U-Net Model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--images', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file')
    parser.add_argument('--num_images', type=int, required=False,default=None, help='number of images to test')
    args = parser.parse_args()
    model_path = args.model
    testimages=args.images
    testlabels=args.labels
    num_images = args.num_images
    main(model_path,testimages,testlabels,num_images=None)