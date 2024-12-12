import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from scipy.linalg import sqrtm
import torch.nn.functional as F

# Placeholder paths for generated and real image folders
GENERATED_IMAGES_PATH = './generated_images'
REAL_IMAGES_PATH = './real_images'

# Transformation to match Inception model requirements
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load InceptionV3 model for feature extraction
inception_model = models.inception_v3(pretrained=True, transform_input=False).eval()

# Function to calculate Inception Score
def calculate_inception_score(images_path, batch_size=32, splits=10):
    dataset = ImageFolder(images_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []

    # Get logits from inception model for all images
    with torch.no_grad():
        for images, _ in dataloader:
            logits = inception_model(images)
            preds.append(F.softmax(logits, dim=1).cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    split_scores = []
    
    # Split predictions and calculate KL divergence for each split
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
        p_y = np.mean(part, axis=0)
        split_scores.append(np.exp(np.mean([np.sum(p * np.log(p / p_y)) for p in part])))
    
    return np.mean(split_scores), np.std(split_scores)

# Function to calculate Fréchet Inception Distance (FID)
def calculate_fid(real_images_path, generated_images_path, batch_size=32):
    real_dataset = ImageFolder(real_images_path, transform=transform)
    generated_dataset = ImageFolder(generated_images_path, transform=transform)
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    generated_dataloader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)
    
    def get_activations(dataloader):
        activations = []
        with torch.no_grad():
            for images, _ in dataloader:
                act = inception_model(images).cpu().numpy()
                activations.append(act)
        return np.concatenate(activations, axis=0)
    
    # Get activations for real and generated images
    real_activations = get_activations(real_dataloader)
    generated_activations = get_activations(generated_dataloader)
    
    # Calculate mean and covariance of activations
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_generated, sigma_generated = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)
    
    # Calculate FID score
    mean_diff = mu_real - mu_generated
    covmean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = mean_diff.dot(mean_diff) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fid

# Example Usage
if __name__ == "__main__":
    # Calculate IS for generated images
    is_mean, is_std = calculate_inception_score(GENERATED_IMAGES_PATH)
    print(f"Inception Score: {is_mean} ± {is_std}")
    
    # Calculate FID between real and generated images
    fid_score = calculate_fid(REAL_IMAGES_PATH, GENERATED_IMAGES_PATH)
    print(f"FID Score: {fid_score}")

#GPT4
