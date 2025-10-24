# CIFAR-100 Classification using ResNet50 (Trained from Scratch)

This project trains a **ResNet50 model from scratch** on the **CIFAR-100** dataset to explore how **learning rate strategies** impact model convergence, training stability, and validation performance. Here is the [notebook and logs](https://github.com/nitin-vig/ERAv4S8/blob/main/Cifar_100_with_Resnt50_final.ipynb). 

---

## 🧠 Model Overview

- **Architecture:** ResNet50 (trained from scratch, no pretrained weights)  
- **Input:** 32×32 RGB images (CIFAR-100)  
- **Output:** 100 classes  
- **Optimizer:** AdamW  
- **Loss Function:** CrossEntropyLoss  
- **Augmentations (Albumentations):**  
  - RandomCrop  
  - HorizontalFlip  
  - ColorJitter  
  - Normalization (using CIFAR-100 mean and std)  
  - Conversion with `ToTensorV2`

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Dataset | CIFAR-100 |
| Input Size | 32 × 32 |
| Batch Size | 128 |
| Epochs | 50 |
| Optimizer | AdamW |
| Starting LR | 0.001 |
| LR Schedulers | CosineAnnealingLR, ReduceLROnPlateau, Constant |

---

## 📈 Learning Rate Experiments

The notebook focuses on comparing **different learning rate schedulers** and how they influence convergence and final accuracy when training a deep model from scratch.

### Learning Rate Strategies Tested

#### 1. Constant LR (Baseline)
- LR = 0.001 fixed throughout training.  
- Model converged slowly and validation accuracy plateaued early (~60%).  

#### 2. Cosine Annealing
- Smoothly decays LR following a cosine curve.  
- May have given better results over longer training cycle.It was stuck around 66% for 50 epochs. 

#### 3. ReduceLROnPlateau
- Dynamically reduces LR when validation accuracy stops improving.  
- Helped recover from stagnation for faster training. (~73% in 36 epochs)  
---

## 🧩 Key Learnings on Optimizer,scheduler

- Weight decay was added in Adam for Regularization but it didnt help much and accuracy was stuck around 65
- Switched to AdamW improved the accuracy.
- Tried Cosine Annealing but it seems that 50 epcohs was not the right window.
- Cosine Annealing uses a predefined curve so switched to ReduceLROnPlateau to make it adaptive.


---

## 📊 Results Summary

| Scheduler | Validation Accuracy | Convergence Speed | Notes |
|------------|---------------------|-------------------|-------|
| Constant LR | ~67% | Slow | Early plateau |
| Cosine Annealing | ~67% | smooth |plateaud in 50 epochs cycle |May have worked better on 100 epochs
| ReduceLROnPlateau | ~73–74% | Fast | Recovers after plateau |
---

## 🚀 Future Work

- Model overfitting improved a bit after Augumentation but can be improved further with more aggresive augumentation, Dropout 
- Experiment with **OneCycleLR** or **Warmup Cosine Annealing**.  
- Add **MixUp** for better regularization.  
- Try **gradient clipping** for stability with higher LRs.  
- Explore smaller ResNet variants (e.g., ResNet18) for faster training.  

---

## 🧩 Dependencies

```bash
torch
torchvision
albumentations
matplotlib
numpy
tqdm

