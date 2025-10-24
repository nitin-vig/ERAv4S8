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
| Batch Size | 256 |
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

---
Logs:
## Logs
EPOCH: 1
Loss=3.869758129119873 Batch_id=195 Accuracy=7.47: 100%|██████████| 196/196 [01:33<00:00,  2.10it/s]
Test set: Average loss: 3.8226 ,Top-1 Accuracy: 1278/10000 (12.78%)%

EPOCH: 2
Loss=3.635023832321167 Batch_id=195 Accuracy=16.71: 100%|██████████| 196/196 [01:33<00:00,  2.10it/s]
Test set: Average loss: 3.2060 ,Top-1 Accuracy: 2102/10000 (21.02%)%

EPOCH: 3
Loss=3.32423996925354 Batch_id=195 Accuracy=24.83: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 2.8436 ,Top-1 Accuracy: 2791/10000 (27.91%)%

EPOCH: 4
Loss=2.894665479660034 Batch_id=195 Accuracy=32.11: 100%|██████████| 196/196 [01:33<00:00,  2.10it/s]
Test set: Average loss: 2.5371 ,Top-1 Accuracy: 3427/10000 (34.27%)%

EPOCH: 5
Loss=2.6400256156921387 Batch_id=195 Accuracy=38.02: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 2.3055 ,Top-1 Accuracy: 3994/10000 (39.94%)%

EPOCH: 6
Loss=2.5170180797576904 Batch_id=195 Accuracy=43.46: 100%|██████████| 196/196 [01:33<00:00,  2.10it/s]
Test set: Average loss: 2.0368 ,Top-1 Accuracy: 4539/10000 (45.39%)%

EPOCH: 7
Loss=2.4625425338745117 Batch_id=195 Accuracy=47.25: 100%|██████████| 196/196 [01:33<00:00,  2.10it/s]
Test set: Average loss: 1.9502 ,Top-1 Accuracy: 4757/10000 (47.57%)%

EPOCH: 8
Loss=2.2403564453125 Batch_id=195 Accuracy=51.44: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 4.8463 ,Top-1 Accuracy: 3889/10000 (38.89%)%

EPOCH: 9
Loss=2.1723709106445312 Batch_id=195 Accuracy=54.00: 100%|██████████| 196/196 [01:33<00:00,  2.10it/s]
Test set: Average loss: 1.6968 ,Top-1 Accuracy: 5376/10000 (53.76%)%

EPOCH: 10
Loss=2.1376733779907227 Batch_id=195 Accuracy=56.92: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.6019 ,Top-1 Accuracy: 5649/10000 (56.49%)%

EPOCH: 11
Loss=1.9623136520385742 Batch_id=195 Accuracy=60.00: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.5650 ,Top-1 Accuracy: 5798/10000 (57.98%)%

EPOCH: 12
Loss=1.9615793228149414 Batch_id=195 Accuracy=61.72: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.4646 ,Top-1 Accuracy: 6009/10000 (60.09%)%

EPOCH: 13
Loss=1.94589102268219 Batch_id=195 Accuracy=63.79: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.5265 ,Top-1 Accuracy: 5949/10000 (59.49%)%

EPOCH: 14
Loss=1.9459686279296875 Batch_id=195 Accuracy=64.67: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 5.7199 ,Top-1 Accuracy: 4736/10000 (47.36%)%

EPOCH: 15
Loss=1.9373314380645752 Batch_id=195 Accuracy=66.59: 100%|██████████| 196/196 [01:34<00:00,  2.07it/s]
Test set: Average loss: 1.4132 ,Top-1 Accuracy: 6220/10000 (62.20%)%

EPOCH: 16
Loss=2.0476608276367188 Batch_id=195 Accuracy=68.57: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.4181 ,Top-1 Accuracy: 6244/10000 (62.44%)%

EPOCH: 17
Loss=1.688154935836792 Batch_id=195 Accuracy=70.23: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.3082 ,Top-1 Accuracy: 6472/10000 (64.72%)%

EPOCH: 18
Loss=1.510732650756836 Batch_id=195 Accuracy=71.43: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.3150 ,Top-1 Accuracy: 6466/10000 (64.66%)%

EPOCH: 19
Loss=1.6763197183609009 Batch_id=195 Accuracy=72.75: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.2663 ,Top-1 Accuracy: 6512/10000 (65.12%)%

EPOCH: 20
Loss=1.612204670906067 Batch_id=195 Accuracy=74.13: 100%|██████████| 196/196 [01:34<00:00,  2.07it/s]
Test set: Average loss: 1.2844 ,Top-1 Accuracy: 6514/10000 (65.14%)%

EPOCH: 21
Loss=1.557685136795044 Batch_id=195 Accuracy=75.57: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.4036 ,Top-1 Accuracy: 6414/10000 (64.14%)%

EPOCH: 22
Loss=1.6112279891967773 Batch_id=195 Accuracy=74.40: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.2777 ,Top-1 Accuracy: 6634/10000 (66.34%)%

EPOCH: 23
Loss=1.600617527961731 Batch_id=195 Accuracy=76.41: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.2556 ,Top-1 Accuracy: 6670/10000 (66.70%)%

EPOCH: 24
Loss=1.4813865423202515 Batch_id=195 Accuracy=78.80: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.2145 ,Top-1 Accuracy: 6749/10000 (67.49%)%

EPOCH: 25
Loss=1.4409700632095337 Batch_id=195 Accuracy=80.15: 100%|██████████| 196/196 [01:33<00:00,  2.09it/s]
Test set: Average loss: 1.2638 ,Top-1 Accuracy: 6727/10000 (67.27%)%

EPOCH: 26
Loss=1.3914510011672974 Batch_id=195 Accuracy=80.89: 100%|██████████| 196/196 [01:35<00:00,  2.05it/s]
Test set: Average loss: 1.1936 ,Top-1 Accuracy: 6885/10000 (68.85%)%

EPOCH: 27
Loss=1.1919128894805908 Batch_id=195 Accuracy=81.84: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.2262 ,Top-1 Accuracy: 6795/10000 (67.95%)%

EPOCH: 28
Loss=1.3154962062835693 Batch_id=195 Accuracy=83.19: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.2124 ,Top-1 Accuracy: 6913/10000 (69.13%)%

EPOCH: 29
Loss=1.3670666217803955 Batch_id=195 Accuracy=83.84: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.2626 ,Top-1 Accuracy: 6799/10000 (67.99%)%

EPOCH: 30
Loss=1.3260791301727295 Batch_id=195 Accuracy=84.01: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.2024 ,Top-1 Accuracy: 6922/10000 (69.22%)%

EPOCH: 31
Loss=1.0704212188720703 Batch_id=195 Accuracy=88.74: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.0906 ,Top-1 Accuracy: 7215/10000 (72.15%)%

EPOCH: 32
Loss=1.0833816528320312 Batch_id=195 Accuracy=89.99: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.0997 ,Top-1 Accuracy: 7252/10000 (72.52%)%

EPOCH: 33
Loss=1.00822114944458 Batch_id=195 Accuracy=90.52: 100%|██████████| 196/196 [01:35<00:00,  2.05it/s]
Test set: Average loss: 1.1272 ,Top-1 Accuracy: 7217/10000 (72.17%)%

EPOCH: 34
Loss=1.045372486114502 Batch_id=195 Accuracy=91.14: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1053 ,Top-1 Accuracy: 7295/10000 (72.95%)%

EPOCH: 35
Loss=1.0840001106262207 Batch_id=195 Accuracy=91.27: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1400 ,Top-1 Accuracy: 7217/10000 (72.17%)%

EPOCH: 36
Loss=1.0881147384643555 Batch_id=195 Accuracy=92.38: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.0797 ,Top-1 Accuracy: 7342/10000 (73.42%)%

EPOCH: 37
Loss=0.987473726272583 Batch_id=195 Accuracy=92.79: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.0832 ,Top-1 Accuracy: 7377/10000 (73.77%)%

EPOCH: 38
Loss=0.9902850985527039 Batch_id=195 Accuracy=93.05: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.0965 ,Top-1 Accuracy: 7395/10000 (73.95%)%

EPOCH: 39
Loss=1.0126115083694458 Batch_id=195 Accuracy=93.16: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1034 ,Top-1 Accuracy: 7344/10000 (73.44%)%

EPOCH: 40
Loss=1.0198535919189453 Batch_id=195 Accuracy=93.30: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1082 ,Top-1 Accuracy: 7345/10000 (73.45%)%

EPOCH: 41
Loss=0.9365944266319275 Batch_id=195 Accuracy=93.69: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.0980 ,Top-1 Accuracy: 7395/10000 (73.95%)%

EPOCH: 42
Loss=0.9793956279754639 Batch_id=195 Accuracy=93.93: 100%|██████████| 196/196 [01:36<00:00,  2.03it/s]
Test set: Average loss: 1.1016 ,Top-1 Accuracy: 7379/10000 (73.79%)%

EPOCH: 43
Loss=0.9550695419311523 Batch_id=195 Accuracy=94.02: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1149 ,Top-1 Accuracy: 7371/10000 (73.71%)%

EPOCH: 44
Loss=0.9819129705429077 Batch_id=195 Accuracy=93.93: 100%|██████████| 196/196 [01:34<00:00,  2.07it/s]
Test set: Average loss: 1.1141 ,Top-1 Accuracy: 7398/10000 (73.98%)%

EPOCH: 45
Loss=1.0116387605667114 Batch_id=195 Accuracy=94.09: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1086 ,Top-1 Accuracy: 7412/10000 (74.12%)%

EPOCH: 46
Loss=0.9274983406066895 Batch_id=195 Accuracy=93.91: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1167 ,Top-1 Accuracy: 7389/10000 (73.89%)%

EPOCH: 47
Loss=0.9243737459182739 Batch_id=195 Accuracy=94.25: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1177 ,Top-1 Accuracy: 7384/10000 (73.84%)%

EPOCH: 48
Loss=0.9691476225852966 Batch_id=195 Accuracy=94.39: 100%|██████████| 196/196 [01:34<00:00,  2.07it/s]
Test set: Average loss: 1.1245 ,Top-1 Accuracy: 7409/10000 (74.09%)%

EPOCH: 49
Loss=0.8690693974494934 Batch_id=195 Accuracy=94.24: 100%|██████████| 196/196 [01:34<00:00,  2.08it/s]
Test set: Average loss: 1.1152 ,Top-1 Accuracy: 7406/10000 (74.06%)%

EPOCH: 50
Loss=0.9070953726768494 Batch_id=195 Accuracy=94.41: 100%|██████████| 196/196 [01:34<00:00,  2.07it/s]
Test set: Average loss: 1.1202 ,Top-1 Accuracy: 7408/10000 (74.08%)%

