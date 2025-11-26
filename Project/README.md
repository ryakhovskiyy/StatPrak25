# Comparative Analysis of Autoencoders: AE vs VAE

**Research Project | Coursework 2025**
**Author:** Aleksey Ryakhovskiy, MSU Faculty of Mechanics and Mathematics.

## 📄 Overview
This repository contains the implementation and comparative analysis of two deep generative models:
1.  **Autoencoder (AE)** — deterministic approach.
2.  **Variational Autoencoder (VAE)** — probabilistic approach.

The project investigates the structure of the Latent Space and the models' ability to generate new data samples.

🔗 **[Read Full Report (PDF)](./Курсовая_работа.pdf)** — *Contains detailed mathematical proofs (ELBO derivation), loss function analysis, and visualizations.*

## 🛠 Key Features & Implementation
* **Framework:** `PyTorch`
* **Architecture:** Convolutional Neural Networks (Conv2d / ConvTranspose2d).
* **Optimization:**
    * Implemented **Reparameterization Trick** for VAE backpropagation.
    * Custom Loss Function: `BCE Loss` (Reconstruction) + `KLD` (Regularization).

## 📊 Experiments & Results
The models were trained on the **MNIST** dataset.

### 1. Reconstruction Quality
Comparison of original images vs. reconstructed outputs shows that while AE produces slightly sharper images, VAE captures the global structure better due to regularization.

### 2. Latent Space Visualization
* **AE:** Shows "holes" in the latent space, leading to poor generation of new samples.
* **VAE:** Demonstrates a smooth, continuous latent manifold (Gaussian distribution), allowing for high-quality interpolation between classes.

### 3. Generative Capabilities
Implemented sampling from the latent space $z \sim N(0, I)$ to generate novel digits that do not exist in the training set.

## 📂 Structure
* `IntroToVAE.ipynb` — Source code for model training and visualization.
* `Курсовая_работа.pdf` — Comprehensive academic report.
* `models.pth` — Saved weights (checkpoints).
