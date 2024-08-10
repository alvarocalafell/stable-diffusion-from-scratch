# Stable Diffusion from Scratch

Welcome to the Stable Diffusion from Scratch project! This repository implements a Stable Diffusion model using PyTorch, offering both text-to-image and image-to-image capabilities. As an AI researcher exploring new frontiers, I've built this project to understand of the inner workings of Stable Diffusion by constructing each component from the ground up.

## 🚀 Features

- Text-to-image generation
- Image-to-image transformation
- Built entirely with PyTorch
- Compatible with Google Colab's free T4 GPUs
- Utilizes pre-trained weights from Hugging Face's v1-5 Stable Diffusion model

## 🧠 Project Structure

The project is organized into several key components:

- `encoder.py`: Implementation of the VAE Encoder
- `decoder.py`: Implementation of the VAE Decoder
- `clip.py`: CLIP (Contrastive Language-Image Pre-training) model implementation
- `attention.py`: Self-attention mechanism used in various parts of the model
- `diffusion.py`: U-Net architecture for the diffusion process
- `pipeline.py`: Integration of all components for the generation pipeline
- `ddpm.py`: Implementation of the Denoising Diffusion Probabilistic Model
- `model_loader.py`: Utility for loading pre-trained weights
- `model_converter.py`: Tool for mapping layer names to the imported pre-trained model

## 🎨 Components

### VAE Encoder (`encoder.py`)
Reduces the dimensionality of input images, mapping them to a latent space representation.

### VAE Decoder (`decoder.py`)
Reconstructs images from the latent space representations.

### CLIP (`clip.py`)
Provides text-to-image and image-to-text embeddings, enabling text-guided image generation.

### Attention (`attention.py`)
Implements the self-attention mechanism used throughout the model to capture long-range dependencies.

### Diffusion (U-Net) (`diffusion.py`)
The core of the diffusion process, responsible for learning noise prediction.

### Pipeline (`pipeline.py`)
Integrates all components for a seamless generation process.

### DDPM Sampler (`ddpm.py`)
Implements the Denoising Diffusion Probabilistic Model for the sampling process.

## 🛠️ Installation

```
git clone https://github.com/yourusername/stable-diffusion-from-scratch.git
cd stable-diffusion-from-scratch
pip install -r requirements.txt
```

## 🚂 Usage
To generate images using the model, run the demo.ipynb notebook:

`jupyter notebook demo.ipynb`

This notebook can be run on Google Colab with free T4 GPUs for those wanting to try it out without local GPU resources.

## 🤝 Contributing
Contributions and suggestions are welcome! Please open an issue or submit a pull request if you'd like to contribute.

## 🙏 Acknowledgements
A huge thank you to @Umar Jamil, a fantastic YouTuber and researcher whose original videos and code served as the foundation for this implementation. Your work has been instrumental in bringing this project to life!
This project is inspired by and based on the following research:

- Denoising Diffusion Probabilistic Models (DDPM) paper
- Stable Diffusion
- Hugging Face's pre-trained Stable Diffusion v1-5 model

## 🔮 Future Work

- Further optimization for improved performance
- Integration with more pre-trained models
- Expansion of the image manipulation capabilities