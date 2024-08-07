# Stable Diffusion from Scratch

This project implements a Stable Diffusion model from scratch, based on the Denoising Diffusion Probabilistic Models (DDPM) paper. The goal is to provide a deep understanding of the inner workings of Stable Diffusion by building each component from the ground up.

## Project Structure

The project is organized into several key components:

- `encoder.py`: Implementation of the VAE Encoder
- `decoder.py`: Implementation of the VAE Decoder
- `clip.py`: CLIP (Contrastive Language-Image Pre-training) model implementation
- `attention.py`: Self-attention mechanism used in various parts of the model
- `diffusion.py`: U-Net architecture for the diffusion process (work in progress)

## Components

### VAE Encoder (`encoder.py`)
The VAE Encoder reduces the dimensionality of the input images, mapping them to a latent space representation.

### VAE Decoder (`decoder.py`)
The VAE Decoder reconstructs images from the latent space representations.

### CLIP (`clip.py`)
The CLIP model provides text-to-image and image-to-text embeddings, allowing for text-guided image generation.

### Attention (`attention.py`)
Implements the self-attention mechanism used throughout the model to capture long-range dependencies.

### Diffusion (U-Net) (`diffusion.py`)
The U-Net architecture is the core of the diffusion process, responsible for learning the noise prediction. This component is currently under development.

## Upcoming Work

- Finish implementing the U-Net architecture in `diffusion.py`
- Develop the pipeline to integrate all components
- Implement the scheduler based on the DDPM paper
- Create the inference pipeline for generating images

## Installation

TO-DO:Include installation instructions once the project is more complete

## Usage

TO-DO: Provide usage examples and instructions once the model is functional

## Contributing

Contributions and suggestions are welcome. Please open an issue or submit a pull request if you'd like to contribute.

## License

## Acknowledgements

This project is inspired by and based on the following research:
- Denoising Diffusion Probabilistic Models (DDPM) paper
- Stable Diffusion
- @Umar Jamil 
