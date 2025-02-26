# Image Caption Model

## Overview
The `ImageCaptionModel` is a PyTorch-based neural network designed for generating captions for images. It leverages a transformer architecture to process encoded images and generate descriptive text.

## Architecture
- **Positional Encoding**: Utilizes positional encoding to maintain the order of input sequences.
- **Transformer Decoder**: Comprises multiple layers of transformer decoders to generate captions based on the encoded image features.
- **Embedding Layer**: Maps vocabulary indices to dense vectors for input into the transformer.

## Installation
To use this model, ensure you have the following dependencies installed:
- PyTorch
- NumPy
- Matplotlib
- Other required libraries as specified in your project.

## Usage
1. **Initialization**:
   ```python
   model = ImageCaptionModel(n_head=8, n_decoder_layer=6, vocab_size=10000, embedding_size=512)
   ```

2. **Forward Pass**:
   ```python
   output, pad_mask = model(encoded_image, decoder_input)
   ```

## Methods
- `__init__(self, n_head, n_decoder_layer, vocab_size, embedding_size)`: Initializes the model with specified parameters.
- `init_weights(self)`: Initializes weights for the embedding and linear layers.
- `generate_Mask(self, size, decoder_inp)`: Generates masks for the decoder input.
- `forward(self, encoded_image, decoder_inp)`: Performs the forward pass of the model.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by the transformer architecture and its applications in natural language processing and computer vision.
