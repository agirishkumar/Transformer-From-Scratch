# Transformer from Scratch

This project is an implementation of a Transformer model from scratch, as described in the "Attention Is All You Need" paper. The primary goal is to explore the intricacies of the Transformer architecture and apply it to a neural machine translation task using the English-to-Italian subset of the Opus Books dataset from Hugging Face.
### Project Structure

    train.py: This is the main training script where the model is trained using the configurations set in config.py. It includes functions to prepare datasets, create tokenizers, and run the training loop.
    config.py: Contains the configurations for the model, such as batch size, learning rate, number of epochs, and sequence length.
    model.py: Defines the Transformer model, including the MultiHeadAttention block, the encoder and decoder layers, and the full Transformer architecture.
    dataset.py: Manages the dataset preparation, including tokenization and the creation of the DataLoader for the model training.

### Installation

*Clone the Repository*


`git clone https://github.com/your-username/Transformer-From-Scratch.git`
`cd Transformer-From-Scratch`

*Set Up Environment*
It is recommended to use a virtual environment:

`python -m venv venv`
`source venv/bin/activate`  # On Windows use `venv\Scripts\activate`

*Install Dependencies*
Install the necessary Python packages:


    `pip install -r requirements.txt`

*Usage*

To train the model, ensure you have the dataset in place and run:


`python3 train.py`

You can adjust training parameters by modifying the config.py file.
Configuration

The config.py file allows you to adjust various parameters including:

    batch_size: Number of samples in each batch.
    num_epochs: Total number of training epochs.
    lr: Learning rate for the optimizer.
    seq_length: Maximum sequence length for the input and output.
    d_model: The dimensionality of the model.

### Dataset

This project uses the English-to-Italian subset of the Opus Books dataset from Hugging Face. Ensure you have downloaded and prepared this dataset before training.
Contributing

Your contributions are welcome! If you have improvements or find bugs, please feel free to open an issue or a pull request.