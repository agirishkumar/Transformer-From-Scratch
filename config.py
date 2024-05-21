from pathlib import Path
def get_config():
    """
    Returns a dictionary containing the configuration parameters for the model.
    
    The dictionary contains the following keys:
    - "batch_size": The size of each batch for training.
    - "num_epochs": The number of epochs to train the model.
    - "lr": The learning rate for the optimizer.
    - "seq_length": The maximum sequence length for the input data.
    - "d_model": The dimension of the model.
    - "datasource": The name of the data source.
    - "src_language": The source language.
    - "target_language": The target language.
    - "model_folder": The folder where the model weights are stored.
    - "model_basename": The basename of the model file.
    - "preload": The name of the preloaded model weights.
    - "tokenizer_file": The format string for the tokenizer file.
    - "experiment_name": The name of the experiment for TensorBoard logging.
    
    Returns:
    - dict: A dictionary containing the configuration parameters.
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 0.0001,
        "seq_length": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "src_language": "en",
        "target_language": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
        }

def get_weights_file_path(config, epoch: str):
    """
    Returns the file path for the weights file of a model based on the given configuration and epoch.

    Parameters:
        config (dict): A dictionary containing the configuration parameters for the model.
        epoch (str): The epoch for which the weights file is requested.

    Returns:
        str: The file path for the weights file.

    Example:
        >>> config = {"model_folder": "weights", "model_basename": "model_"}
        >>> get_weights_file_path(config, "1")
        'weights/model_1.pt'
    """
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)

# Find the latest weights file in the weights folder
# def latest_weights_file_path(config):
#     """
#     Returns the file path of the latest weights file for a model based on the given configuration.

#     Parameters:
#         config (dict): A dictionary containing the configuration parameters for the model.
#             - 'datasource' (str): The name of the data source.
#             - 'model_folder' (str): The folder where the model weights are stored.
#             - 'model_basename' (str): The basename of the model file.

#     Returns:
#         str or None: The file path of the latest weights file, or None if no weights file is found.

#     Example:
#         >>> config = {
#         ...     'datasource': 'opus_books',
#         ...     'model_folder': 'weights',
#         ...     'model_basename': 'model_'
#         ... }
#         >>> latest_weights_file_path(config)
#         'weights/model_1.pt'
#     """
#     model_folder = f"{config['datasource']}_{config['model_folder']}"
#     model_filename = f"{config['model_basename']}*"
#     weights_files = list(Path(model_folder).glob(model_filename))
#     if len(weights_files) == 0:
#         return None
#     weights_files.sort()
#     return str(weights_files[-1])

def latest_weights_file_path(config):
    model_folder = Path(config['model_folder'])
    weights_files = list(model_folder.glob(f"{config['model_basename']}*.pt"))
    if not weights_files:
        return None
    weights_files.sort()
    return str(weights_files[-1])

