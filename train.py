import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from pathlib import Path

# def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_length,  device):
#     sos_idx = tokenizer_src.token_to_id("[SOS]")
#     eos_idx = tokenizer_tgt.token_to_id("[EOS]")

#     encoder_out = model.encode(src, src_mask)

#     decoder_input = torch.empty(1,1).fill_(sos_idx).type(src.dtype).to(device)
#     while True:
#         if decoder_input.size(1) == max_length:
#             break

#         decoder_mask = causal_mask(decoder_input.size(1).type_as(src_mask).to(device))

#         out = model.decode( encoder_out, src_mask, decoder_input, decoder_mask)

#         prob = model.project(out[:, -1])

#         _, next_word = torch.max(prob, dim=1)
#         decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(src).fill_(next_word.item()).to(device)], dim=1)

#         if next_word.item() == eos_idx:
#             break

#     return decoder_input.squeeze(0)

def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_length,  device):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_out = model.encode(src, src_mask)

    decoder_input = torch.full((1, 1), sos_idx, dtype=torch.long, device=device)  # ensure it's long
    while True:
        if decoder_input.size(1) == max_length:
            break

        decoder_mask = causal_mask(torch.tensor(decoder_input.size(1), device=device).type_as(src_mask))

        out = model.decode(encoder_out, src_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()  # get the scalar value

        decoder_input = torch.cat([decoder_input, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)  # ensure it's long

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt,max_length, device, print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = 80

    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch["enc_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.shape[0] == 1, "Batch size should be 1 for validation"

            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_length, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text) 
            predicted.append(model_out_text)

            print_msg('-'*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Target: {target_text}")
            print_msg(f"Predicted: {model_out_text}")
            
            if count == num_examples:
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(dataset, language):
    """
    Generates an iterator that yields all the sentences from a given dataset in a specific language.

    Parameters:
        dataset (iterable): An iterable containing the dataset.
        language (str): The language of the sentences to be yielded.

    Yields:
        str: The next sentence in the dataset in the specified language.
    """
    for item in dataset:
        yield item['translation'][language]

def get_tokenizer(config, dataset, language):
    """
    Generates a tokenizer for a given dataset and language.

    Args:
        config (dict): The configuration dictionary containing the tokenizer file path.
        dataset (iterable): The dataset to generate the tokenizer from.
        language (str): The language of the dataset.

    Returns:
        Tokenizer: The generated tokenizer.

    Description:
        This function generates a tokenizer for a given dataset and language. It first checks if the tokenizer file exists. If it does not exist, it creates a new tokenizer using the WordLevel tokenizer model with the unk_token set to "[UNK]". It then sets the pre_tokenizer to Whitespace. It trains the tokenizer using the get_all_sentences function on the dataset and language. Finally, it saves the tokenizer to the tokenizer file path. If the tokenizer file exists, it loads the tokenizer from the file path.

    Note:
        The get_all_sentences function is assumed to be defined elsewhere in the codebase.
    """
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]", "[SOS]", "[EOS]"], min_frequency=2, show_progress=True)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset(config):
    """
    Generates a dataset for training and validation based on the given configuration.

    Parameters:
        config (dict): The configuration dictionary containing the following keys:
            - src_language (str): The source language.
            - target_language (str): The target language.
            - seq_length (int): The maximum sequence length.
            - batch_size (int): The batch size for the dataloader.

    Returns:
        tuple: A tuple containing the following:
            - train_dataloader (torch.utils.data.DataLoader): The dataloader for the training dataset.
            - validation_dataloader (torch.utils.data.DataLoader): The dataloader for the validation dataset.
            - tokenizer_src (Tokenizer): The tokenizer for the source language.
            - tokenizer_tgt (Tokenizer): The tokenizer for the target language.

    """
    dataset_raw = load_dataset('opus_books', f'{config["src_language"]}-{config["target_language"]}', split = 'train')

    # build tokenizer
    tokenizer_src = get_tokenizer(config, dataset_raw, config["src_language"])
    tokenizer_tgt = get_tokenizer(config, dataset_raw, config["target_language"])

    # 90% of the data will be used for training and the rest for validation
    train_dataset_size = int(len(dataset_raw) * 0.9)
    validation_dataset_size = len(dataset_raw) - train_dataset_size

    train_dataset_raw, validation_dataset_raw = random_split(dataset_raw, [train_dataset_size, validation_dataset_size])

    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config["src_language"], config["target_language"], config["seq_length"])
    validation_dataset = BilingualDataset(validation_dataset_raw, tokenizer_src, tokenizer_tgt, config["src_language"], config["target_language"], config["seq_length"])

    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config["src_language"]]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config["target_language"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max len src: {max_len_src}")
    print(f"Max len tgt: {max_len_tgt}")

    # config["max_len_src"] = max_len_src
    # config["max_len_tgt"] = max_len_tgt

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    """
    Builds and returns a transformer model based on the given configuration and vocabulary sizes.

    Parameters:
        config (dict): A dictionary containing the configuration parameters for building the model.
        vocab_src_len (int): The size of the source vocabulary.
        vocab_tgt_len (int): The size of the target vocabulary.

    Returns:
        torch.nn.Module: The built transformer model.

    """
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_length'], config['seq_length'], d_model=config['d_model'])
    return model


def train_model(config):
    """
    Trains a transformer model using the given configuration.

    Parameters:
        config (dict): A dictionary containing the configuration parameters for training the model.

    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)  # Move model to device

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload'] :
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing= 0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()

        batch_iterator = tqdm(train_dataloader, desc= f'Processing epoch {epoch:02d}')

        for batch in batch_iterator:
            encoder_input = batch['enc_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['dec_input'].to(device) # (batch_size, seq_len)

            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode( decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, vocab_size)
            projection_output = model.project(decoder_output) # (batch_size, seq_len, tgt vocab_size)

            label = batch['label'].to(device) # (batch_size, seq_len)
            loss = loss_fn(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            

            global_step += 1
        
        run_validation(model, validation_dataloader, tokenizer_src ,tokenizer_tgt, config['seq_length'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)