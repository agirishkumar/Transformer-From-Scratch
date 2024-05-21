import torch
import torch.nn as nn

from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, src_language, tgt_language, seq_length):
        """
        Initializes a BilingualDataset object.

        Args:
            dataset (Dataset): The dataset object containing the source and target language pairs.
            src_tokenizer (Tokenizer): The tokenizer object for the source language.
            tgt_tokenizer (Tokenizer): The tokenizer object for the target language.
            src_language (str): The source language.
            tgt_language (str): The target language.
            seq_length (int): The maximum sequence length.

        Initializes the following attributes:
            - dataset (Dataset): The dataset object.
            - src_tokenizer (Tokenizer): The tokenizer object for the source language.
            - tgt_tokenizer (Tokenizer): The tokenizer object for the target language.
            - src_language (str): The source language.
            - tgt_language (str): The target language.
            - seq_length (int): The maximum sequence length.
            - sos_token (torch.Tensor): The start of sequence token as a tensor.
            - eos_token (torch.Tensor): The end of sequence token as a tensor.
            - pad_token (torch.Tensor): The padding token as a tensor.
        """
        super().__init__()
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_language = src_language
        self.tgt_language = tgt_language
        self.seq_length = seq_length

        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset)

    # def __getitem__(self, index):
    #     """
    #     Retrieves a single item from the dataset at the specified index.

    #     Parameters:
    #         index (int): The index of the item to retrieve.

    #     Returns:
    #         dict: A dictionary containing the following keys:
    #             - "enc_input" (torch.Tensor): The encoded input sequence with SOS and EOS tokens added, and padded to the specified sequence length.
    #             - "dec_input" (torch.Tensor): The decoder input sequence with SOS token added and padded to the specified sequence length.
    #             - "encoder_mask" (torch.Tensor): A binary mask indicating the padding tokens in the encoded input sequence.
    #             - "decoder_mask" (torch.Tensor): A binary mask indicating the padding tokens and causal dependencies in the decoder input sequence.
    #             - "label" (torch.Tensor): The target sequence with EOS token added and padded to the specified sequence length.
    #             - "src_text" (str): The source language text.
    #             - "tgt_text" (str): The target language text.

    #     Raises:
    #         ValueError: If the sequence length is too short.

    #     Note:
    #         - The encoded input sequence and decoder input sequence are both padded to the specified sequence length.
    #         - The decoder input sequence is masked to indicate causal dependencies.
    #         - The padding tokens are represented by the pad_token.
    #         - The SOS and EOS tokens are represented by the sos_token and eos_token respectively.
    #     """
    #     src_tgt_pair = self.dataset[index]
    #     src_text = src_tgt_pair['translation'][self.src_language]
    #     tgt_text = src_tgt_pair['translation'][self.tgt_language]

    #     enc_inp_tokens = self.src_tokenizer.encode(src_text).ids
    #     dec_inp_tokens = self.tgt_tokenizer.encode(tgt_text).ids

    #     print("enc_inp_tokens: ", enc_inp_tokens)
    #     print("dec_inp_tokens: ", dec_inp_tokens)

    #     enc_num_padding_tokens = self.seq_length - len(enc_inp_tokens) -2
    #     dec_num_padding_tokens = self.seq_length - len(dec_inp_tokens) -1

    #     print("enc_num_padding_tokens: ", enc_num_padding_tokens)
    #     print("dec_num_padding_tokens: ", dec_num_padding_tokens)

    #     if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
    #         raise ValueError("The sequence length is too short.")

    #     # Add SOS and EOS tokens to source text 
    #     enc_input = torch.cat(
    #         [
    #             self.sos_token,
    #             torch.tensor(enc_inp_tokens, dtype=torch.int64),
    #             self.eos_token,
    #             torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
    #         ]
    #     )

    #     # Add SOS token to decoder input text
    #     dec_input = torch.cat(
    #         [
    #             self.sos_token,
    #             torch.tensor(dec_inp_tokens, dtype=torch.int64),
    #             torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
    #         ]
    #     )

    #     # Add EOS token to label text
    #     label = torch.cat(
    #         [
    #             torch.tensor(dec_inp_tokens, dtype=torch.int64),
    #             self.eos_token,
    #             torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
    #         ]
    #     )

    #     print("enc_input size: ", enc_input.shape[0])

    #     print("dec_input size: ", dec_input.shape[0])
    #     print("seq_length: ", self.seq_length)

    #     # Sanity check
    #     assert len(enc_input) == self.seq_length, f"Expected {self.seq_length}, got {len(enc_input)}"
    #     assert len(dec_input) == self.seq_length, f"Expected {self.seq_length}, got {len(dec_input)}"
    #     assert len(label) == self.seq_length, f"Expected {self.seq_length}, got {len(label)}"

    #     return {"enc_input": enc_input, # sequence length 
    #             "dec_input": dec_input, # sequence length
    #             "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_length)
    #             "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.shape[0]), # (1, seq_length, seq_length)
    #             "label": label,
    #             "src_text": src_text,
    #             "tgt_text": tgt_text
    #             }

    def __getitem__(self, index):
        src_tgt_pair = self.dataset[index]
        src_text = src_tgt_pair['translation'][self.src_language]
        tgt_text = src_tgt_pair['translation'][self.tgt_language]

        enc_inp_tokens = self.src_tokenizer.encode(src_text).ids
        dec_inp_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        # Deduct 2 for SOS and EOS for enc_input
        enc_num_padding_tokens = self.seq_length - len(enc_inp_tokens) - 2
        # Deduct 1 for SOS in dec_input and 1 for EOS in label
        dec_num_padding_tokens = self.seq_length - len(dec_inp_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("The sequence length is too short.")

        enc_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_inp_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ])

        dec_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_inp_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ])

        label = torch.cat([
            torch.tensor(dec_inp_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ])
        # Sanity check
        # print(f"Enc Input Length: {len(enc_input)}, Dec Input Length: {len(dec_input)}, Label Length: {len(label)}")

        # assert len(enc_input) == self.seq_length, f"Expected {self.seq_length}, got {len(enc_input)}"
        # assert len(dec_input) == self.seq_length, f"Expected {self.seq_length}, got {len(dec_input)}"
        # assert len(label) == self.seq_length, f"Expected {self.seq_length}, got {len(label)}"

        return {
            "enc_input": enc_input,
            "dec_input": dec_input,
            "encoder_mask": (enc_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (dec_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input.shape[0]),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    """
    Generates a causal mask for a given size.

    Parameters:
        size (int): The size of the mask.

    Returns:
        torch.Tensor: The causal mask of shape (size, size).
    """
    mask = torch.triu(torch.ones(size, size),  diagonal = 1).type(torch.int)
    return mask