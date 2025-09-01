import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
from dataclasses_json import dataclass_json
from datasets import IterableDataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

from bacformer.modeling.config import SPECIAL_TOKENS_DICT

# masking strategy inspired by BERT where we mask 80%, leave 10% unchanged and replace 10% with random tokens
MASKING_STRATEGY = {
    "mask": 0.875,
    "no_change": 0.125,
}


@dataclass_json
@dataclass
class DataReaderOutput:
    """A dataclass for storing the output of the data reader."""

    train_dataset: IterableDataset = None
    val_dataset: IterableDataset = None
    test_dataset: IterableDataset = None


def mask_prot_embs(
    special_tokens_mask: np.ndarray,
    labels: np.ndarray,
    mgm_probability: float,
    prot_emb_token_id: int,
    mask_token_id: int,
    masking_strategy: dict[str, float] = MASKING_STRATEGY,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask protein embeddings with the given probability."""
    if mgm_probability == 0.0:
        return special_tokens_mask, labels

    mask_labels = np.ones_like(labels) * -100
    indices = np.where(special_tokens_mask == prot_emb_token_id)[0]

    # replace tokens with mask token
    n_prots_to_mask = int(mgm_probability * len(indices) * masking_strategy["mask"])
    # randomly sample proteins to mask
    indices_to_mask = np.random.choice(indices, n_prots_to_mask, replace=False)
    # replace the selected protein embeddings with the mask token
    special_tokens_mask[indices_to_mask] = mask_token_id
    mask_labels[indices_to_mask] = labels[indices_to_mask]

    # leave some tokens unchanged
    n_prots_no_change = int(mgm_probability * len(indices) * masking_strategy["no_change"])
    indices_no_change = np.random.choice(indices, n_prots_no_change, replace=False)
    mask_labels[indices_no_change] = labels[indices_no_change]

    return special_tokens_mask, mask_labels


def transform_sample(
    mgm_probability: float = 0.0,
    special_tokens_dict: dict[str, int] = SPECIAL_TOKENS_DICT,
    max_n_proteins: int = 6000,
    max_n_contigs: int = 1000,
    end_token_id: int = 20000,
    embeddings_col_name: str = "protein_embeddings",
    prot_cluster_id_col_name: str = "prot_cluster_id",
    sample: dict[str, Any] = None,
):
    """Transform sample for the model input for ESMC.

    Args:
        mgm_probability (float): Probability of masking protein embeddings.
        special_tokens_dict (dict[str, int]): Dictionary of special tokens.
        max_n_proteins (int): Maximum number of proteins.
        max_n_contigs (int): Maximum number of contigs.
        end_token_id (int): ID for the end token.
        sample (dict[str, Any]): Sample containing protein embeddings and indices.

    Returns
    -------
        dict[str, Any]: Processed sample with protein embeddings, special tokens mask, token type IDs, labels, and attention mask.
    """
    pad_emb = np.zeros(len(sample[embeddings_col_name][0]), dtype=np.float32)
    # add CLS token
    labels = [-100]
    special_tokens_mask = [special_tokens_dict["CLS"]]
    protein_embeddings = [pad_emb]
    attention_mask = [1.0]
    token_type_ids = [0]

    if "contig_idx" not in sample:
        sample["contig_idx"] = np.ones_like(sample[embeddings_col_name])

    if prot_cluster_id_col_name not in sample:
        sample[prot_cluster_id_col_name] = [-100] * len(sample[embeddings_col_name])
    curr_contig_idx = 0
    for prot_emb, label, cidx in zip(
        sample[embeddings_col_name], sample[prot_cluster_id_col_name], sample["contig_idx"], strict=False
    ):
        # if the contig index is greater than the max number of contigs, set it to the max
        cidx = min(cidx, max_n_contigs - 1)
        if cidx != curr_contig_idx:
            # add a separator token for the previous contig
            special_tokens_mask.append(special_tokens_dict["SEP"])
            protein_embeddings.append(pad_emb)
            token_type_ids.append(curr_contig_idx)
            labels.append(-100)
            attention_mask.append(1.0)
        special_tokens_mask.append(special_tokens_dict["PROT_EMB"])
        labels.append(label)
        protein_embeddings.append(prot_emb)
        token_type_ids.append(cidx)
        attention_mask.append(1.0)
        curr_contig_idx = cidx

    # add the last separator token
    special_tokens_mask.append(special_tokens_dict["SEP"])
    protein_embeddings.append(pad_emb)
    token_type_ids.append(curr_contig_idx)
    labels.append(-100)
    attention_mask.append(1.0)

    # add END token
    protein_embeddings = protein_embeddings[: max_n_proteins - 1] + [pad_emb]
    special_tokens_mask = special_tokens_mask[: max_n_proteins - 1] + [special_tokens_dict["END"]]
    token_type_ids = token_type_ids[: max_n_proteins - 1] + [curr_contig_idx]
    labels = labels[: max_n_proteins - 1] + [end_token_id]
    attention_mask = attention_mask[: max_n_proteins - 1] + [1.0]

    # do the masking
    special_tokens_mask, labels = mask_prot_embs(
        special_tokens_mask=np.array(special_tokens_mask, dtype=np.int64),
        labels=np.array(labels, dtype=np.int64),
        mgm_probability=mgm_probability,
        prot_emb_token_id=special_tokens_dict["PROT_EMB"],
        mask_token_id=special_tokens_dict["MASK"],
    )

    return {
        "prot_embeddings": protein_embeddings,
        "special_tokens_mask": special_tokens_mask,
        "token_type_ids": np.array(token_type_ids, dtype=np.int64),
        "labels": labels,
        "attention_mask": np.array(attention_mask, dtype=np.float32),
    }


def fetch_training_data(
    input_dir: str,
    mgm_probability: float,
    max_n_proteins: int,
    max_n_contigs: int,
    end_token_idx: int = 50000,
    embeddings_col_name: str = "protein_embeddings",
    indices_col_name: str = "prot_cluster_idx",
    test: bool = False,
    random_state: int = 42,
):
    """A function which orchestrates getting the data.

    The pretraining data is stored in chunks of parquet files. The function reads the parquet files and returns
    IterableDataset objects for training, validation, and test data.
    """
    # get train data from SPIRE and MGNify
    set_seed(random_state)
    # get train files
    train_files = [
        os.path.join(input_dir, "train", f)
        for f in os.listdir(os.path.join(input_dir, "train"))
        if f.endswith("parquet")
    ]
    # shuffle the files
    random.shuffle(train_files)
    data_files = {
        "train": train_files,
        "validation": [
            os.path.join(input_dir, "val", f)
            for f in os.listdir(os.path.join(input_dir, "val"))
            if f.endswith("parquet")
        ],
    }
    transform_fn = partial(
        transform_sample,
        mgm_probability,
        SPECIAL_TOKENS_DICT,
        max_n_proteins,
        max_n_contigs,
        end_token_idx,
        embeddings_col_name,
        indices_col_name,
    )
    cols = ["contig_idx", embeddings_col_name, indices_col_name]
    train_dataset = (
        load_dataset("parquet", data_files=data_files, split="train", streaming=True)
        .select_columns(cols)
        .map(transform_fn, batched=False, with_indices=False, remove_columns=cols)
    )
    val_dataset = (
        load_dataset("parquet", data_files=data_files, split="validation", streaming=True)
        .select_columns(cols)
        .map(transform_fn, batched=False, with_indices=False, remove_columns=cols)
    )

    if not test:
        return DataReaderOutput(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    data_files = {
        "test": [
            os.path.join(input_dir, "test", f)
            for f in os.listdir(os.path.join(input_dir, "test"))
            if f.endswith("parquet")
        ]
    }
    test_dataset = (
        load_dataset("parquet", data_files=data_files, split="test", streaming=True)
        .select_columns(cols)
        .map(transform_fn, batched=False, with_indices=False, remove_columns=cols)
    )
    return DataReaderOutput(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
    )


def collate_genome_samples(
    pad_token_id: int = 0,
    max_n_contigs: int = 1000,
    samples: list[dict] = None,
) -> dict[str, torch.Tensor]:
    """Collate function for GenomeSample."""
    prot_emb = pad_sequence(
        [torch.tensor(sample["prot_embeddings"], dtype=torch.float32) for sample in samples],
        batch_first=True,
        padding_value=pad_token_id,
    )
    special_tokens_mask = pad_sequence(
        [torch.tensor(sample["special_tokens_mask"], dtype=torch.long) for sample in samples],
        batch_first=True,
        padding_value=pad_token_id,
    )
    token_type_ids = pad_sequence(
        [torch.tensor(sample["token_type_ids"], dtype=torch.long) for sample in samples],
        batch_first=True,
        padding_value=max_n_contigs,
    )

    if "labels" in samples[0]:
        labels = pad_sequence(
            [torch.tensor(sample["labels"], dtype=torch.long) for sample in samples],
            batch_first=True,
            padding_value=-100,
        )
    else:
        labels = torch.tensor([])

    output = {
        "protein_embeddings": prot_emb,
        "special_tokens_mask": special_tokens_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }
    if "attention_mask" in samples[0]:
        padding_mask = pad_sequence(
            [torch.tensor(sample["attention_mask"], dtype=torch.float32) for sample in samples],
            batch_first=True,
            padding_value=pad_token_id,
        )
        output["attention_mask"] = padding_mask

    return output
