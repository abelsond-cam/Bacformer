from collections import defaultdict
from itertools import combinations

import pandas as pd
import torch
from tap import Tap
from torch import sigmoid
from tqdm import tqdm

from bacformer.modeling import SPECIAL_TOKENS_DICT, BacformerForProteinProteinInteraction
from bacformer.modeling.utils import get_gpu_info
from bacformer.pp import extract_protein_info_from_genbank, protein_seqs_to_bacformer_inputs


def run(
    pretrained_model_dir: str,
    strain_gbff_filepath: str,
    output_filepath: str,
    ppi_batch_size: int = 50000,
):
    """Run genome-wide PPI inference on a single genome."""
    # load model
    model = BacformerForProteinProteinInteraction.from_pretrained(
        pretrained_model_dir, torch_dtype=torch.bfloat16
    ).eval()

    # get GPU info
    n_gpus, use_ipex = get_gpu_info()

    if use_ipex:
        device = "xpu"
    else:
        device = "cuda" if n_gpus > 0 else "cpu"

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        model.cuda()
    elif use_ipex:
        import intel_extension_for_pytorch as ipex

        model = model.to("xpu")
        model = ipex.optimize(model, dtype=torch.float32, level="O1", conv_bn_folding=False)

    # preprocess the bacterial genome assembly
    genome_df = extract_protein_info_from_genbank(strain_gbff_filepath)

    # update the RsaM sequence
    rsam_shorter_seq = "MQSLAPDQLVRLSYPELIDLSFQPYLAWIDTSLTAELKEFGLPVAYAGYSEWECQDSAPKLSISWNWFKEAFSGKVLIAPGGISCNIMLRSPRGYDLGPDMTQQLLLVWISRQGLECKLPAGLMFDRES"
    genome_df[genome_df["protein_id"] == "UUQ65455.1"].iloc[0]["protein_sequence"] = rsam_shorter_seq

    # get genome data
    assert genome_df["contig_idx"].nunique() == 1, (
        "The lines below only work for genomes made out of a single contig (i.e. chromosome)."
    )
    protein_sequences = genome_df["protein_sequence"].tolist()
    locus_tags = genome_df["protein_name"].tolist()
    protein_ids = genome_df["protein_id"].tolist()

    # embed the proteins with ESM-2 to get average protein embeddings
    inputs = protein_seqs_to_bacformer_inputs(
        protein_sequences=protein_sequences,
        device=device,
        batch_size=128,  # the batch size for computing the protein embeddings
        max_n_proteins=9000,  # the maximum number of proteins Bacformer was trained with
    )

    output = defaultdict(list)
    with torch.no_grad():
        model_output = model.bacformer(
            protein_embeddings=inputs["protein_embeddings"],
            special_tokens_mask=inputs["special_tokens_mask"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        # remove special tokens
        model_output.last_hidden_state = model_output.last_hidden_state[
            inputs["special_tokens_mask"] == SPECIAL_TOKENS_DICT["PROT_EMB"]
        ]
        # pass through the PPI head
        model_output.last_hidden_state = model.dense(model.dropout(model_output.last_hidden_state))
        # create genome-wide protein pairs
        prot_indices = list(range(model_output.last_hidden_state.shape[0]))
        prot_pairs = list(combinations(prot_indices, 2))
        print(f"Number of protein pairs: {len(prot_pairs)}")
        prot_pairs = torch.tensor([[p[0] for p in prot_pairs], [p[1] for p in prot_pairs]], dtype=torch.long)
        # predict in batches to avoid OOM
        for idx in tqdm(range(0, prot_pairs.shape[1], ppi_batch_size)):
            prot_pairs_batch = prot_pairs[:, idx : idx + ppi_batch_size]
            # get protein pair representations
            reprs = torch.stack(
                [
                    model_output.last_hidden_state[prot_pairs_batch[0]],
                    model_output.last_hidden_state[prot_pairs_batch[1]],
                ],
                dim=0,
            ).mean(dim=0)
            # get PPI logits
            logits = sigmoid(model.ppi_head(reprs))
            output["probability"] += logits.type(torch.float32).cpu().numpy().tolist()
            output["prot_1"] += prot_pairs_batch[0].cpu().numpy().tolist()
            output["prot_2"] += prot_pairs_batch[1].cpu().numpy().tolist()

    # save the output
    output_df = pd.DataFrame(output)
    # add locus tags and protein ids
    output_df["locus_tag_1"] = [locus_tags[i] for i in output_df["prot_1"]]
    output_df["locus_tag_2"] = [locus_tags[i] for i in output_df["prot_2"]]
    output_df["protein_id_1"] = [protein_ids[i] for i in output_df["prot_1"]]
    output_df["protein_id_2"] = [protein_ids[i] for i in output_df["prot_2"]]

    # save to parquet
    output_df.to_parquet(output_filepath)


class ArgumentParser(Tap):
    """Argument parser for genome-wide PPI inference."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    # file paths for loading data
    pretrained_model_dir: str
    strain_gbff_filepath: str
    output_filepath: str
    ppi_batch_size: int = 50000


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    run(
        pretrained_model_dir=args.pretrained_model_dir,
        strain_gbff_filepath=args.strain_gbff_filepath,
        output_filepath=args.output_filepath,
        ppi_batch_size=args.ppi_batch_size,
    )
