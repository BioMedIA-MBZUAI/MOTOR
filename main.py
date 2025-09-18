import os
import argparse
import datetime
import pandas as pd

import torch

from modules.helper import get_image_text_paths
from modules.data import load_data

from modules.retriever import (
    initialize_models,
    build_retriever,
)
from modules.vlm import run_vlm

# Create an argument parser
parser = argparse.ArgumentParser()

# Set arguments
parser.add_argument(
    "--root_dir", type=str, required=True, help="Data root directory (required)"
)
parser.add_argument(
    "--outputs_dir", type=str, help="Output directory", default="outputs"
)
parser.add_argument(
    "--texts_folder",
    type=str,
    help="Train Texts folder",
    default="/physionet.org/files/mimic-cxr/2.1.0/files",
)
parser.add_argument(
    "--generated_texts_folder",
    type=str,
    help="Path to generated reports folder",
    required=True,
)
parser.add_argument(
    "--images_folder",
    type=str,
    help="Images folder",
    default="/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
)
parser.add_argument("--top_k", type=int, help="Top k", default=10)
parser.add_argument(
    "--query_modality",
    type=str,
    help="Query modality",
    default="image",
)
parser.add_argument(
    "--search_modality",
    type=str,
    help="Search modality",
    default="image",
)
parser.add_argument(
    "--embedding_model",
    type=str,
    help="Embedding model",
    default="dino",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="Model name",
)
parser.add_argument(
    "--quantize",
    type=str,
    help="Quantize",
    default="disabled",
    choices=["enabled", "disabled"],
)
parser.add_argument("--retriever", type=str, help="Retriever name", default="qdrant")
parser.add_argument(
    "--eval_without_rag",
    type=str,
    help="Evaluate without RAG",
    default="disabled",
    choices=["enabled", "disabled", "exact"],
)  # exact means that the correct report is used
parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="Max new tokens",
    default=200,
)
parser.add_argument("--start_index", type=int, help="First row index", default=0)
parser.add_argument("--subset_size", type=int, help="Samples count", default=5000)
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Dataset name",
    choices=["mimic-cxr-vqa", "med-diff-vqa"],
    required=True,
)
parser.add_argument(
    "--use_caption",
    type=str,
    help="Wether to path image captions to VLM",
    default="enabled",
    choices=["enabled", "disabled"],
)
parser.add_argument(
    "--ot",
    type=str,
    help="Optimal transport enabled or disabled",
    default="enabled",
    choices=["enabled", "disabled"],
)
parser.add_argument(
    "--ot_alpha", type=float, help="OT weight for question", default=0.2
)
parser.add_argument("--ot_beta", type=float, help="OT weight for text", default=0.3)
parser.add_argument("--ot_delta", type=float, help="OT weight for vision", default=0.5)

args = parser.parse_args()
print(args)

# Create a directory to store the results
if args.eval_without_rag == "disabled" or args.eval_without_rag == "exact":
    args.outputs_dir = f"{args.outputs_dir}/{args.dataset_name}/{args.embedding_model}"
else:
    args.outputs_dir = f"{args.outputs_dir}/{args.dataset_name}"
os.makedirs(args.outputs_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("Time start: ", now)

#=========================================================================
# Load data
#=========================================================================
print("\nLoading dataset\n")
(
    train_queries_df,
    train_study_ids,
    test_queries_df,
    test_queries,
    test_answers,
    test_study_ids,
) = load_data(args)

test_image_path_list, test_texts_list, test_corresponding_study_ids = (
    get_image_text_paths(args, test_queries_df, test_study_ids, train=False)
)

#=========================================================================
# Initialize the retriever engine
#=========================================================================
if args.eval_without_rag == "exact" or args.eval_without_rag == "enabled":
    retriever_engine = None

else:
    # Activate the retriever engine
    db_dir = "db"
    os.makedirs(db_dir, exist_ok=True)

    # Knowledge data data
    train_image_path_list, train_texts_list, train_corresponding_study_ids = (
        get_image_text_paths(args, train_queries_df, train_study_ids, train=True)
    )

    initialize_models(args=args, device=device)
    retriever_engine = build_retriever(
        args,
        train_image_path_list,
        train_texts_list,
        device=device,
        db_name=f"{db_dir}/{now}_{args.search_modality}_{args.embedding_model}_db",
    )
#=========================================================================
# Querying the DB
#=========================================================================
torch.cuda.empty_cache()
vlm_args = {
    "device": device,
    "queries_df": test_queries_df,
    "queries": test_queries,
    "answers": test_answers,
    "image_path_list": test_image_path_list,
    "texts_list": test_texts_list,
    "retriever_engine": retriever_engine,
    "corresponding_study_ids": test_corresponding_study_ids,
    "max_new_tokens": args.max_new_tokens,
}
contexts, relevant_images_paths, rag_responses, responses, sim_scores, save_file = (
    run_vlm(args=args, vlm_args=vlm_args)
)

#=========================================================================
# Save results in a CSV file
#=========================================================================
if args.eval_without_rag != "enabled":
    if len(contexts) > 0 and len(rag_responses) > 0:
        result_df = pd.DataFrame(
            {
                "Image Path": (
                    test_queries_df["image_path"]
                    if args.dataset_name == "mimic-cxr-vqa"
                    else test_queries_df["study_id"]
                ),
                "Question": test_queries,
                "Answer": test_answers,
                "Retrieved Image": relevant_images_paths,
                "Retrieved Context": contexts,
                "RAG Response": rag_responses,
            }
        )
else:
    result_df = pd.DataFrame(
        {
            "Image Path": (
                test_queries_df["image_path"]
                if args.dataset_name == "mimic-cxr-vqa"
                else test_queries_df["study_id"]
            ),
            "Question": test_queries,
            "Answer": test_answers,
            "Response": responses,
        }
    )

last_save_file = f"{args.outputs_dir}/{save_file}.csv"
if os.path.exists(last_save_file):
    # Load existing CSV to check columns
    existing_df = pd.read_csv(f"{last_save_file}")
    # append results
    result_df = pd.concat([existing_df, result_df], axis=0)
    result_df.to_csv(f"{last_save_file}", header=True, index=False)
else:
    result_df.to_csv(f"{last_save_file}", header=True, index=False)


now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("Time end: ", now)
