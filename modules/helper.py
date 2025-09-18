import evaluate
import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import torch
import gc
import time


def get_file_path(subject_id):
    """Get the file path for the given subject_id"""
    prefix = "p" + subject_id[:2]

    return prefix + "/p" + subject_id


def get_processed_reports(args):
    reports_df = pd.read_csv(
        f"{args.root_dir}/physionet.org/files/cxr-pro/1.0.0/mimic_train_impressions.csv"
    )
    reports_df["study_id"] = reports_df["study_id"].astype(str)
    reports_df["subject_id"] = reports_df["subject_id"].astype(str)
    return reports_df


def get_image_text_paths(args, queries_df, study_ids, train=True):
    """Get the image and text paths for the given study_ids"""
    corresponding_study_ids = []
    image_path_list = []
    texts_list = []
    if args.dataset_name == "mimic-cxr-vqa" or args.dataset_name == "med-diff-vqa":
        reports_df = get_processed_reports(args)
    else:
        reports_df = None
    if args.dataset_name == "mimic-cxr-vqa" or args.dataset_name == "med-diff-vqa":
        for study_id in tqdm(study_ids, desc="Processing File Paths"):
            subject_id = queries_df.loc[
                queries_df["study_id"] == study_id, "subject_id"
            ].values[0]
            file_path = get_file_path(subject_id=subject_id)

            if args.dataset_name == "med-diff-vqa":
                dicom_id = queries_df.loc[
                    queries_df["study_id"] == study_id, "dicom_id"
                ].values[0]

                folder_path = (
                    f"{args.root_dir}/{args.images_folder}/{file_path}/s{study_id}/"
                )
                image_path = folder_path + f"{dicom_id}.jpg"
                image_path_list.append(image_path)

            elif args.dataset_name == "mimic-cxr-vqa":
                image_path = queries_df.loc[
                    queries_df["study_id"] == study_id, "image_path"
                ].values[0]
                image_path_list.append(
                    f"{args.root_dir}/{args.images_folder}/{image_path}"
                )

            if train:
                # report text instead of path to the report
                text = reports_df.loc[
                    reports_df["study_id"] == study_id, "report"
                ].tolist()[0]
                texts_list.append(text)
            else:
                # for test data, get generated reports by MAIRA-2 instead of original reports
                if hasattr(args, "generated_texts_folder"):
                    text_path = f"{args.root_dir}/{args.generated_texts_folder}/{args.dataset_name}/s{study_id}.txt"

                    if os.path.exists(text_path):
                        texts_list.append(text_path)
                else:
                    texts_list.append("")

            corresponding_study_ids.append(study_id)

    assert len(corresponding_study_ids) == len(image_path_list) == len(texts_list)
    return image_path_list, texts_list, corresponding_study_ids



def save_json(data, dir_name, file_name):
    """Write the data to a JSONL file."""
    os.makedirs(dir_name, exist_ok=True)
    with open(f"{dir_name}/{file_name}.json", "w") as f:
        json.dump(data, f)


def create_output_file_name(args):
    """Create a save file name based on the arguments."""
    # Get the current date in the desired format (e.g., YYYYMMDD)
    # current_date = datetime.now().strftime("%Y%m%d")

    # Extract the model file name from the full path
    model_file_name = args.model_name.split("/")[-1]

    # Construct the output filename with the current date
    if args.eval_without_rag == "disabled" or args.eval_without_rag == "exact":
        output_file = f"{model_file_name}_search_{args.search_modality}_query_{args.query_modality}_top_{args.top_k}"
    elif args.eval_without_rag == "enabled":
        output_file = f"{model_file_name}_no_rag"
    # elif args.eval_without_rag == "exact":
    #     output_file = f"{model_file_name}_exact"

    return output_file


def get_dcm_names(path):
    dcm_names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in [".dcm"]:
                dcm_names.append(filename)

        return dcm_names


def compute_metrics(predictions, references, bert_only=False):
    references = [[text] for text in references]

    # Load evaluation metrics
    bert_score = evaluate.load("bertscore")
    bert_score_results = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
    )
    average_bert_score = {
        "precision": np.mean(bert_score_results["precision"]),
        "recall": np.mean(bert_score_results["recall"]),
        "f1": np.mean(bert_score_results["f1"]),
    }
    print(f"BERTScore: {average_bert_score}")
    if bert_only:
        return {
            "bert_score": average_bert_score,
        }

    else:
        rouge = evaluate.load("rouge")
        meteor = evaluate.load("meteor")
        bleu = evaluate.load("bleu")
        google_bleu = evaluate.load("google_bleu")

        # Compute metrics
        rouge_results = rouge.compute(predictions=predictions, references=references)
        meteor_results = meteor.compute(predictions=predictions, references=references)
        bleu_results = bleu.compute(predictions=predictions, references=references)
        google_bleu_results = google_bleu.compute(
            predictions=predictions, references=references
        )

        # Calculate average scores
        average_rouge = {
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "rougeLsum": rouge_results["rougeLsum"],
        }

        average_meteor = meteor_results["meteor"]

        average_bleu = bleu_results["bleu"]
        average_google_bleu = google_bleu_results["google_bleu"]

        # Display results
        print(f"ROUGE: {average_rouge}")
        print(f"METEOR: {average_meteor}")
        print(f"BLEU: {average_bleu}")
        print(f"Google BLEU: {average_google_bleu}")

        return {
            "rouge": average_rouge,
            "meteor": average_meteor,
            "bert_score": average_bert_score,
            "bleu": average_bleu,
            "google_bleu": average_google_bleu,
        }


def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)
