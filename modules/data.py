import pandas as pd
import json
import os
from modules.helper import create_output_file_name
from sklearn.model_selection import train_test_split

# Set fixed seed for reproducibility
SEED = 42


def load_data(args):
    if args.dataset_name == "med-diff-vqa":
        train_queries_df, test_queries_df = load_med_diff_vqa(args)
        train_queries_df = load_mimic_knowledge_data(args, train_queries_df)
        train_queries_df.to_csv("med_diff_train_queries.csv", index=False)
        test_queries_df = exclude_pretrained_data(args, test_queries_df)
        test_queries_df.to_csv("med_diff_test_queries.csv", index=False)
    elif args.dataset_name == "mimic-cxr-vqa":
        train_queries_df = load_mimic_cxr_vqa(
            args,
            "physionet.org/files/mimic-ext-mimic-cxr-vqa/1.0.0/MIMIC-Ext-MIMIC-CXR-VQA/dataset/train.json",
        )
        train_queries_df = load_mimic_knowledge_data(args, train_queries_df)
        train_queries_df.to_csv("mimic_train_queries.csv", index=False)
        test_queries_df = load_mimic_cxr_vqa(
            args,
            "physionet.org/files/mimic-ext-mimic-cxr-vqa/1.0.0/MIMIC-Ext-MIMIC-CXR-VQA/dataset/test.json",
        )
        test_queries_df = exclude_pretrained_data(args, test_queries_df)
        test_queries_df.to_csv("mimic_test_queries.csv", index=False)

    print("Train Queries Shape: ", train_queries_df.shape)
    print("Test Queries Shape: ", test_queries_df.shape)

    test_queries_df = continue_from_last_save(args, test_queries_df)

    if args.dataset_name == "med-diff-vqa" or args.dataset_name == "mimic-cxr-vqa":
        train_subject_ids = train_queries_df["subject_id"].unique().tolist()
        print("Train unique subjects: ", len(train_subject_ids))
        train_study_ids = train_queries_df["study_id"].unique().tolist()
        print("Train unique studies: ", len(train_study_ids))

        test_subject_ids = test_queries_df["subject_id"].unique().tolist()
        print("Test unique subjects: ", len(test_subject_ids))
        test_study_ids = test_queries_df["study_id"].unique().tolist()
        print("Test unique studies: ", len(test_study_ids))

    test_queries = test_queries_df["question"].tolist()
    test_answers = test_queries_df["answer"].tolist()
    print("Queries: ", len(test_queries))
    print("Answers: ", len(test_answers))
    print("First query: ", test_queries[0])
    print("First answer: ", test_answers[0])
    print("Last query: ", test_queries[-1])
    print("Last answer: ", test_answers[-1])

    return (
        train_queries_df,
        train_study_ids,
        test_queries_df,
        test_queries,
        test_answers,
        test_study_ids,
    )


def continue_from_last_save(args, queries_df):
    if hasattr(args, "model_name"):
        output_file = create_output_file_name(args)
        last_save_file = f"{args.outputs_dir}/{output_file}.csv"

        # Splitting data into batches
        if args.start_index > 0 and os.path.exists(last_save_file):
            start_index = args.start_index
        elif os.path.exists(last_save_file):
            # Continue from last save
            last_save_data = pd.read_csv(last_save_file)
            start_index = len(last_save_data)
        else:
            start_index = 0

    else:
        start_index = 0
    print("Starting from index: ", start_index)
    end_index = start_index + args.subset_size
    if end_index > queries_df.shape[0]:
        end_index = queries_df.shape[0]
    print("End index: ", end_index)
    queries_df = queries_df.iloc[start_index:end_index, :]
    print("Subest Queries Shape: ", queries_df.shape)
    print(queries_df.head())

    return queries_df


def load_med_diff_vqa(args):
    queries_df = pd.read_csv(
        f"{args.root_dir}/physionet.org/files/medical-diff-vqa/1.0.0/mimic_pair_questions.csv"
    )

    selected_questions = ["abnormality", "presence"]
    test_queries_df = queries_df[
        (queries_df["split"] == "test")
        & (queries_df["question_type"].isin(selected_questions))
    ]
    train_queries_df = queries_df[
        (queries_df["split"] == "train")
        & (queries_df["question_type"].isin(selected_questions))
    ]
    train_queries_df["study_id"] = train_queries_df["study_id"].astype(str)
    test_queries_df["study_id"] = test_queries_df["study_id"].astype(str)
    train_queries_df["subject_id"] = train_queries_df["subject_id"].astype(str)
    test_queries_df["subject_id"] = test_queries_df["subject_id"].astype(str)

    # Include only samples with frontal view
    mimic_metadata_df = pd.read_csv(
        f"{args.root_dir}/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"
    )
    mimic_metadata_df = mimic_metadata_df[
        mimic_metadata_df["ViewPosition"].isin(["PA", "AP"])
    ]
    mimic_metadata_df["dicom_id"] = mimic_metadata_df["dicom_id"].astype(str)
    mimic_metadata_df["study_id"] = mimic_metadata_df["study_id"].astype(str)
    mimic_metadata_df["subject_id"] = mimic_metadata_df["subject_id"].astype(str)
    train_queries_df = train_queries_df[
        train_queries_df["study_id"].isin(mimic_metadata_df["study_id"])
    ]
    test_queries_df = test_queries_df[
        test_queries_df["study_id"].isin(mimic_metadata_df["study_id"])
    ]
    # dicom_id is needed to get the image path
    train_queries_df = pd.merge(
        train_queries_df,
        mimic_metadata_df[["study_id", "dicom_id"]],
        on="study_id",
        how="left",
    )
    test_queries_df = pd.merge(
        test_queries_df,
        mimic_metadata_df[["study_id", "dicom_id"]],
        on="study_id",
        how="left",
    )
    # # Exclude study_id == 53836631 (has error in the generated captions)
    train_queries_df = train_queries_df[train_queries_df["study_id"] != "53836631"]
    train_queries_df = train_queries_df[train_queries_df["study_id"] != "50866109"]
    train_queries_df = train_queries_df[train_queries_df["study_id"] != "51146516"]
    train_queries_df = train_queries_df[train_queries_df["study_id"] != "53047419"]
    # train_queries_df = train_queries_df[train_queries_df["study_id"] != "52224646"]
    test_queries_df = test_queries_df[test_queries_df["study_id"] != "53836631"]
    test_queries_df = test_queries_df[test_queries_df["study_id"] != "50866109"]
    test_queries_df = test_queries_df[test_queries_df["study_id"] != "51146516"]
    test_queries_df = test_queries_df[test_queries_df["study_id"] != "53047419"]
    # test_queries_df = test_queries_df[test_queries_df["study_id"] != "52224646"]

    return train_queries_df, test_queries_df


def load_mimic_cxr_vqa(args, file_path):
    # Read JSON file
    with open(f"{args.root_dir}/{file_path}", "r") as file:
        json_data = json.load(file)

    # Convert JSON data to DataFrame
    queries_df = pd.DataFrame(json_data)

    # Normalize column "answer" to string instead of list
    queries_df["answer"] = [",".join(map(str, l)) for l in queries_df["answer"]]
    # remove any rows with empty answer
    queries_df = queries_df.dropna(subset=["answer"])

    queries_df = queries_df[
        queries_df["content_type"].isin(
            ["abnormality", "presence", "anatomy", "attribute"]
        )
    ]

    queries_df["study_id"] = queries_df["study_id"].astype(str)
    queries_df["subject_id"] = queries_df["subject_id"].astype(str)

    # # Exclude study_id == 53836631 (has error in the generated captions)
    # queries_df = queries_df[queries_df["study_id"] != "53836631"]
    # queries_df = queries_df[queries_df["study_id"] != "50866109"]
    # queries_df = queries_df[queries_df["study_id"] != "51146516"]
    # queries_df = queries_df[queries_df["study_id"] != "53047419"]
    # queries_df = queries_df[queries_df["study_id"] != "52224646"]

    return queries_df


def exclude_pretrained_data(args, vqa_df):
    """Exclude data that is already used in pretraining medical VLMs"""
    df = pd.read_csv(
        f"{args.root_dir}/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv"
    )
    df = df[df["split"] == "train"]
    vqa_df = vqa_df[~vqa_df["study_id"].isin(df["study_id"])]
    return vqa_df


def load_mimic_knowledge_data(args, vqa_df):
    # NOTE: we cannot load all reports in mimic because of memory overload, so we pick a subset of reports while making sure we include all labels
    mimic_df = pd.read_csv(
        f"{args.root_dir}/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
    )
    # Ensure to include only "train" split
    mimic_split_df = pd.read_csv(
        f"{args.root_dir}/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv"
    )
    mimic_split_df = mimic_split_df[mimic_split_df["split"] == "train"]

    # Filter rows where study_id is in train vqa_df study_id
    mimic_df = mimic_df[mimic_df["study_id"].isin(mimic_split_df["study_id"])]
    # Exclude study_id == 58612055 (has error in the generated captions)
    mimic_df = mimic_df[mimic_df["study_id"] != 58612055]
    # Include only samples with reports
    reports_df = pd.read_csv(
        f"{args.root_dir}/physionet.org/files/cxr-pro/1.0.0/mimic_train_impressions.csv"
    )
    reports_df = reports_df.dropna()
    mimic_df = mimic_df[mimic_df["study_id"].isin(reports_df["study_id"])]

    # Filtering the data for rows = '1' only in any of the specific 14 columns (labels)
    filtered_rows = []
    for idx, row in mimic_df.iterrows():
        for col in mimic_df.columns[2:]:  # Skipping 'subject_id' and 'study_id'
            if row[col] == 1:
                filtered_rows.append(
                    {
                        "subject_id": row["subject_id"],
                        "study_id": row["study_id"],
                        "label": col,
                    }
                )
                break  # Stop at the first found '1' to take only one label

    # Converting the filtered rows to a new DataFrame
    filtered_df = pd.DataFrame(filtered_rows)
    filtered_df["study_id"] = filtered_df["study_id"].apply(
        lambda x: str(int(float(x)))
    )
    filtered_df["subject_id"] = filtered_df["subject_id"].apply(
        lambda x: str(int(float(x)))
    )

    # Filter rows where study_id is in train vqa_df
    filtered_df = filtered_df[filtered_df["study_id"].isin(vqa_df["study_id"])]

    # For each unique label, take 40 rows only
    filtered_df = (
        filtered_df.groupby("label")
        .apply(lambda x: x.sample(n=40, random_state=SEED))
        .reset_index(drop=True)
    )

    if args.dataset_name == "mimic-cxr-vqa":
        # merge column "image_path" from vqa_df to filtered_df based on study_id
        filtered_df = pd.merge(
            filtered_df,
            vqa_df[["study_id", "image_path"]],
            on="study_id",
            how="left",
        )
    elif args.dataset_name == "med-diff-vqa":
        # dicom_id is needed to get the image path
        filtered_df = pd.merge(
            filtered_df,
            vqa_df[["study_id", "dicom_id"]],
            on="study_id",
            how="left",
        )

    return filtered_df
