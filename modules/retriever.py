import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
)
from PIL import Image

from qdrant_client import models, QdrantClient

from modules.rag_ot import RAGOptimalTransport

from tqdm import tqdm
import ast


rad_dino_model, rad_dino_processor = None, None
bert_model, bert_tokenizer = None, None
ot_module = None
DB_COLLECTION_NAME = "medical_data"


def initialize_models(args, device="cuda"):
    """Initialize the models and tokenizers"""

    global rad_dino_model, rad_dino_processor
    global bert_model, bert_tokenizer
    global ot_module

    if args.embedding_model == "dino":
        rad_dino_model_path = "microsoft/rad-dino"
        rad_dino_model = AutoModel.from_pretrained(rad_dino_model_path)
        rad_dino_processor = AutoImageProcessor.from_pretrained(rad_dino_model_path)
        rad_dino_model.to(device)
        rad_dino_model.eval()

        bert_model_path = "emilyalsentzer/Bio_ClinicalBERT"
        bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        bert_model = AutoModel.from_pretrained(bert_model_path)
        bert_model.to(device)
        bert_model.eval()

        if args.ot == "enabled":
            ot_module = RAGOptimalTransport(
                alpha=args.ot_alpha,
                beta=args.ot_beta,
                delta=args.ot_delta,
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                image_model=rad_dino_model,
                image_preprocessor=rad_dino_processor,
            )


def build_retriever(args, image_path_list, texts_list, device="cuda", db_name="db"):
    # ==============================================================================
    # Generate embeddings for each image and text (medical report)
    # ==============================================================================
    image_embeddings = []
    text_embeddings = []

    for text, image_path in zip(texts_list, image_path_list):
        if args.embedding_model == "dino":
            image_preprocessed = rad_dino_processor(
                images=Image.open(image_path), return_tensors="pt"
            ).to(device)
            text_preprocessed = bert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

        with torch.no_grad():
            if args.embedding_model == "dino":
                image_encoding = rad_dino_model(**image_preprocessed).pooler_output
                text_encoding = bert_model(**text_preprocessed).pooler_output

        image_embeddings.append(image_encoding)
        text_embeddings.append(text_encoding)
        del (text_encoding, image_encoding, text_preprocessed, image_preprocessed)

    # ==============================================================================
    # Store embeddings in a vector database Qdrant
    # ==============================================================================
    qdrant_client = QdrantClient(path=db_name)

    qdrant_client.recreate_collection(
        collection_name=DB_COLLECTION_NAME,
        vectors_config={
            "report": models.VectorParams(
                size=text_embeddings[0].shape[1], distance=models.Distance.COSINE
            ),
            "image": models.VectorParams(
                size=image_embeddings[0].shape[1],
                distance=models.Distance.COSINE,
            ),
        },
    )

    samples = []
    # Iterate through both lists and combine them into dictionaries
    for processed_text, image_path in zip(texts_list, image_path_list):
        sample = {
            "processed_text": processed_text,
            "image_path": image_path,
        }
        samples.append(sample)

    # Create the final dictionary
    data_dict = {"samples": samples}

    print("Qdrant creating recordings")
    points = []
    for idx, sample in tqdm(
        enumerate(data_dict["samples"]), total=len(data_dict["samples"])
    ):
        points.append(
            models.PointStruct(
                id=idx,
                vector={
                    "report": text_embeddings[idx],
                    "image": image_embeddings[idx],
                },
                payload=sample,
            )
        )

    print("Qdrant uploading collections")
    qdrant_client.upsert(collection_name=DB_COLLECTION_NAME, points=points)

    print("Qdrant successfully uploaded to VDB")
    return qdrant_client


def retrieve_by_query(
    query_index,
    args,
    retriever_engine,
    text_query=None,
    image_caption=None,
    image_query=None,
    device="cuda",
):

    # ==============================================================================
    # Query
    # ==============================================================================
    if args.query_modality == "image":
        # Search by image
        if args.embedding_model == "dino":
            image_preprocessed = rad_dino_processor(
                images=Image.open(image_query), return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                embeddings = rad_dino_model(**image_preprocessed).pooler_output

        del image_preprocessed

    # ==============================================================================
    # Search
    # ==============================================================================
    hits = retriever_engine.search(
        collection_name=DB_COLLECTION_NAME,
        query_vector=models.NamedVector(name="image", vector=embeddings[0].tolist()),
        limit=args.top_k,
        # Avoid returning the exact image
        query_filter=models.Filter(
            must_not=[
                models.FieldCondition(
                    key="image_path", match=models.MatchValue(value=image_query)
                )
            ]
        ),
        # search_params=models.SearchParams(hnsw_ef=128, exact=False),
    )

    if args.ot == "enabled":
        hits, min_ot_cost = rank_by_ot(
            args=args,
            question=text_query,
            image=Image.open(image_query),
            image_caption=image_caption,
            hits=hits,
        )

    # join top_k results with "\n"
    if len(hits) > 0:
        report_text = "\n".join(hit.payload["processed_text"] for hit in hits)
        return (
            report_text,
            ",".join(hit.payload["image_path"] for hit in hits),
            hits[0].score,
        )
    else:
        return (None, None, None)


def rank_by_ot(args, question, image, image_caption, hits):
    retrieved_images = []
    retrieved_reports = []
    retrieved_captions = []
    for hit in hits:
        hit_image = Image.open(hit.payload["image_path"])
        retrieved_images.append(hit_image)
        retrieved_reports.append(hit.payload["processed_text"])

        if args.dataset_name == "vqa-rad":
            hit_caption_path = (
                args.root_dir
                + "/"
                + args.generated_texts_folder
                + "/"
                + args.dataset_name
                + "/"
                + hit.payload["image_path"].split("/")[-1].split(".")[0]
                + ".txt"
            )
        else:
            hit_caption_path = (
                args.root_dir
                + "/"
                + args.generated_texts_folder
                + "/"
                + args.dataset_name
                + "/"
                + hit.payload["image_path"].split("/")[-2]
                + ".txt"
            )
        with open(hit_caption_path, "r") as f:
            retrieved_captions.append(parse_caption_data(f.read()))

    results = ot_module.compute_optimal_transport_cost(
        question=question,
        query_caption=parse_caption_data(image_caption),
        query_image=image,
        retrieved_reports=retrieved_reports,
        retrieved_captions=retrieved_captions,
        retrieved_images=retrieved_images,
    )

    ot_scores = [cost for cost in results.values()]

    # Sort the hits by OT score (lowest OT score is the most similar)
    hits = [
        hits[i]
        for i in sorted(
            range(len(ot_scores)),
            key=lambda k: ot_scores[k],
            reverse=False,
        )
    ]
    number_of_new_hits = len(hits) // 2
    return hits[:number_of_new_hits], ot_scores[0]


def parse_caption_data(text):
    """Format the caption data."""
    caption_data = []
    for line in text.strip().splitlines():
        if line.strip():  # Skip empty lines
            caption_data.append(ast.literal_eval(line.strip()))
    return caption_data
