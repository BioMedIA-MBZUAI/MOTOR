from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import argparse
import os
import glob
import wandb
import ast
from tqdm import tqdm

import sys

sys.path.append("..")

from modules.helper import get_image_text_paths, compute_metrics
from modules.data import load_data

# Set fixed seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(SEED)
torch.random.manual_seed(SEED)


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir", type=str, required=True, help="Data root directory (required)"
)
parser.add_argument(
    "--images_folder",
    type=str,
    help="Images folder",
    default="/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
)
parser.add_argument(
    "--texts_folder",
    type=str,
    help="Texts folder",
    default="/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Dataset name",
    choices=[
        "mimic-cxr-vqa",
        "med-diff-vqa",
    ],
    required=True,
)
parser.add_argument("--subset_size", type=int, help="Samples count", default=5000)
args = parser.parse_args()

IMAGES_DIR = f"{args.root_dir}/{args.images_folder}"


# Load model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/maira-2", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.eval()
model = model.to(device)


def parse_caption_data(text):
    """Format the caption data from the model output."""
    caption_data = []
    for line in text.strip().splitlines():
        if line.strip():  # Skip empty lines
            caption_data.append(ast.literal_eval(line.strip()))
    return caption_data


def plot_bboxes_with_labels(
    image,
    bboxes_list,
    labels_list,
    colors_list,
    font_size=24,
    output_file="output.jpg",
):
    """
    Draw multiple bounding boxes with labels and colors on an image.

    Args:
    - image (PIL.Image): The image to draw on.
    - bboxes_list (list of lists): List of bounding boxes, where each bbox is a list of [x_min, y_min, x_max, y_max].
    - labels_list (list of str): List of labels for each bounding box.
    - colors_list (list of str): List of colors for each bounding box.
    - font_size (int): Font size for the labels.
    - output_file (str): The output file name for saving the image.

    Returns:
    - image (PIL.Image): The image with bounding boxes and labels drawn.
    """
    # Convert image to RGB if it's grayscale
    if image.mode != "RGB":
        image = image.convert("RGB")

    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    # Load a font with adjustable size
    font = ImageFont.truetype("times.ttf", font_size)

    for i, bbox in enumerate(bboxes_list):
        # Adjust bbox for original image size
        bbox = processor.adjust_box_for_original_image_size(
            width=image_width,
            height=image_height,
            box=bbox[0],
        )

        # Convert normalized bbox to pixel coordinates
        x_topleft = int(bbox[0] * image_width)
        y_topleft = int(bbox[1] * image_height)
        x_bottomright = int(bbox[2] * image_width)
        y_bottomright = int(bbox[3] * image_height)

        # Draw the bounding box
        draw.rectangle(
            [(x_topleft, y_topleft), (x_bottomright, y_bottomright)],
            outline=colors_list[i],  # Color of the bounding box
            width=3,  # Line thickness
        )

        # Draw the label text
        text = labels_list[i]

        # Get text size using getbbox()
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = (
            x_topleft,
            max(0, y_topleft - text_height),
        )  # Ensure text position is within image bounds

        # Draw label background
        draw.rectangle(
            [
                (text_position[0], text_position[1]),
                (text_position[0] + text_width, text_position[1] + text_height),
            ],
            fill=colors_list[i],  # Background matches the box color
        )

        # Draw label text
        draw.text(
            text_position, text, fill="white", font=font
        )  # White text for contrast

    # Save the image with the bounding boxes and labels
    image.save(output_file)
    print(f"Image saved to {output_file}")


def inference(
    args, model, processor, image_path_list, ground_truth_texts=None, grounded=True
):
    """Generate captions for the images."""
    generated_texts = []

    if grounded:
        generated_dir = (
            f"{args.root_dir}/grounded_reports_maira2_v3/{args.dataset_name}"
        )
    else:
        generated_dir = (
            f"{args.root_dir}/non_grounded_reports_maira2_v3/{args.dataset_name}"
        )
    os.makedirs(generated_dir, exist_ok=True)

    for image_path in tqdm(image_path_list):
        file_name = image_path.split("/")[-2]
        file_path = f"{generated_dir}/{file_name}.txt"
        if os.path.exists(file_path):
            continue
        image = Image.open(image_path)
        sample_data = {
            "frontal": image,
            "lateral": None,
            "indication": "None.",
            "comparison": "None.",
            "technique": "Frontal view of the chest.",
            # "phrase": "Pleural effusion.",  # For the phrase grounding example. This patient has pleural effusion.
        }
        processed_inputs = processor.format_and_preprocess_reporting_input(
            current_frontal=sample_data["frontal"],
            current_lateral=sample_data["lateral"],
            prior_frontal=None,  # Our example has no prior
            indication=sample_data["indication"],
            technique=sample_data["technique"],
            comparison=sample_data["comparison"],
            prior_report=None,  # Our example has no prior
            return_tensors="pt",
            get_grounding=grounded,  # For this example we generate a non-grounded report
        )

        processed_inputs = processed_inputs.to(device)
        with torch.no_grad():
            output_decoding = model.generate(
                **processed_inputs,
                max_new_tokens=(
                    450 if grounded else 300
                ),  # Set to 450 for grounded reporting, 300 for non-grounded
                use_cache=True,
            )
        prompt_length = processed_inputs["input_ids"].shape[-1]
        decoded_text = processor.decode(
            output_decoding[0][prompt_length:], skip_special_tokens=True
        )
        decoded_text = (
            decoded_text.lstrip()
        )  # Findings generation completions have a single leading space
        try:
            prediction = processor.convert_output_to_plaintext_or_grounded_sequence(
                decoded_text
            )
        except:
            try:
                decoded_text += "</obj>"
                prediction = processor.convert_output_to_plaintext_or_grounded_sequence(
                    decoded_text
                )
            except:
                # Open the file in append mode ('a')
                with open("bad.txt", "a") as bad_file:
                    bad_file.write(image_path + "\n")
                continue

        print("Prediction:", prediction)
        if grounded:
            formatted_output = "\n".join(map(str, prediction))
            prediction = formatted_output.strip()

        parsed_prediction = parse_caption_data(prediction)
        captions, bboxes = zip(*parsed_prediction)
        generated_texts.append(" ".join(captions))

        torch.cuda.empty_cache()

        with open(file_path, "w") as f:
            f.write(prediction)

    if ground_truth_texts is not None:
        wandb.init(project=f"Captions {args.dataset_name} with MAIRA-2")
        scores = compute_metrics(generated_texts, ground_truth_texts)
        wandb.log(scores)


train_df, train_study_ids, test_df, test_queries, test_answers, test_study_ids = (
    load_data(args)
)

# =================================================================================================
# Run inference on train data
# =================================================================================================
train_image_path_list, train_texts_list, train_corresponding_study_ids = (
    get_image_text_paths(args, train_df, train_study_ids, train=True)
)

# Test the model
inference(
    args=args,
    model=model,
    processor=processor,
    image_path_list=train_image_path_list,
    # ground_truth_texts=texts_list, # Uncomment this line to compute metrics on train data
    grounded=True,
)

# =================================================================================================
# Run inference on test data
# =================================================================================================
test_image_path_list, _, test_corresponding_study_ids = get_image_text_paths(
    args, test_df, test_study_ids, train=False
)

# Test the model
inference(
    args=args,
    model=model,
    processor=processor,
    image_path_list=test_image_path_list,
    grounded=True,
)
