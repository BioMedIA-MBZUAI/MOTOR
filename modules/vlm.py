import random
import re
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoProcessor,
)
from transformers import AutoProcessor, AutoTokenizer

from dragonfly.models.modeling_dragonfly import DragonflyForCausalLM
from dragonfly.models.processing_dragonfly import DragonflyProcessor

import sys

sys.path.append("llavamed")
from llavamed.llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llavamed.llava.conversation import conv_templates
from llavamed.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)
from llavamed.llava.model.builder import load_pretrained_model


from modules.retriever import (
    retrieve_by_query,
)
from modules.helper import create_output_file_name

# Set fixed seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(SEED)
torch.random.manual_seed(SEED)


def run_vlm(args, vlm_args):
    if args.model_name == "llava-med-v1.5-mistral-7b":
        print("Using LlaVA-Med")
        return init_llavamed(args=args, vlm_args=vlm_args)
    elif args.model_name == "Dragonfly":
        print("Using Dragonfly")
        return init_dragonfly(args=args, vlm_args=vlm_args)


def run_queries(
    args,
    vlm_args,
    model_name,
    model,
    processor,
    rag_prompt_template,
    prompt_template,
):

    output_file = create_output_file_name(args)
    contexts = []
    relevant_images_paths = []
    rag_responses = []
    responses = []
    sim_scores = []

    for i in tqdm(range(len(vlm_args["queries"]))):
        # ========================RAG========================
        # Get study_id corresponding to the query
        current_study_id = (
            vlm_args["queries_df"]
            .loc[
                vlm_args["queries_df"]["question"] == vlm_args["queries"][i], "study_id"
            ]
            .values[0]
        )
        # Get the image path corresponding to the study_id
        image_query = vlm_args["image_path_list"][
            vlm_args["corresponding_study_ids"].index(current_study_id)
        ]
        caption = vlm_args["texts_list"][
            vlm_args["corresponding_study_ids"].index(current_study_id)
        ]
        with open(caption, "r") as f:
            caption = f.read()
        if "grounded" in args.generated_texts_folder:
            tuples = [eval(line) for line in caption.splitlines()]
            captions_without_bbox = [t[0] for t in tuples]
            caption_without_bbox = " ".join(captions_without_bbox)
        else:
            caption_without_bbox = caption

        print("\nQuery ", i, ": ", vlm_args["queries"][i])
        print("Ground Truth: ", vlm_args["answers"][i])
        if model_name == "llava-1.5-7b-hf" or model_name == "med_flamingo":
            raw_image = Image.open(image_query)
        else:
            raw_image = Image.open(image_query).convert("RGB")

        # ==============================================================================
        # ========================With Retrieval=====================================
        # ==============================================================================
        if args.eval_without_rag != "enabled":
            if vlm_args["retriever_engine"] is not None:
                if args.embedding_model == "dino":
                    retrieved_context, relevant_image_path, sim_score = (
                        retrieve_by_query(
                            query_index=i,
                            args=args,
                            retriever_engine=vlm_args[
                                "retriever_engine"
                            ],
                            text_query=vlm_args["queries"][i],
                            image_caption=caption,
                            image_query=image_query,
                        )
                    )
                if sim_score is None:
                    sim_scores.append(0.0)
                else:
                    sim_scores.append(sim_score)

            if retrieved_context is not None:
                contexts.append(retrieved_context)

                if relevant_image_path is not None:
                    relevant_images_paths.append(
                        relevant_image_path.replace(args.root_dir, "").replace(
                            args.images_folder, "_"
                        )
                    )
                else:
                    relevant_images_paths.append("No relevant image found.")

                if args.use_caption == "enabled":
                    rag_prompt = rag_prompt_template.format(
                        question=vlm_args["queries"][i],
                        context=retrieved_context,
                        caption=caption_without_bbox,
                    )

                else:
                    rag_prompt = rag_prompt_template.format(
                        question=vlm_args["queries"][i], context=retrieved_context
                    )

            if model_name == "llava-med-v1.5-mistral-7b":
                refined_result = run_llavamed(
                    model,
                    processor[0],
                    processor[1],
                    raw_image,
                    rag_prompt,
                    vlm_args["max_new_tokens"],
                )

            elif model_name == "Dragonfly":
                refined_result = run_dragonfly(
                    vlm_args=vlm_args,
                    model=model,
                    processor=processor[0],
                    tokenizer=processor[1],
                    image=raw_image,
                    prompt=rag_prompt,
                )

            print("\nResponse with RAG: ", refined_result)
            rag_responses.append(refined_result)
            torch.cuda.empty_cache()

        # ==============================================================================
        # ========================Without Retrieval=====================================
        # ==============================================================================
        elif args.eval_without_rag == "enabled":
            if args.use_caption == "enabled":
                prompt = prompt_template.format(
                    question=vlm_args["queries"][i], caption=caption_without_bbox
                )
            else:
                prompt = prompt_template.format(question=vlm_args["queries"][i])

            if model_name == "llava-med-v1.5-mistral-7b":
                refined_result = run_llavamed(
                    model,
                    processor[0],
                    processor[1],
                    raw_image,
                    prompt,
                    vlm_args["max_new_tokens"],
                )

            elif model_name == "Dragonfly":
                refined_result = run_dragonfly(
                    vlm_args=vlm_args,
                    model=model,
                    processor=processor[0],
                    tokenizer=processor[1],
                    image=raw_image,
                    prompt=prompt,
                )

            print("\nResponse without RAG: ", refined_result)
            responses.append(refined_result)
            torch.cuda.empty_cache()

    return (
        contexts,
        relevant_images_paths,
        rag_responses,
        responses,
        sim_scores,
        output_file,
    )


def init_llavamed(args, vlm_args):
    model_name = "llava-med-v1.5-mistral-7b"
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_4bit=True if args.quantize == "enabled" else False,
    )

    if args.use_caption == "enabled":
        rag_prompt_template = """
        You are a Vision-Language Model designed to answer questions based on medical images and relevant medical reports retrieved from medical database.

        Instructions:
        - Use the visual details from the query image as the primary context to answer the question.
        - Cross-reference the retrieved reports and the image description to improve the accuracy of your answer.
        - Provide a concise, factual answer relevant to the question.

        Retrieved Reports: {context}
        Image Description: {caption}
        Question: {question}
        """

        prompt_template = """Image Description: {caption} Question: {question}"""

    else:
        rag_prompt_template = """
        You are a Vision-Language Model designed to answer questions based on medical images and relevant medical reports retrieved from medical database.

        Instructions:
        - Use the visual details from the query image as the primary context to answer the question.
        - Cross-reference the retrieved reports to improve the accuracy of your answer.
        - Provide a concise, factual answer relevant to the question.

        Retrieved Reports: {context}
        Question: {question}
        """

        prompt_template = """{question}"""

    return run_queries(
        args=args,
        vlm_args=vlm_args,
        model_name=model_name,
        model=model,
        processor=(processor, tokenizer),
        rag_prompt_template=rag_prompt_template,
        prompt_template=prompt_template,
    )


def process_image_llavamed(image, image_processor):
    args = {"image_aspect_ratio": "pad"}
    return process_images([image], image_processor, args)


def create_prompt_llavamed(prompt: str, conv_mode="mistral_instruct"):
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt(), conv


def run_llavamed(
    model, image_processor, tokenizer, raw_image, prompt, max_new_tokens=100
):
    image_tensor = process_image_llavamed(raw_image, image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    prompt, conv = create_prompt_llavamed(prompt)
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    temperature = 0.01

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()


def init_dragonfly(args, vlm_args):
    model_name = "Dragonfly"
    model_path = "togethercomputer/Llama-3.1-8B-Dragonfly-Med-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    image_processor = clip_processor.image_processor
    processor = DragonflyProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        image_encoding_style="llava-hd",
    )

    model = DragonflyForCausalLM.from_pretrained(model_path)
    model = model.to(torch.bfloat16)
    model = model.to(vlm_args["device"])

    if args.use_caption == "enabled":
        rag_prompt_template = """
        You are a Vision-Language Model designed to answer questions based on medical images and relevant medical reports retrieved from medical database.

        Instructions:
        - Use the visual details from the query image as the primary context to answer the question.
        - Cross-reference the retrieved reports and the image description to improve the accuracy of your answer.
        - Provide a concise, factual answer relevant to the question.

        Retrieved Reports: {context}
        Image Description: {caption}
        Question: {question}
        """

        prompt_template = """

        Image Description: {caption}

        Question: {question}
        
        """

    else:
        # rag_prompt_template = """
        # You are a Vision-Language Model designed to answer questions based on medical images and relevant medical reports retrieved from medical database.

        # Instructions:
        # - Use the visual details from the query image as the primary context to answer the question.
        # - Cross-reference the retrieved reports to improve the accuracy of your answer.
        # - Provide a concise, factual answer relevant to the question.

        # Retrieved Reports: {context}
        # Question: {question}
        # """

        # PROMPT
        rag_prompt_template = """
        You are a medical Vision-Language Model. Use the query image and the retrieved medical reports to answer the question accurately and concisely.

        - Focus on the visual details from the image.
        - Cross-check with the reports for accuracy.
        - Keep your answer short, factual, and relevant.

        **Retrieved Reports:** {context}
        **Question:** {question}
        """

        prompt_template = """

        {question}
        
        """

    return run_queries(
        args=args,
        vlm_args=vlm_args,
        model_name=model_name,
        model=model,
        processor=(processor, tokenizer),
        rag_prompt_template=rag_prompt_template,
        prompt_template=prompt_template,
    )


def text_process_dragonfly(text, system_prompt=""):
    instruction = f"{system_prompt} {text}" if system_prompt else text
    prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    return prompt


def run_dragonfly(vlm_args, model, processor, tokenizer, image, prompt):
    images = [image]
    # images = [None] # if you do not want to pass any images
    prompt = text_process_dragonfly(text=prompt)

    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        is_generate=True,
    )
    inputs = inputs.to(vlm_args["device"])

    temperature = 0.01

    with torch.inference_mode():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=vlm_args["max_new_tokens"],
            eos_token_id=tokenizer.encode("<|eot_id|>"),
            do_sample=temperature > 0,
            temperature=temperature,
            use_cache=True,
        )

    result = processor.batch_decode(generation_output, skip_special_tokens=False)
    result = (
        result[0]
        .replace("<|reserved_special_token_0|>", "")
        .replace("<|reserved_special_token_1|>", "")
    )

    # Extract the assistant's response using regex
    match = re.search(
        r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*(.*?)(<\|eot_id\|>|$)",
        result,
        re.DOTALL,
    )

    # Get the extracted response
    if match:
        assistant_response = match.group(1).strip()
        return assistant_response
    else:
        return "No response found."
