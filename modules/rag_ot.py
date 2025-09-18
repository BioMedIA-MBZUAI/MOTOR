# https://pythonot.github.io/

import numpy as np
import torch
from sentence_transformers import util
from sentencex import segment

import ot


class RAGOptimalTransport:
    def __init__(
        self,
        alpha=0.2,
        beta=0.3,
        delta=0.5,
        bert_model=None,
        bert_tokenizer=None,
        image_model=None,
        image_preprocessor=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_model = bert_model.to(self.device)
        self.bert_tokenizer = bert_tokenizer
        self.image_model = image_model.to(self.device)
        self.image_preprocessor = image_preprocessor
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def get_text_embedding(self, text):
        text_processed = self.bert_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        return self.bert_model(**text_processed).pooler_output

    def adjust_bbox_to_image_size(self, img_width, img_height, normalized_bbox):
        """
        Adjusts the bounding box from normalized coordinates to pixel coordinates.

        Args:
        - img_width (int): Width of the image.
        - img_height (int): Height of the image.
        - normalized_bbox (list or tuple): Bounding box in normalized form [x_min, y_min, x_max, y_max].

        Returns:
        - tuple: Bounding box in pixel coordinates (x_topleft, y_topleft, x_bottomright, y_bottomright).
        """

        # Convert normalized bbox to pixel coordinates
        x_topleft = int(normalized_bbox[0] * img_width)
        y_topleft = int(normalized_bbox[1] * img_height)
        x_bottomright = int(normalized_bbox[2] * img_width)
        y_bottomright = int(normalized_bbox[3] * img_height)
        return x_topleft, y_topleft, x_bottomright, y_bottomright

    def compute_spatial_cost(self, box1, box2, image1, image2):
        if not box1 and not box2:
            return 0.0  # No bounding boxes (most similar, least cost)
        elif not box1 or not box2:
            return 1.0  # Missing bounding box in one of the images (least similar, highest cost)

        img_width1, img_height1 = image1.size
        img_width2, img_height2 = image2.size

        # Adjust bbox for original image size
        box1_minx, box1_miny, box1_maxx, box1_maxy = self.adjust_bbox_to_image_size(
            img_width=img_width1, img_height=img_height1, normalized_bbox=box1[0]
        )
        box2_minx, box2_miny, box2_maxx, box2_maxy = self.adjust_bbox_to_image_size(
            img_width=img_width2, img_height=img_height2, normalized_bbox=box2[0]
        )

        # Crop the region of interest
        try:
            cropped_image1 = image1.crop((box1_minx, box1_miny, box1_maxx, box1_maxy))
            image_preprocessed1 = self.image_preprocessor(
                images=cropped_image1, return_tensors="pt"
            ).to(self.device)
            image_encoding1 = self.image_model(**image_preprocessed1).pooler_output

            cropped_image2 = image2.crop((box2_minx, box2_miny, box2_maxx, box2_maxy))
            image_preprocessed2 = self.image_preprocessor(
                images=cropped_image2, return_tensors="pt"
            ).to(self.device)
            image_encoding2 = self.image_model(**image_preprocessed2).pooler_output

            return 1 - util.pytorch_cos_sim(image_encoding1, image_encoding2).item()
        except:
            # Compute centroids
            box1 = box1[0]
            box2 = box2[0]
            cx1 = (box1[0] + box1[2]) / 2
            cy1 = (box1[1] + box1[3]) / 2
            cx2 = (box2[0] + box2[2]) / 2
            cy2 = (box2[1] + box2[3]) / 2

            # Compute Euclidean distance
            distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            return distance

    def create_cost_matrix(
        self,
        question,
        query_caption,
        query_image,
        retrieved_report,
        retrieved_caption,
        retrieved_image,
    ):
        n, m = len(query_caption), len(retrieved_caption)

        # Extract captions and bounding boxes
        q_captions, q_bboxes = zip(*query_caption)
        r_captions, r_bboxes = zip(*retrieved_caption)

        # Compute textual costs
        q_embeddings = (
            self.get_text_embedding(q_captions).clone().detach().to(self.device)
        )
        r_embeddings = (
            self.get_text_embedding(r_captions).clone().detach().to(self.device)
        )
        text_cost_matrix = 1 - util.pytorch_cos_sim(q_embeddings, r_embeddings)

        # Compute spatial costs
        spatial_cost_matrix = torch.zeros((n, m), device=self.device)
        for i in range(n):
            for j in range(m):
                spatial_cost_matrix[i, j] = self.compute_spatial_cost(
                    q_bboxes[i],
                    r_bboxes[j],
                    query_image,
                    retrieved_image,
                )

        # Compute question cost
        question_cost = (
            1
            - util.pytorch_cos_sim(
                self.get_text_embedding(question),
                self.get_text_embedding(retrieved_report),
            ).item()
        )

        # Combine textual and spatial costs
        cost_matrix = (
            self.alpha * question_cost
            + self.beta * text_cost_matrix
            + self.delta * spatial_cost_matrix
        )

        return cost_matrix

    def compute_optimal_transport_cost(
        self,
        question,
        query_caption,
        query_image,
        retrieved_reports,
        retrieved_captions,
        retrieved_images,
    ):
        results = {}

        # Uniform distribution for query captions
        query_weights = torch.ones(len(query_caption), device=self.device) / len(
            query_caption
        )

        for idx, retrieved_caption in enumerate(retrieved_captions):
            # Create cost matrix
            cost_matrix = self.create_cost_matrix(
                question,
                query_caption,
                query_image,
                retrieved_reports[idx],
                retrieved_caption,
                retrieved_images[idx],
            ).to(self.device)

            # Uniform distribution for retrieved captions
            retrieved_weights = (
                torch.ones(cost_matrix.shape[1], device=self.device)
                / cost_matrix.shape[1]
            )

            ot_cost = ot.sinkhorn2(
                query_weights,
                retrieved_weights,
                cost_matrix,
                reg=1,
            )
            results[f"retrieved_item_{idx}"] = ot_cost

        return results
