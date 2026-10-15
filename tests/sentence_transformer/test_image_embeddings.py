# [MODULE DOC]
# Tests computing image embeddings with a CLIP SentenceTransformer and comparing them with text embeddings.
"""
Compute image embeddings
"""

from __future__ import annotations

import os

from PIL import Image

from sentence_transformers import SentenceTransformer, util


def test_simple_encode(clip_vit_b_32_model: SentenceTransformer) -> None:
    # [FUNCTION DOC]
    # Encodes a sample image of two dogs in snow (read from the examples directory relative to this file) and three captions with the
    # CLIP ViT-B/32 model, computes cosine similarity of the image against each caption, and asserts each score is within 0.01 of
    # reference values (about 0.307, 0.101, 0.109). Reads an image file from disk.
    model = clip_vit_b_32_model
    # Encode an image:
    image_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../examples/sentence_transformer/applications/image-search/two_dogs_in_snow.jpg",
    )
    img_emb = model.encode(Image.open(image_filepath))

    # Encode text descriptions
    text_emb = model.encode(["Two dogs in the snow", "A cat on a table", "A picture of London at night"])

    # Compute cosine similarities
    cos_scores = util.cos_sim(img_emb, text_emb)[0]
    assert abs(cos_scores[0] - 0.3069) < 0.01
    assert abs(cos_scores[1] - 0.1010) < 0.01
    assert abs(cos_scores[2] - 0.1086) < 0.01
