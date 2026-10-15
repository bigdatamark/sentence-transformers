# [MODULE DOC]
# Script: export sentence-transformers/parallel-sentences-wikimatrix (LASER-filtered) to gzip TSV files
# parallel-sentences/WikiMatrix-en-<lang>-<split>.tsv.gz for de, es, it, fr, ar, tr and splits train/dev.
# Writes stripped "english<TAB>non_english" lines, dropping pairs with an empty side; skips existing files and
# unloadable subsets/splits with a printed message. Prints counts and "---DONE---". Side effects: network, disk, stdout.
"""
This script writes parallel sentence tsv files that can be used to extend existing sentence
embedding models to new languages.

It loads the WikiMatrix parallel corpus (mined from Wikipedia, see
https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix) from the Hugging Face dataset
``sentence-transformers/parallel-sentences-wikimatrix``, which is already filtered by LASER
similarity score. The training procedure can be found in ``make_multilingual.py``, which loads the
same Hugging Face dataset directly.

Further information can be found in our paper:
Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation
https://huggingface.co/papers/2004.09813
"""

import gzip
import os

from datasets import load_dataset

hf_dataset = "sentence-transformers/parallel-sentences-wikimatrix"
source_lang = "en"  # Language our (monolingual) teacher model understands
target_languages = ["de", "es", "it", "fr", "ar", "tr"]  # New languages we want to extend to

output_folder = "parallel-sentences/"
os.makedirs(output_folder, exist_ok=True)

for target_lang in target_languages:
    subset = f"{source_lang}-{target_lang}"
    for split in ("train", "dev"):
        output_filename = os.path.join(output_folder, f"WikiMatrix-{source_lang}-{target_lang}-{split}.tsv.gz")
        if os.path.exists(output_filename):
            continue
        try:
            dataset = load_dataset(hf_dataset, subset, split=split)
        except Exception as e:
            print(f"Skipping {subset} ({split}): {e}")
            continue
        num_written = 0
        with gzip.open(output_filename, "wt", encoding="utf8") as fOut:
            for row in dataset:
                english = row["english"].strip()
                non_english = row["non_english"].strip()
                if english and non_english:
                    fOut.write(f"{english}\t{non_english}\n")
                    num_written += 1
        print(f"Wrote {output_filename}: {num_written} pairs")

print("---DONE---")
