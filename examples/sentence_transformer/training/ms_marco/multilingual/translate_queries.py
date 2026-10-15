# [MODULE DOC]
# Script: machine-translate MS MARCO training queries from English into a target language (first CLI arg, required)
# with EasyNMT "opus-mt". Output: multilingual-data/train_queries.en-<lang>.tsv, one "qid<TAB>translation" line per
# query (tabs in translations replaced by spaces). Resumable: qids already present in the output file are skipped and
# new lines are appended. Queries are the qids in the mteb/msmarco "default" train qrels, with text from the
# "queries" split (qids without text are dropped). Translates as a stream (beam 2, no sentence splitting, chunk 256,
# batch 64), flushing after each line. Side effects: creates the folder, downloads data/models, prints progress.
"""
This script translates the queries in the MS MARCO dataset to the defined target languages.

For machine translation, we use EasyNMT: https://github.com/UKPLab/EasyNMT
You can install it via: pip install easynmt

Usage:
python translate_queries [target_language]
"""

import logging
import os
import sys

from datasets import load_dataset
from easynmt import EasyNMT

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

target_lang = sys.argv[1]
output_folder = "multilingual-data"

output_filename = os.path.join(output_folder, f"train_queries.en-{target_lang}.tsv")
os.makedirs(output_folder, exist_ok=True)


# Does the output file exists? If yes, read it so we can continue the translation
translated_qids = set()
if os.path.exists(output_filename):
    with open(output_filename, encoding="utf8") as fIn:
        for line in fIn:
            splits = line.strip().split("\t")
            translated_qids.add(splits[0])

# Read the MS MARCO dataset

# Train queries that have relevance judgements (the set the original script translated), minus the ones already translated.
train_queries = {}
for row in load_dataset("mteb/msmarco", "default", split="train"):
    qid = str(row["query-id"])
    if qid not in translated_qids:
        train_queries[qid] = None

# Fill in the query texts
for row in load_dataset("mteb/msmarco", "queries", split="queries"):
    qid = str(row["_id"])
    if qid in train_queries:
        train_queries[qid] = row["text"].strip()


qids = [qid for qid in train_queries if train_queries[qid] is not None]
queries = [train_queries[qid] for qid in qids]

# Define our translation model
translation_model = EasyNMT("opus-mt")

print(f"Start translation of {len(queries)} queries.")
print("This can take a while. But you can stop this script at any point")


with open(output_filename, "a" if os.path.exists(output_filename) else "w", encoding="utf8") as fOut:
    for qid, query, translated_query in zip(
        qids,
        queries,
        translation_model.translate_stream(
            queries,
            source_lang="en",
            target_lang=target_lang,
            beam_size=2,
            perform_sentence_splitting=False,
            chunk_size=256,
            batch_size=64,
        ),
    ):
        fOut.write("{}\t{}\n".format(qid, translated_query.replace("\t", " ")))
        fOut.flush()
