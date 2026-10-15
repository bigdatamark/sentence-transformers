# [MODULE DOC] TSDAE denoising auto-encoder loss: the sentence embedding of a damaged input is fed as the only cross-attention
# memory to a Hugging Face causal-LM decoder, which must reconstruct the original input token by token. Includes a
# helper that ties encoder parameters to the decoder's.
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import StaticEmbedding

logger = logging.getLogger(__name__)


def _tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module) -> None:
    # [FUNCTION DOC] HELPER: Tie weights by name: for each decoder parameter, if the encoder has a submodule with the same dotted
    # module path and an attribute of the same name, replace that encoder attribute with the decoder parameter
    # object (shared storage; the decoder's values win). Decoder-only parameters are skipped. Logs a warning if
    # nothing was tied. Mutates the encoder.
    """Tie the encoder's parameters to the decoder's: each encoder parameter is replaced by the
    identically-named decoder parameter, so the two share storage and the decoder's values become
    the shared initialization.

    The encoder and decoder must share an architecture for the parameter names to line up.
    Decoder-only parameters, such as the cross-attention weights, have no encoder counterpart
    and are skipped.
    """
    encoder_modules = dict(encoder.named_modules())
    tied = False
    for name, param in decoder.named_parameters():
        module_name, _, attr = name.rpartition(".")
        encoder_module = encoder_modules.get(module_name)
        if encoder_module is not None and hasattr(encoder_module, attr):
            setattr(encoder_module, attr, param)  # encoder param now references the decoder's
            tied = True
    if not tied:
        logger.warning("No encoder weights were tied to the decoder; are they the same architecture?")


class DenoisingAutoEncoderLoss(nn.Module):
    # [CLASS DOC] Loss module. Attributes: encoder (the SentenceTransformer, used at inference), tokenizer_encoder,
    # tokenizer_decoder, need_retokenization (true when the encoder tokenizer is not an instance of the decoder
    # tokenizer's class), decoder (causal LM configured as a decoder with cross-attention).
    # KNOWN ISSUE: The docstring example's noise function keeps each word with probability del_ratio instead of deleting it with that probability, so it deletes about 40% rather than 60% of the words.
    def __init__(
        self, model: SentenceTransformer, decoder_name_or_path: str | None = None, tie_encoder_decoder: bool = True
    ) -> None:
        # [FUNCTION DOC] Raises ValueError for models whose first module is a static embedding. Determines the encoder's checkpoint
        # name. Asserts a decoder path is given when not tying. When tying, any given decoder path is ignored with a
        # warning and the encoder checkpoint is used. Loads the decoder tokenizer and config (network/disk), marks the
        # config as decoder with cross-attention and loads the causal LM (logs an error and re-raises ValueError if the
        # architecture cannot be a decoder). Asserts equal hidden sizes. If the decoder tokenizer has no pad token, uses
        # its end-of-sequence token for padding (also in the decoder config). Warns if the encoder vocabulary size differs
        # from a freshly loaded tokenizer's. When tying: asserts no retokenization is needed, adopts the encoder tokenizer
        # and resizes decoder embeddings if vocab sizes differ (with a warning), then ties the encoder transformer's
        # weights to the decoder's base model.
        r"""
        This loss expects as input pairs of damaged inputs and the corresponding original ones.
        During training, the decoder reconstructs the original inputs from the encoded embeddings.
        Here the argument 'decoder_name_or_path' indicates the pretrained model (supported by Hugging Face) to be used as the decoder.
        Since decoding process is included, here the decoder should have a class called XXXLMHead (in the context of Hugging Face's Transformers).
        The 'tie_encoder_decoder' flag indicates whether to tie the trainable parameters of encoder and decoder,
        which is shown beneficial to model performance while limiting the amount of required memory.
        Only when the encoder and decoder are from the same architecture, can the flag 'tie_encoder_decoder' work.

        The data generation process (i.e. the 'damaging' process) has already been implemented in ``DenoisingAutoEncoderDataset``,
        allowing you to only provide regular inputs.

        Args:
            model (SentenceTransformer): The SentenceTransformer model.
            decoder_name_or_path (str, optional): Model name or path for initializing a decoder (compatible with Hugging Face's Transformers). Defaults to None.
            tie_encoder_decoder (bool): Whether to tie the trainable parameters of encoder and decoder. Defaults to True.

        References:
            * TSDAE paper: https://huggingface.co/papers/2104.06979
            * `Unsupervised Learning > TSDAE <../../../examples/sentence_transformer/unsupervised_learning/TSDAE/README.html>`_

        Requirements:
            1. The decoder should have a class called XXXLMHead (in the context of Hugging Face's Transformers)
            2. Should use a large corpus

        Inputs:
            +------------------------------------------------------+--------+
            | Inputs                                               | Labels |
            +======================================================+========+
            | (damaged\_input, original\_input) pairs              | none   |
            +------------------------------------------------------+--------+
            | input fed through ``DenoisingAutoEncoderDataset``    | none   |
            +------------------------------------------------------+--------+

        Example:
            ::

                import random
                from datasets import Dataset
                from nltk import word_tokenize
                from nltk.tokenize.treebank import TreebankWordDetokenizer
                from sentence_transformers import SentenceTransformer
                from sentence_transformers.sentence_transformer.losses import DenoisingAutoEncoderLoss
                from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer

                model_name = "google-bert/bert-base-cased"
                model = SentenceTransformer(model_name)

                def noise_transform(batch, del_ratio=0.6):
                    texts = batch["text"]
                    noisy_texts = []
                    for text in texts:
                        words = word_tokenize(text)
                        n = len(words)
                        if n == 0:
                            noisy_texts.append(text)
                            continue

                        kept_words = [word for word in words if random.random() < del_ratio]
                        # Guarantee that at least one word remains
                        if len(kept_words) == 0:
                            noisy_texts.append(random.choice(words))
                            continue

                        noisy_texts.append(TreebankWordDetokenizer().detokenize(kept_words))
                    return {"noisy": noisy_texts, "text": texts}

                train_sentences = [
                    "First training sentence", "Second training sentence", "Third training sentence", "Fourth training sentence",
                ]

                train_dataset = Dataset.from_dict({"text": train_sentences})
                train_dataset.set_transform(transform=lambda batch: noise_transform(batch), columns=["text"], output_all_columns=True)
                train_loss = DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)
                trainer = SentenceTransformerTrainer(model=model, train_dataset=train_dataset, loss=train_loss)
                trainer.train()
        """
        super().__init__()

        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "DenoisingAutoEncoderLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding."
            )

        self.encoder = model  # This will be the final model used during the inference time.
        self.tokenizer_encoder = model.tokenizer

        encoder_name_or_path = model.transformers_model.config._name_or_path
        if decoder_name_or_path is None:
            assert tie_encoder_decoder, (
                "Must indicate the decoder_name_or_path argument when tie_encoder_decoder=False!"
            )
        if tie_encoder_decoder:
            if decoder_name_or_path:
                logger.warning("When tie_encoder_decoder=True, the decoder_name_or_path will be invalid.")
            decoder_name_or_path = encoder_name_or_path

        self.tokenizer_decoder = AutoTokenizer.from_pretrained(decoder_name_or_path)
        self.need_retokenization = not isinstance(self.tokenizer_encoder, type(self.tokenizer_decoder))

        decoder_config = AutoConfig.from_pretrained(decoder_name_or_path)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        kwargs_decoder = {"config": decoder_config}
        try:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name_or_path, **kwargs_decoder)
        except ValueError as e:
            logger.error(
                f'Model name or path "{decoder_name_or_path}" does not support being as a decoder. Please make sure the decoder model has an "XXXLMHead" class.'
            )
            raise e
        assert model.transformers_model.config.hidden_size == decoder_config.hidden_size, "Hidden sizes do not match!"
        if self.tokenizer_decoder.pad_token is None:
            # Needed by GPT-2, etc.
            self.tokenizer_decoder.pad_token = self.tokenizer_decoder.eos_token
            self.decoder.config.pad_token_id = self.decoder.config.eos_token_id

        if len(AutoTokenizer.from_pretrained(encoder_name_or_path)) != len(self.tokenizer_encoder):
            logger.warning(
                "WARNING: The vocabulary of the encoder has been changed. One might need to change the decoder vocabulary, too."
            )

        if tie_encoder_decoder:
            assert not self.need_retokenization, "The tokenizers should be the same when tie_encoder_decoder=True."
            if len(self.tokenizer_encoder) != len(self.tokenizer_decoder):  # The vocabulary has been changed.
                self.tokenizer_decoder = self.tokenizer_encoder
                self.decoder.resize_token_embeddings(len(self.tokenizer_decoder))
                logger.warning(
                    "Since the encoder vocabulary has been changed and --tie_encoder_decoder=True, now the new vocabulary has also been used for the decoder."
                )
            # Tie the encoder and decoder so they share weights during training.
            decoder_base = self.decoder._modules[self.decoder.base_model_prefix]
            _tie_encoder_decoder_weights(model.transformers_model, decoder_base)

    def retokenize(self, sentence_features: dict[str, Tensor]) -> dict[str, Tensor]:
        # [FUNCTION DOC] Decode the encoder token ids back to text (skipping special tokens) and re-tokenize with the decoder tokenizer
        # (padding, longest-first truncation, tensors), moved to the original device. Returns the new feature dict.
        input_ids = sentence_features["input_ids"]
        device = input_ids.device
        sentences_decoded = self.tokenizer_encoder.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        retokenized = self.tokenizer_decoder(
            sentences_decoded, padding=True, truncation="longest_first", return_tensors="pt", max_length=None
        ).to(device)
        return retokenized

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # [FUNCTION DOC] Unpack exactly two feature dicts (damaged source, original target). Retokenize the target if needed. Embed the
        # source to sentence embeddings. Decoder inputs are the target ids without the last token; labels are the target
        # ids shifted by one. Run the decoder with the embedding as a single-position encoder hidden state (attention mask
        # from the source's first mask column) and without caching. Returns token-level cross-entropy, ignoring the decoder
        # pad token id. Labels passed in are ignored.
        source_features, target_features = tuple(sentence_features)
        if self.need_retokenization:
            # since the sentence_features here are all tokenized by encoder's tokenizer,
            # retokenization by the decoder's one is needed if different tokenizers used
            target_features = self.retokenize(target_features)
        reps = self.encoder(source_features)["sentence_embedding"]  # (bsz, hdim)

        # Prepare input and output
        target_length = target_features["input_ids"].shape[1]
        decoder_input_ids = target_features["input_ids"].clone()[:, : target_length - 1]
        label_ids = target_features["input_ids"][:, 1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=None,
            attention_mask=None,
            encoder_hidden_states=reps[:, None],  # (bsz, hdim) -> (bsz, 1, hdim)
            encoder_attention_mask=source_features["attention_mask"][:, 0:1],
            labels=None,
            return_dict=None,
            use_cache=False,
        )

        # Calculate loss
        lm_logits = decoder_outputs[0]
        ce_loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer_decoder.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), label_ids.reshape(-1))
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        # [FUNCTION DOC] Return a config dict with the decoder's checkpoint name and the retokenization flag.
        return {
            "decoder_name_or_path": self.decoder.config._name_or_path,
            "need_retokenization": self.need_retokenization,
        }

    @property
    def citation(self) -> str:
        # [FUNCTION DOC] Return the BibTeX citation for the TSDAE paper.
        return """
@inproceedings{wang-2021-TSDAE,
    title = "TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning",
    author = "Wang, Kexin and Reimers, Nils and Gurevych, Iryna",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    pages = "671--688",
    url = "https://arxiv.org/abs/2104.06979",
}
"""
