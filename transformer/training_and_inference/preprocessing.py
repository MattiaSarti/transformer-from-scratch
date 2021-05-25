"""
Utilities for preprocessing, tokenizing and reconstructing text.
"""


from typing import List

from spacy import load as spacy_load


class Tokenizer:
    """
    Source and target tokenizers.
    """

    def __init__(self, src_language: str = 'de', tgt_language: str = 'en')\
            -> None:
        super(Tokenizer, self).__init__()

        self.src_language_model = spacy_load(src_language)
        self.tgt_language_model = spacy_load(tgt_language)

        self.bos_token = "<s>"  # beginning of sequence token
        self.eos_token = "</s>"  # end of sequence token
        self.unk_token = "<unk>"  # unknown token
        self.padding_token = "<blank>"  # padding token

    def tokenize_src(self, text: str) -> List[str]:
        """
        Tokenize source language text.
        """
        return list(self.src_language_model.tokenizer(text))

    def tokenize_tgt(self, text: str) -> List[str]:
        """
        Tokenize target language text.
        """
        return list(self.tgt_language_model.tokenizer(text))
