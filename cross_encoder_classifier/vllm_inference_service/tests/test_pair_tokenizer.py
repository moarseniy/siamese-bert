from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from app.pair_tokenizer import PairTokenizer


VOCAB = {
    "<s>": 0,
    "</s>": 1,
    "[UNK]": 2,
    "первый": 3,
    "текст": 4,
    "второй": 5,
}


def save_tokenizer(path: Path, bert_type_ids: bool) -> None:
    tokenizer = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    if bert_type_ids:
        pair_template = "<s>:0 $A:0 </s>:0 $B:1 </s>:1"
    else:
        pair_template = "<s>:0 $A:0 </s>:0 </s>:0 $B:0 </s>:0"
    tokenizer.post_processor = TemplateProcessing(
        single="<s>:0 $A:0 </s>:0",
        pair=pair_template,
        special_tokens=[("<s>", 0), ("</s>", 1)],
    )
    tokenizer.save(str(path))


def test_accepts_bge_style_pair_tokenizer(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    save_tokenizer(path, bert_type_ids=False)

    tokenizer = PairTokenizer(path, max_length=32)
    (ids,) = tokenizer.encode_batch([("первый текст", "второй текст")])

    assert ids == [0, 3, 4, 1, 1, 5, 4, 1]


def test_rejects_bert_token_type_ids(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    save_tokenizer(path, bert_type_ids=True)

    with pytest.raises(ValueError, match="incompatible with BAAI/bge-reranker"):
        PairTokenizer(path, max_length=32)
