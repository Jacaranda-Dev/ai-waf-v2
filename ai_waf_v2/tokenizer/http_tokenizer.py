"""
ai_waf_v2.tokenizer.http_tokenizer
-------------------------------
Train and load the custom BPE tokenizer for HTTP requests (Track B).

Key design decisions:
- Byte-level pre-tokenisation: handles arbitrary binary content in HTTP bodies
- HTTP-aware split regex: treats ?, &, =, /, . as split boundaries so
  URL structure is captured at the token level
- Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
- Vocab size: 8000 (configurable) — balances coverage vs embedding table size

Usage
-----
    from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer

    # Train from scratch on a corpus file
    tok = HttpTokenizer.train(
        corpus_path="data/normalized/train_corpus.txt",
        vocab_size=8000,
        output_dir="tokenizers/track_b",
    )

    # Load from saved tokenizer
    tok = HttpTokenizer.load("tokenizers/track_b")

    # Encode
    enc = tok.encode("GET /api/v1/users?id=1 HTTP/1.1")
    print(enc.ids, enc.tokens)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Encoding


# HTTP-aware pre-tokenisation regex
# Splits on: whitespace, ?, &, =, /, ., ;, :, (, ), [, ], {, }, "
# but keeps each delimiter as its own token so the model sees structure
_HTTP_SPLIT_REGEX = (
    r"(?i:"
    r"\s+"            # whitespace
    r"|[?&=/.;:(){}\[\]\"']"   # URL and HTTP delimiters
    r"|%[0-9a-f]{2}"  # URL-encoded bytes (kept together)
    r"|\w+"           # alphanumeric runs
    r"|."             # any other single character
    r")"
)

# Special tokens — positions are fixed (match config/pipeline.yaml)
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD_TOKEN_ID   = 0
UNK_TOKEN_ID   = 1
CLS_TOKEN_ID   = 2
SEP_TOKEN_ID   = 3
MASK_TOKEN_ID  = 4


class HttpTokenizer:
    """
    Wrapper around a HuggingFace ``tokenizers.Tokenizer`` trained on
    HTTP request data with BPE.

    Attributes
    ----------
    tokenizer : tokenizers.Tokenizer  — the underlying fast tokenizer
    vocab_size : int
    seq_len    : int — configured max sequence length
    """

    def __init__(self, tokenizer: object, seq_len: int = 256) -> None:
        self._tok    = tokenizer
        self.seq_len = seq_len

    @classmethod
    def train(
        cls,
        corpus_path: str | Path,
        vocab_size:  int  = 8000,
        output_dir:  str | Path = "tokenizers/track_b",
        seq_len:     int  = 256,
        min_frequency: int = 2,
    ) -> "HttpTokenizer":
        """
        Train a BPE tokenizer on the HTTP corpus and save to disk.

        Parameters
        ----------
        corpus_path   : path to a plain-text file with one HTTP request per line
        vocab_size    : target vocabulary size (including special tokens)
        output_dir    : directory to save tokenizer.json and vocab files
        seq_len       : max sequence length (stored in config, not enforced here)
        min_frequency : minimum token pair frequency for BPE merge

        Returns
        -------
        HttpTokenizer instance
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Split
        from tokenizers.trainers import BpeTrainer
        from tokenizers.normalizers import Lowercase, Sequence as NormSeq
        from tokenizers.processors import TemplateProcessing
        from tokenizers.decoders import BPEDecoder

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build tokenizer with BPE model
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # Normalizer: lowercase (HTTP headers are case-insensitive;
        # method and path are case-sensitive but lowercasing helps generalisation)
        tokenizer.normalizer = NormSeq([Lowercase()])

        # Pre-tokenizer: HTTP-aware split regex
        tokenizer.pre_tokenizer = Split(
            pattern=_HTTP_SPLIT_REGEX,
            behavior="isolated",        # keep delimiters as tokens
        )

        # Decoder
        tokenizer.decoder = BPEDecoder()

        # Trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        tokenizer.train([str(corpus_path)], trainer=trainer)

        # Post-processor: automatically add [CLS] ... [SEP] around sequences
        cls_id = tokenizer.token_to_id("[CLS]")
        sep_id = tokenizer.token_to_id("[SEP]")
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
        )

        # Enable padding and truncation
        pad_id = tokenizer.token_to_id("[PAD]")
        tokenizer.enable_padding(
            pad_id=pad_id, pad_token="[PAD]", length=seq_len
        )
        tokenizer.enable_truncation(max_length=seq_len)

        # Save
        save_path = output_dir / "tokenizer.json"
        tokenizer.save(str(save_path))

        # Also save vocab for inspection
        vocab = tokenizer.get_vocab()
        with (output_dir / "vocab.txt").open("w") as f:
            for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{token}\n")

        return cls(tokenizer, seq_len=seq_len)

    @classmethod
    def load(cls, tokenizer_dir: str | Path, seq_len: int = 256) -> "HttpTokenizer":
        """
        Load a saved tokenizer from disk.

        Parameters
        ----------
        tokenizer_dir : directory containing tokenizer.json
        seq_len       : max sequence length (re-applies padding/truncation)
        """
        from tokenizers import Tokenizer

        path = Path(tokenizer_dir) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found in {tokenizer_dir}. "
                "Run HttpTokenizer.train() first."
            )

        tokenizer = Tokenizer.from_file(str(path))

        # Re-apply padding/truncation (may have been saved without it)
        pad_id = tokenizer.token_to_id("[PAD]") or 0
        tokenizer.enable_padding(
            pad_id=pad_id, pad_token="[PAD]", length=seq_len
        )
        tokenizer.enable_truncation(max_length=seq_len)

        return cls(tokenizer, seq_len=seq_len)

    # ── Public interface ──────────────────────────────

    def encode(self, text: str) -> "Encoding":
        """Encode a single HTTP request string."""
        return self._tok.encode(text)

    def encode_batch(self, texts: list[str]) -> list["Encoding"]:
        """Encode a list of HTTP request strings."""
        return self._tok.encode_batch(texts)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return self._tok.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        return self._tok.id_to_token(token_id)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._tok.token_to_id("[PAD]") or 0

    @property
    def cls_token_id(self) -> int:
        return self._tok.token_to_id("[CLS]") or 2

    def compute_oov_rate(self, texts: list[str]) -> float:
        """
        Compute the fraction of tokens that are [UNK] across all texts.
        A high OOV rate (> 15%) signals poor vocab coverage.
        """
        unk_id = self._tok.token_to_id("[UNK]") or 1
        total = unk = 0

        for enc in self.encode_batch(texts):
            ids    = [i for i in enc.ids if i not in (self.pad_token_id, self.cls_token_id)]
            total += len(ids)
            unk   += sum(1 for i in ids if i == unk_id)

        return unk / max(1, total)

    def compute_token_fertility(self, texts: list[str]) -> float:
        """
        Average number of tokens per input character.
        Lower is better: high fertility means the tokenizer is over-splitting.
        """
        total_tokens = total_chars = 0
        for text, enc in zip(texts, self.encode_batch(texts)):
            n_real = sum(
                1 for i in enc.ids
                if i not in (self.pad_token_id, self.cls_token_id,
                             self._tok.token_to_id("[SEP]"))
            )
            total_tokens += n_real
            total_chars  += max(1, len(text))
        return total_tokens / max(1, total_chars)