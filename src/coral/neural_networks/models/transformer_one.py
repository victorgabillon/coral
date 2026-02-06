# mypy: ignore-errors
"""Transformer-based neural network model for board evaluation."""

from dataclasses import dataclass
from typing import Any, Literal

import chess
import torch
import torch.nn.functional as functional
from torch import nn

from coral.chi_nn import ChiNN
from coral.neural_networks.nn_model_type import NNModelType

NUMBER_SQUARES = len(chess.SQUARES)
NUMBER_PIECES_TYPES = len(chess.PIECE_TYPES)
NUMBER_COLORS = len(chess.COLORS)
NUMBER_OCCUPANCY_TYPES = (
    NUMBER_PIECES_TYPES * NUMBER_COLORS + 1
)  # could be empty (+1) or one of the pieces in black or white
LEN_SQUARE_TENSOR = NUMBER_SQUARES * NUMBER_OCCUPANCY_TYPES
LEN_ALL_POSSIBLE_TENSOR_INPUT = (
    LEN_SQUARE_TENSOR + 1
)  # +1 for the vector that will bear the output embedding
NUMBER_PARALLEL_TRACKS = NUMBER_SQUARES + 1


@dataclass()
class TransformerArgs:
    """Transformer model hyperparameters."""

    number_occupancy_types: int = NUMBER_OCCUPANCY_TYPES
    len_square_tensor: int = LEN_SQUARE_TENSOR
    number_pieces_types: int = NUMBER_PIECES_TYPES

    type: Literal[NNModelType.TRANSFORMER] = NNModelType.TRANSFORMER
    n_embd: int = 27
    n_head: int = 3
    n_layer: int = 2
    dropout_ratio: float = 0.0

    def __str__(self) -> str:
        """Return a string representation of the TransformerArgs instance."""
        return (
            f"TransformerArgs(n_embd={self.n_embd}, "
            f"n_head={self.n_head}, "
            f"n_layer={self.n_layer}, "
            f"dropout_ratio={self.dropout_ratio})"
        )

    def filename(self) -> str:
        """Generate a filename based on the TransformerArgs instance."""
        return (
            f"transformer_{self.n_embd}embd_"
            f"{self.n_head}head_{self.n_layer}layer_"
            f"{self.dropout_ratio:.2f}dropout"
        )


class Head(nn.Module):
    """one head of self-attention."""

    def __init__(self, n_embd: int, head_size: int, dropout_ratio: float) -> None:
        """Initialize attention head layers."""
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=True)
        self.query = nn.Linear(n_embd, head_size, bias=True)
        self.value = nn.Linear(n_embd, head_size, bias=True)

        self.dropout = nn.Dropout(dropout_ratio)
        # TODO(victor): investigate gap.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, time-step, channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch, time-step, head size).

        """
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        # B, T, C is x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = functional.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out: torch.Tensor = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel."""

    def __init__(
        self, num_heads: int, head_size: int, dropout_ratio: float, n_embd: int
    ) -> None:
        """Initialize multi-head attention layers."""
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(head_size=head_size, n_embd=n_embd, dropout_ratio=dropout_ratio)
                for _ in range(num_heads)
            ]
        )  # (B, T, hs) * num_heads
        self.proj = nn.Linear(
            head_size * num_heads, n_embd
        )  # (B, T, hs) * num_heads -> (B, T, n_embd)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head attention output for the input tensor."""
        out: torch.Tensor = torch.empty(1)  # to make mypy and jit happy
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd: int, dropout_ratio: float) -> None:
        """Initialize the feed-forward network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to the input tensor."""
        a: torch.Tensor = self.net(x)
        return a


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int, dropout_ratio: float) -> None:
        """Initialize a transformer block with attention and feed-forward layers."""
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head, head_size, dropout_ratio=dropout_ratio, n_embd=n_embd
        )  # (B, T, n_embd) -> (B, T, n_embd)
        self.ffwd = FeedFoward(
            n_embd, dropout_ratio=dropout_ratio
        )  # (B, T, n_embd) -> (B, T, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer block forward pass."""
        y = self.sa(self.ln1(x))
        x = x + y
        return x + self.ffwd(self.ln2(x))


class TransformerOne(ChiNN):
    """Transformer neural network implementation for board evaluation."""

    def __init__(self, args: TransformerArgs) -> None:
        """Initialize the TransformerOne model with the provided arguments."""
        super().__init__()

        self.board_embedding_table = nn.Parameter(
            torch.randn(LEN_ALL_POSSIBLE_TENSOR_INPUT, args.n_embd)
        )

        self.blocks = nn.Sequential(
            *[
                Block(args.n_embd, n_head=args.n_head, dropout_ratio=args.dropout_ratio)
                for _ in range(args.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(args.n_embd)  # final layer norm
        self.lm_head = nn.Linear(NUMBER_PARALLEL_TRACKS * args.n_embd, 1)

        self.tan_h = nn.Tanh()

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def init_weights(self) -> None:
        """Initialize model weights."""
        # TODO: fix the weird init_weights logics
        return

    def _init_weights(self, module: Any) -> None:
        """Initialize linear layer weights and biases."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            torch.nn.init.zeros_(module.bias)

    def forward(self, indices: list[list[int]]) -> torch.Tensor:
        """Run the transformer forward pass on indexed input data."""
        # idx and targets are both (B,T) tensor of integers
        y = self.board_embedding_table[
            indices, :
        ]  # (B,len_all_possible_tensor_input,n_embd)
        z = self.blocks(y)  # (B,T,C)
        x: torch.Tensor = self.lm_head(z.flatten(start_dim=1))  # (B,T,vocab_size)

        return self.tan_h(x)

    def get_nn_input(self, node: Any) -> None:
        """Get the input tensor for the given node.

        Args:
            node (Any): Current node.

        Returns:
            None

        """
        raise NotImplementedError(f"to be recoded in {__name__}")

    def print_param(self) -> None:
        """Print the parameters of the model."""
        print(
            "print param not implmeented but is it necessary given the summary thingy in pythorch?"
        )
