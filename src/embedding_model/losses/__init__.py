"""Public training objectives."""

from embedding_model.losses.contrastive import InfoNCELoss, MultipleNegativesRankingLoss
from embedding_model.losses.cosine_regression import CosineRegressionLoss
from embedding_model.losses.distillation import DistillationLoss
from embedding_model.losses.triplet import TripletEmbeddingLoss

__all__ = [
    "CosineRegressionLoss",
    "DistillationLoss",
    "InfoNCELoss",
    "MultipleNegativesRankingLoss",
    "TripletEmbeddingLoss",
]
