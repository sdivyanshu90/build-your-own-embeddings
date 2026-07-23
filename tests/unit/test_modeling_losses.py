from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional
from torch.testing import assert_close

from embedding_model.losses import (
    CosineRegressionLoss,
    DistillationLoss,
    InfoNCELoss,
    MultipleNegativesRankingLoss,
    TripletEmbeddingLoss,
)
from embedding_model.modeling.pooling import Pooling
from embedding_model.modeling.similarity import (
    cosine_similarity,
    euclidean_distance,
    similarity_matrix,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("mean", [[2.0, 3.0]]),
        ("cls", [[1.0, 2.0]]),
        ("max", [[3.0, 4.0]]),
        ("mean_sqrt_length", [[4.0 / 2**0.5, 6.0 / 2**0.5]]),
    ],
)
def test_pooling_ignores_padding(strategy: str, expected: list[list[float]]) -> None:
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert Pooling(strategy)(hidden, mask) == pytest.approx(torch.tensor(expected))


def test_pooling_rejects_fully_padded_rows() -> None:
    with pytest.raises(ValueError, match="fully padded"):
        Pooling("mean")(torch.ones(1, 2, 3), torch.zeros(1, 2))


def test_similarity_matches_hand_calculation() -> None:
    left = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    right = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert cosine_similarity(left, right).tolist() == pytest.approx([0.0, 2**-0.5])
    assert euclidean_distance(left, right).tolist() == pytest.approx([2**0.5, 1.0])
    matrix = similarity_matrix(torch.eye(2), torch.eye(2), temperature=0.5)
    assert_close(matrix, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))


def test_multiple_negatives_matches_cross_entropy_and_has_finite_gradients() -> None:
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    loss = MultipleNegativesRankingLoss(temperature=1.0)(anchors, positives)
    expected = functional.cross_entropy(torch.eye(2), torch.arange(2))
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert anchors.grad is not None and torch.isfinite(anchors.grad).all()
    assert positives.grad is not None and torch.isfinite(positives.grad).all()


def test_info_nce_triplet_regression_and_distillation_are_real_objectives() -> None:
    anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    positives = anchors.detach().clone()
    negatives = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])
    info = InfoNCELoss(temperature=1.0)(anchors, positives, negatives)
    expected_logits = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    expected_info = functional.cross_entropy(expected_logits, torch.zeros(2, dtype=torch.long))
    assert info.item() == pytest.approx(expected_info.item())

    triplet = TripletEmbeddingLoss(margin=0.2)(
        anchors, positives, torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    )
    assert triplet.item() == pytest.approx(0.0)

    regression = CosineRegressionLoss()(anchors, positives, torch.ones(2))
    assert regression.item() == pytest.approx(0.0)

    student = torch.tensor([[0.8, 0.2], [0.2, 0.8]], requires_grad=True)
    teacher = torch.eye(2)
    distillation = DistillationLoss(distribution_weight=0.5)(student, teacher)
    distillation.backward()
    assert distillation.item() > 0
    assert student.grad is not None and torch.isfinite(student.grad).all()
