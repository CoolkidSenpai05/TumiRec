import math
import pytest
from tumirec.eval.metrics import ranking_metrics


def test_hand_computed_metrics():
    result = ranking_metrics([[3, 1, 2], [4, 5, 6]], [{1, 2}, {4}], ks=[1, 2, 10])
    assert result["Recall@1"] == 0.5
    assert result["Recall@2"] == 0.75
    assert result["Recall@10"] == 1.0
    assert result["NDCG@1"] == 0.5
    discount = 1 / math.log2(3)
    assert result["NDCG@2"] == pytest.approx((discount / (1 + discount) + 1) / 2)
    assert result["NDCG@10"] == pytest.approx(((discount + 0.5) / (1 + discount) + 1) / 2)
    assert result["MRR"] == result["MRR@2"] == 0.75
    assert result["MRR@1"] == 0.5


def test_empty_truth_misses_and_empty_ranking():
    assert all(v == 0 for v in ranking_metrics([[], [1], [2]], [{1}, set(), {3}]).values())


@pytest.mark.parametrize("rankings,truth,ks", [([], [], [1]), ([[1]], [], [1]),
    ([[1, 1]], [{1}], [1]), ([[1]], [{1}], [0]), ([[1]], [{1}], [])])
def test_invalid_metrics(rankings, truth, ks):
    with pytest.raises(ValueError):
        ranking_metrics(rankings, truth, ks)
