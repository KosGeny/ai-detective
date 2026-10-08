import itertools
import pytest
from src.padic import PadicPath, distance, is_ultrametric, path_from_claim


def test_identical_paths():
    p = PadicPath(a0=1, a1=2, a2=0)
    assert distance(p, p) == 0.0


def test_identical_paths_from_claims_have_zero_distance(make_claim):
    c1 = make_claim(claim_id="C01", path=(1, 2, 0))
    c2 = make_claim(claim_id="C02", path=(1, 2, 0))
    p1 = path_from_claim(c1)
    p2 = path_from_claim(c2)
    assert p1.as_tuple() == p2.as_tuple()
    assert distance(p1, p2) == 0.0


def test_difference_only_on_a0_gives_distance_1():
    x = PadicPath(a0=0, a1=0, a2=0)
    y = PadicPath(a0=1, a1=0, a2=0)
    assert distance(x, y) == 1.0


def test_difference_only_on_a1_gives_distance_third():
    x = PadicPath(a0=0, a1=0, a2=0)
    y = PadicPath(a0=0, a1=1, a2=0)
    assert distance(x, y) == pytest.approx(1 / 3)


def test_difference_only_on_a2_gives_distance_ninth():
    x = PadicPath(a0=0, a1=0, a2=0)
    y = PadicPath(a0=0, a1=0, a2=1)
    assert distance(x, y) == pytest.approx(1 / 9)


def test_ultrametric_inequality_holds_for_all_triples():
    all_paths = [PadicPath(a0=a, a1=b, a2=c)
                 for a, b, c in itertools.product((0, 1, 2), repeat=3)]

    for x, y, z in itertools.product(all_paths, repeat=3):
        d_xz = distance(x, z)
        d_xy = distance(x, y)
        d_yz = distance(y, z)
        assert d_xz <= max(d_xy, d_yz) + 1e-12, (x, y, z, d_xz, d_xy, d_yz)


def test_ultrametric_helper_returns_true_on_known_triple():
    x = PadicPath(a0=0, a1=0, a2=0)
    y = PadicPath(a0=0, a1=0, a2=1)
    z = PadicPath(a0=0, a1=0, a2=2)
    assert is_ultrametric(x, y, z) is True