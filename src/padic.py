from dataclasses import dataclass
from typing import Iterable

from src.schemas import Claim

P = 3
POWERS = (1, P, P * P)

PATH_LABELS = {
    0: ("системный/документальный", "свидетельство", "правило/интерпретация"),
    1: ("установленный факт", "сообщение/предположение", "спорное/исключение"),
    2: ("до события", "во время события", "после/неизвестно"),
}

VALID_LEVELS = {
    0: {0, 1, 2},
    1: {0, 1, 2},
    2: {0, 1, 2},
}


@dataclass(frozen=True)
class PadicPath:
    a0: int
    a1: int
    a2: int

    def __post_init__(self) -> None:
        for idx, value in enumerate(self.as_tuple()):
            if value not in VALID_LEVELS[idx]:
                raise ValueError(
                    f"Недопустимое значение уровня a{idx}={value!r}; "
                    f"ожидается одно из {sorted(VALID_LEVELS[idx])}"
                )

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.a0, self.a1, self.a2)

    def code(self) -> int:
        return self.a0 * POWERS[0] + self.a1 * POWERS[1] + self.a2 * POWERS[2]

    def label(self) -> str:
        return (
            f"[{self.a0},{self.a1},{self.a2}] — "
            f"{PATH_LABELS[0][self.a0]}, "
            f"{PATH_LABELS[1][self.a1]}, "
            f"{PATH_LABELS[2][self.a2]}"
        )


def path_from_claim(claim: Claim) -> PadicPath:
    raw = getattr(claim, "path", None)
    if raw is None:
        raise ValueError(f"Claim {claim.claim_id}: отсутствует поле path.")

    if isinstance(raw, PadicPath):
        return raw

    if isinstance(raw, (list, tuple)):
        if len(raw) != 3:
            raise ValueError(
                f"Claim {claim.claim_id}: path должен содержать ровно 3 уровня, "
                f"получено {raw!r}"
            )
        a0, a1, a2 = raw
        return PadicPath(a0=int(a0), a1=int(a1), a2=int(a2))

    raise TypeError(
        f"Claim {claim.claim_id}: path должен быть списком/кортежем из 3 чисел, "
        f"получено {type(raw).__name__}"
    )


def encode(claim: Claim) -> int:
    return path_from_claim(claim).code()


def encode_all(claims: Iterable[Claim]) -> dict[str, PadicPath]:
    out: dict[str, PadicPath] = {}
    for c in claims:
        if c.claim_id in out:
            raise ValueError(f"Дублирующийся claim_id: {c.claim_id}")
        out[c.claim_id] = path_from_claim(c)
    return out


def distance(
    path_x: PadicPath | tuple[int, int, int],
    path_y: PadicPath | tuple[int, int, int],
) -> float:
    px = path_x.as_tuple() if isinstance(path_x, PadicPath) else tuple(path_x)
    py = path_y.as_tuple() if isinstance(path_y, PadicPath) else tuple(path_y)

    if len(px) != 3 or len(py) != 3:
        raise ValueError("p-adic path должен содержать ровно 3 уровня")

    if px == py:
        return 0.0

    k = 0
    for a, b in zip(px, py):
        if a == b:
            k += 1
        else:
            break
    return float(P ** (-k))


def is_ultrametric(
    path_x: PadicPath | tuple[int, int, int],
    path_y: PadicPath | tuple[int, int, int],
    path_z: PadicPath | tuple[int, int, int],
) -> bool:
    return distance(path_x, path_z) <= max(distance(path_x, path_y), distance(path_y, path_z))


def group_by_branch(
    claim_ids: Iterable[str],
    paths: dict[str, PadicPath],
) -> dict[PadicPath, list[str]]:
    groups: dict[PadicPath, list[str]] = {}
    for cid in claim_ids:
        if cid not in paths:
            raise KeyError(f"Нет p-adic пути для claim_id={cid}")
        groups.setdefault(paths[cid], []).append(cid)
    return groups


def branches_as_dict(
    groups: dict[PadicPath, list[str]],
) -> dict[tuple[int, int, int], list[str]]:
    """Приводит результат group_by_branch к виду, удобному для JSON/Streamlit."""
    return {path.as_tuple(): cids for path, cids in groups.items()}