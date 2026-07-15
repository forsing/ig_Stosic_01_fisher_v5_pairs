from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
v5: zajednički nosioci (par)
P(y | x∈last ∧ z∈last) za parove iz last (ili min/geom. sredina P(y|x)·P(y|z)), pa Fisher. Stroža uslovnost od max/prosek.

P(y|x,z) parovi iz last → Fisher → next.


IG korak 1 v5 — Fisher na zajedničkoj uslovnosti para nosilaca.

Za par (x,z) iz last: P(y | x∈t ∧ z∈t) iz t→t+1.
rate_y = prosek tih P preko svih parova u last (samo parovi sa dovoljno podrške).
simplex → Fisher; skor (p_cond − p_glob)·√g; next jedna kombinacija.
CSV ceo, seed=39.
"""



import csv
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
MIN_PAIR = 20
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def pair_transition_tables(draws: np.ndarray) -> tuple[dict, dict]:
    """
    pair_count[(a,b)] = #t sa a<b u draw_t
    pair_next[(a,b)][y] = # puta y u next | a,b u t
    indeksi 0..38
    """
    present = np.zeros((len(draws), FRONT_N), dtype=np.uint8)
    for i, d in enumerate(draws):
        for x in d.tolist():
            present[i, int(x) - 1] = 1

    pair_count: dict[tuple[int, int], int] = {}
    pair_next: dict[tuple[int, int], np.ndarray] = {}

    for t in range(len(draws) - 1):
        xs = np.where(present[t] == 1)[0]
        ys = np.where(present[t + 1] == 1)[0]
        for a, b in combinations(xs.tolist(), 2):
            key = (a, b) if a < b else (b, a)
            pair_count[key] = pair_count.get(key, 0) + 1
            if key not in pair_next:
                pair_next[key] = np.zeros(FRONT_N, dtype=np.float64)
            for yi in ys:
                pair_next[key][yi] += 1.0
    return pair_count, pair_next


def conditional_from_last_pairs(
    last: np.ndarray,
    pair_count: dict,
    pair_next: dict,
    min_pair: int = MIN_PAIR,
) -> tuple[np.ndarray, int]:
    """Prosek P(y|x,z) preko parova u last sa count>=min_pair."""
    carriers = sorted(int(x) - 1 for x in last.tolist())
    masses = []
    used = 0
    for a, b in combinations(carriers, 2):
        key = (a, b)
        c = pair_count.get(key, 0)
        if c < min_pair:
            continue
        rates = pair_next[key] / float(c)
        masses.append(rates)
        used += 1
    if not masses:
        # fallback: sve parove bez praga
        for a, b in combinations(carriers, 2):
            key = (a, b)
            c = pair_count.get(key, 0)
            if c <= 0:
                continue
            rates = pair_next[key] / float(c)
            masses.append(rates)
            used += 1
    if not masses:
        mass = np.ones(FRONT_N, dtype=np.float64)
    else:
        mass = np.mean(np.stack(masses, axis=0), axis=0)
    mass = mass + 1e-6
    return mass / mass.sum(), used


def fisher_diagonal(p: np.ndarray) -> np.ndarray:
    return 1.0 / np.clip(p, 1e-18, None)


def number_scores(p_cond: np.ndarray, p_glob: np.ndarray, g: np.ndarray) -> dict[int, float]:
    return {
        i + 1: float((p_cond[i] - p_glob[i]) * np.sqrt(g[i]))
        for i in range(FRONT_N)
    }


def _combo_fit(
    combo: list[int],
    score: dict[int, float],
    target_sum: float,
    pos_means: list[float],
    target_odd: float,
) -> float:
    nums = sorted(combo)
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(
    draws: np.ndarray,
    p_cond: np.ndarray,
    p_glob: np.ndarray,
    g: np.ndarray,
) -> list[int]:
    score = number_scores(p_cond, p_glob, g)
    ranked = sorted(score, key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))

    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))

    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_01_v5(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    p_glob = global_p(draws)
    pair_count, pair_next = pair_transition_tables(draws)
    p_cond, n_pairs = conditional_from_last_pairs(last, pair_count, pair_next)
    g = fisher_diagonal(p_cond)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | MIN_PAIR={MIN_PAIR} | ig_01_v5 Fisher par")
    print(f"last: {last.tolist()} | parova korišćeno: {n_pairs}")
    print()

    anisotropy = float(g.max() / g.min()) if g.min() > 0 else float("inf")
    print("=== p_cond (prosek P(y|x,z)) + Fisher ===")
    print(
        {
            "sum_p": round(float(p_cond.sum()), 6),
            "p_min": float(p_cond.min()),
            "p_max": float(p_cond.max()),
            "g_min": float(g.min()),
            "g_max": float(g.max()),
            "anisotropy": round(anisotropy, 4),
        }
    )
    print()

    score = number_scores(p_cond, p_glob, g)
    ranked = sorted(
        ((n, float(p_cond[n - 1]), float(score[n])) for n in range(1, FRONT_N + 1)),
        key=lambda t: (-t[2], t[0]),
    )
    print("=== top12 po (p_cond − p_glob)·√g ===")
    print([(n, round(pc, 5), round(sc, 5)) for n, pc, sc in ranked[:12]])
    print()

    combo = predict_next(draws, p_cond, p_glob, g)
    print("=== next (ig_01_v5 par Fisher) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_ig_01_v5()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | MIN_PAIR=20 | ig_01_v5 Fisher par
last: [4, 5, 6, 11, 12, 18, 28] | parova korišćeno: 21

=== p_cond (prosek P(y|x,z)) + Fisher ===
{'sum_p': 1.0, 'p_min': 0.02176566669464515, 'p_max': 0.030021276106844134, 'g_min': 33.30970996839218, 'g_max': 45.94391773195823, 'anisotropy': 1.3793}

=== top12 po (p_cond − p_glob)·√g ===
[(26, 0.03002, 0.01883), (30, 0.02647, 0.01372), (12, 0.02727, 0.0135), (23, 0.02991, 0.01164), (35, 0.02784, 0.01127), (5, 0.02736, 0.0109), (21, 0.02718, 0.01075), (38, 0.0277, 0.01044), (8, 0.0298, 0.00981), (32, 0.02782, 0.0084), (4, 0.02633, 0.00758), (27, 0.02528, 0.00593)]

=== next (ig_01_v5 par Fisher) ===
next: [5, 12, 14, 21, 23, 30, 35]
"""
