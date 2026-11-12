"""Személyre szabott HR-korrekciós modellek tesztelése.

A `comparison.csv` kimenetekből tanul korrekciót az óra HR-jére, kizárólag
óraoldali (inference-időben elérhető) jellemzőkből. A cél nem az, hogy egy
modell szép in-sample számot adjon, hanem hogy eldőljön: mennyi és milyen
kalibráció kell ahhoz, hogy a korrekció egy új szakaszon is javítson.

Validációs sémák:

  A)  prefix   – a session első X percén tanul, a maradékon értékel.
                 Ez a valódi "személyre szabás" kérdés: elég-e X perc EKG?
  B)  blokk-CV – 5 perces blokkok szerinti csoportos k-fold ugyanazon a
                 sessionön. Ez a tanulhatóság felső korlátja, nem üzemi becslés.
  C)  transfer – az egyik napon tanul, a másikon értékel. Ez dönti el, hogy a
                 korrekció személyre (napokon át stabil) vagy csak felhelyezésre
                 (session-specifikus) szabható-e.
  C2) LOSO     – a többi nap egyesített anyagán tanul, a kihagyotton értékel.
                 Három sessiontől kezdve ez a végső döntési szám.
  D)  hibrid   – populációs alak a másik napról + rövid in-session eltolás.
  E)  sávok    – HR-sávonkénti bontás.
  F)  görbe    – k darab véletlen 5 perces blokkon tanít: mennyitől áll be?
  G)  magas HR – csak az a tartomány, ahol a hiba tömege keletkezik.

A szignifikanciát 5 perces blokk-bootstrap adja, mert az 1 Hz-es minták
erősen autokorreláltak (tau ~ 7-64 s), így a nyers n félrevezető.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BLOCK_SECONDS = 300.0
RNG = np.random.default_rng(20260902)


# --------------------------------------------------------------------------- #
# adat
# --------------------------------------------------------------------------- #


@dataclass
class Session:
    name: str
    time: np.ndarray
    watch: np.ndarray
    reference: np.ndarray
    features: np.ndarray = field(default_factory=lambda: np.empty(0))
    feature_names: tuple[str, ...] = ()
    motion: np.ndarray = field(default_factory=lambda: np.empty(0))
    posture: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def target(self) -> np.ndarray:
        """A keresett korrekció: mennyit kell az órához adni."""
        return self.reference - self.watch

    @property
    def blocks(self) -> np.ndarray:
        return ((self.time - self.time[0]) // BLOCK_SECONDS).astype(int)

    def subset(self, mask: np.ndarray) -> "Session":
        return Session(
            self.name,
            self.time[mask],
            self.watch[mask],
            self.reference[mask],
            self.features[mask],
            self.feature_names,
            self.motion[mask] if len(self.motion) else self.motion,
            self.posture[mask] if len(self.posture) else self.posture,
        )


POSTURE_CODE = {"fekvő": 1.0, "álló/ülő nyugalom": 2.0, "álló + mozgás": 3.0}


def load_session(name: str, path: Path) -> Session:
    time, watch, reference, motion, posture = [], [], [], [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["usable"] != "1":
                continue
            time.append(float(row["watch_timestamp_raw"]) / 1000.0)
            watch.append(float(row["watch_hr_bpm"]))
            reference.append(float(row["ecg_reference_hr_bpm"]))
            raw = row.get("mox_motion_count") or ""
            motion.append(float(raw) if raw else math.nan)
            posture.append(POSTURE_CODE.get(row.get("mox_posture") or "", math.nan))
    session = Session(
        name,
        np.asarray(time, dtype=np.float64),
        np.asarray(watch, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
        motion=np.asarray(motion, dtype=np.float64),
        posture=np.asarray(posture, dtype=np.float64),
    )
    session.features, session.feature_names = build_features(session.time, session.watch)
    return session


# --------------------------------------------------------------------------- #
# jellemzők – kizárólag óraoldali, kauzális (csak múltbeli minták)
# --------------------------------------------------------------------------- #


def _causal_ewma(values: np.ndarray, times: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        dt = max(times[i] - times[i - 1], 0.0)
        alpha = 1.0 - math.exp(-dt / tau) if dt > 0 else 0.0
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _causal_window(values: np.ndarray, times: np.ndarray, span: float, fn) -> np.ndarray:
    out = np.empty_like(values)
    start = 0
    for i in range(len(values)):
        while times[i] - times[start] > span:
            start += 1
        out[i] = fn(values[start : i + 1])
    return out


def build_features(times: np.ndarray, watch: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Óraoldali jellemzők. A nyugalmi alapvonal a személyre szabás horgonya."""
    resting = float(np.percentile(watch, 5.0))
    columns: dict[str, np.ndarray] = {"hr": watch}

    for tau in (10.0, 30.0, 120.0):
        smooth = _causal_ewma(watch, times, tau)
        columns[f"hr_ewma{int(tau)}"] = smooth
        columns[f"hr_dev{int(tau)}"] = watch - smooth

    for span in (5.0, 15.0):
        lagged = _causal_window(watch, times, span, lambda w: w[0])
        columns[f"slope{int(span)}"] = (watch - lagged) / span

    for span in (15.0, 60.0):
        columns[f"sd{int(span)}"] = _causal_window(watch, times, span, np.std)

    columns["range30"] = _causal_window(watch, times, 30.0, np.ptp)
    columns["abs_slope_ewma"] = _causal_ewma(
        np.abs(columns["slope5"]), times, 30.0
    )
    # Személyes horgony: a nyugalmi pulzushoz viszonyított terhelés.
    columns["above_rest"] = watch - resting
    columns["rest_ratio"] = watch / max(resting, 1.0)

    names = tuple(columns)
    matrix = np.column_stack([columns[name] for name in names])
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), names


# --------------------------------------------------------------------------- #
# modellek – mind ugyanazt az interfészt adja: fit(session) -> predict(session)
# --------------------------------------------------------------------------- #


class Model:
    name = "model"
    needs_sklearn = False

    def fit(self, train: Session) -> "Model":
        raise NotImplementedError

    def correction(self, session: Session) -> np.ndarray:
        raise NotImplementedError

    def predict(self, session: Session) -> np.ndarray:
        return session.watch + self.correction(session)


class Identity(Model):
    name = "identity"

    def fit(self, train: Session) -> "Identity":
        return self

    def correction(self, session: Session) -> np.ndarray:
        return np.zeros(len(session.watch))


class Bias(Model):
    name = "bias"

    def fit(self, train: Session) -> "Bias":
        self.value = float(np.median(train.target)) if len(train.target) else 0.0
        return self

    def correction(self, session: Session) -> np.ndarray:
        return np.full(len(session.watch), self.value)


class Affine(Model):
    name = "affine"

    def fit(self, train: Session) -> "Affine":
        x, y = train.watch, train.reference
        if len(x) < 20:
            self.slope, self.intercept = 1.0, 0.0
            return self
        design = np.column_stack([x, np.ones_like(x)])
        self.slope, self.intercept = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def correction(self, session: Session) -> np.ndarray:
        return self.slope * session.watch + self.intercept - session.watch


class PiecewiseHR(Model):
    """HR-sávonkénti medián korrekció, sávok között lineárisan interpolálva."""

    name = "hr-piecewise"
    edges = np.array([60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 160.0])

    def fit(self, train: Session) -> "PiecewiseHR":
        centres, values = [], []
        for lo, hi in zip(self.edges[:-1], self.edges[1:]):
            mask = (train.watch >= lo) & (train.watch < hi)
            if mask.sum() >= 60:
                centres.append(0.5 * (lo + hi))
                values.append(float(np.median(train.target[mask])))
        if len(centres) < 2:
            centres = [60.0, 160.0]
            base = float(np.median(train.target)) if len(train.target) else 0.0
            values = [base, base]
        self.centres = np.asarray(centres)
        self.values = np.asarray(values)
        return self

    def correction(self, session: Session) -> np.ndarray:
        return np.interp(session.watch, self.centres, self.values)


def blocked_folds(session: Session, folds: int, seed: int = 0) -> np.ndarray:
    """Fold-hozzárendelés 5 perces blokkok szerint, hogy ne szivárogjon."""
    blocks = session.blocks
    unique = np.unique(blocks)
    order = np.random.default_rng(seed).permutation(unique)
    assignment = {b: i % folds for i, b in enumerate(order)}
    return np.array([assignment[b] for b in blocks])


def tune(session: Session, build, grid, fallback, folds: int = 3) -> object:
    """Hiperparaméter-választás a tanítóhalmazon belüli blokkolt CV-vel.

    Ha nincs elég blokk a hangoláshoz, a `fallback` (a legerősebben
    korlátozott beállítás) jár, nem a rács első eleme – különben pont a
    legrövidebb kalibrációnál kapnánk a legszabadabb modellt.
    """
    if len(np.unique(session.blocks)) < folds + 1:
        return fallback
    fold_of = blocked_folds(session, folds, seed=17)
    best, best_mae = fallback, math.inf
    for setting in grid:
        errors = []
        for fold in range(folds):
            mask = fold_of == fold
            if mask.sum() < 30 or (~mask).sum() < 60:
                continue
            model = build(setting).fit(session.subset(~mask))
            part = session.subset(mask)
            errors.append(np.abs(model.predict(part) - part.reference))
        if not errors:
            continue
        mae = float(np.mean(np.concatenate(errors)))
        if mae < best_mae:
            best, best_mae = setting, mae
    return best


class Ridge(Model):
    name = "ridge"

    def __init__(self, penalty: float | None = None) -> None:
        self.penalty = penalty
        self.auto = penalty is None

    def _design(self, features: np.ndarray) -> np.ndarray:
        standard = (features - self.mean) / np.where(self.live, self.scale, 1.0)
        standard[:, ~self.live] = 0.0
        # A tanítóhalmazon kívüli szélsőértékek ne robbantsák a normálegyenletet.
        np.clip(standard, -20.0, 20.0, out=standard)
        return np.column_stack([standard, np.ones(len(features))])

    def fit(self, train: Session) -> "Ridge":
        if self.auto:
            grid = [1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0]
            self.penalty = float(tune(train, Ridge, grid, fallback=grid[-1]))
        x = np.asarray(train.features, dtype=np.float64)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        # A közel konstans oszlopokat kihagyjuk, különben a standardizálás robban.
        self.live = self.scale > 1e-3 * (np.abs(self.mean) + 1.0)
        z = self._design(x)
        # A macOS Accelerate BLAS látszólagos FPE-jelzőt állít a matmulon; a
        # normálegyenlet maga véges, ezért a jelzőket itt elnyomjuk.
        with np.errstate(all="ignore"):
            gram = z.T @ z + self.penalty * np.eye(z.shape[1])
            gram[-1, -1] -= self.penalty
            self.weights = np.linalg.lstsq(gram, z.T @ train.target, rcond=None)[0]
        return self

    def correction(self, session: Session) -> np.ndarray:
        z = self._design(np.asarray(session.features, dtype=np.float64))
        with np.errstate(all="ignore"):
            return z @ self.weights


class Boosted(Model):
    """Gradiens boosting, blokkolt belső CV-vel hangolt kapacitással.

    A hangolás nélküli változat drasztikusan túlilleszt, ezért a modellt nem
    szalmabábként, hanem a tanítóhalmazon belül tisztességesen megválasztott
    mélységgel és iterációszámmal futtatjuk.
    """

    name = "gbm"
    needs_sklearn = True
    grid = ((2, 60), (2, 200), (3, 100), (3, 300), (4, 200))

    def __init__(self, setting: tuple[int, int] | None = None) -> None:
        self.setting = setting
        self.auto = setting is None

    def fit(self, train: Session) -> "Boosted":
        from sklearn.ensemble import HistGradientBoostingRegressor

        if self.auto:
            self.setting = tune(train, Boosted, list(self.grid), fallback=self.grid[0])
        depth, iterations = self.setting
        self.model = HistGradientBoostingRegressor(
            max_depth=depth,
            max_iter=iterations,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=100,
            random_state=0,
        )
        self.model.fit(train.features, train.target)
        return self

    def correction(self, session: Session) -> np.ndarray:
        return self.model.predict(session.features)


class FIR(Model):
    """Dinamikus modell: az óra a valódi HR simított/késleltetett változata.

    A korrekciót az óra elmúlt L másodpercének lineáris kombinációjából
    becsüljük, azaz a szenzor simítását próbáljuk visszafejteni. Ez más
    hipotézisosztály, mint a pontonkénti HR-függő eltolás.
    """

    name = "fir"

    def __init__(self, span: float = 20.0, penalty: float | None = None) -> None:
        self.span = span
        self.penalty = penalty
        self.auto = penalty is None

    def _lags(self, session: Session) -> np.ndarray:
        watch, times = session.watch, session.time
        columns = [watch]
        for lag in (2.0, 5.0, 8.0, 12.0, 16.0, self.span):
            index = np.searchsorted(times, times - lag, side="left")
            columns.append(watch[np.clip(index, 0, len(watch) - 1)])
        return np.column_stack(columns)

    def _design(self, x: np.ndarray) -> np.ndarray:
        standard = (x - self.mean) / np.where(self.live, self.scale, 1.0)
        standard[:, ~self.live] = 0.0
        np.clip(standard, -20.0, 20.0, out=standard)
        return np.column_stack([standard, np.ones(len(x))])

    def fit(self, train: Session) -> "FIR":
        if self.auto:
            grid = [1.0, 10.0, 100.0, 1_000.0, 10_000.0]
            self.penalty = float(
                tune(train, lambda p: FIR(self.span, p), grid, fallback=grid[-1])
            )
        x = self._lags(train)
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.live = self.scale > 1e-3 * (np.abs(self.mean) + 1.0)
        z = self._design(x)
        with np.errstate(all="ignore"):
            gram = z.T @ z + self.penalty * np.eye(z.shape[1])
            gram[-1, -1] -= self.penalty
            self.weights = np.linalg.lstsq(gram, z.T @ train.target, rcond=None)[0]
        return self

    def correction(self, session: Session) -> np.ndarray:
        with np.errstate(all="ignore"):
            return self._design(self._lags(session)) @ self.weights


class Shrunk(Model):
    """Populációs modell + session-szintű maradék bias.

    Ez a klasszikus személyre szabási szerkezet: a másik napról hozott alak
    plusz egy rövid in-session kalibrációból becsült eltolás, James-Stein
    jellegű zsugorítással, hogy kevés kalibrációs perc ne rántsa el.
    """

    name = "population+offset"

    def __init__(self, population: Model, prior_strength: float = 600.0) -> None:
        self.population = population
        self.prior_strength = prior_strength

    def fit(self, train: Session) -> "Shrunk":
        residual = train.target - self.population.correction(train)
        n = len(residual)
        raw = float(np.median(residual)) if n else 0.0
        self.offset = raw * n / (n + self.prior_strength)
        return self

    def correction(self, session: Session) -> np.ndarray:
        return self.population.correction(session) + self.offset


# --------------------------------------------------------------------------- #
# metrikák
# --------------------------------------------------------------------------- #


def score(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    return {
        "n": float(len(error)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "p5": float(np.mean(np.abs(error) <= 5.0) * 100.0),
    }


def block_bootstrap_gain(
    baseline: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    blocks: np.ndarray,
    draws: int = 2000,
) -> tuple[float, float, float]:
    """MAE-nyereség (baseline - candidate) 5 perces blokk-bootstrap CI-vel."""
    unique = np.unique(blocks)
    if len(unique) < 3:
        return float("nan"), float("nan"), float("nan")
    per_block = {
        b: (
            np.abs(baseline[blocks == b] - reference[blocks == b]),
            np.abs(candidate[blocks == b] - reference[blocks == b]),
        )
        for b in unique
    }
    point = float(
        np.mean(np.abs(baseline - reference)) - np.mean(np.abs(candidate - reference))
    )
    gains = np.empty(draws)
    for draw in range(draws):
        picked = RNG.choice(unique, size=len(unique), replace=True)
        base = np.concatenate([per_block[b][0] for b in picked])
        cand = np.concatenate([per_block[b][1] for b in picked])
        gains[draw] = base.mean() - cand.mean()
    return point, float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def band_table(
    prediction: np.ndarray, reference: np.ndarray, watch: np.ndarray
) -> dict[str, dict[str, float]]:
    bands = [(0, 90), (90, 110), (110, 130), (130, 250)]
    table = {}
    for lo, hi in bands:
        mask = (watch >= lo) & (watch < hi)
        if mask.sum() >= 30:
            table[f"{lo}-{hi}"] = score(prediction[mask], reference[mask])
    return table


# --------------------------------------------------------------------------- #
# validációs sémák
# --------------------------------------------------------------------------- #


def make_models(has_sklearn: bool) -> list[Model]:
    models: list[Model] = [Identity(), Bias(), Affine(), PiecewiseHR(), FIR(), Ridge()]
    if has_sklearn:
        models.append(Boosted())
    return models


def prefix_scheme(session: Session, minutes: list[float], has_sklearn: bool) -> None:
    print(f"\n### A) Prefix-kalibráció – {session.name}")
    print("Az első X percen tanul, a maradékon értékel (kronologikus, nincs szivárgás).")
    header = f"{'X perc':>7} {'teszt n':>8} " + " ".join(
        f"{m.name:>14}" for m in make_models(has_sklearn)
    )
    print(header)
    for span in minutes:
        cut = session.time[0] + span * 60.0
        train = session.subset(session.time < cut)
        test = session.subset(session.time >= cut)
        if len(train.time) < 120 or len(test.time) < 600:
            continue
        row = [f"{span:>7.0f} {len(test.time):>8d} "]
        for model in make_models(has_sklearn):
            model.fit(train)
            row.append(f"{score(model.predict(test), test.reference)['MAE']:>14.3f}")
        print("".join(row))
    print("(értékek: teszt MAE bpm)")


def blocked_cv(session: Session, has_sklearn: bool, folds: int = 5) -> None:
    print(f"\n### B) Blokkolt keresztvalidáció – {session.name}")
    print("5 perces blokkok szerinti csoportos k-fold: a tanulhatóság felső korlátja.")
    blocks = session.blocks
    fold_of = blocked_folds(session, folds, seed=3)
    print(f"{'modell':<16}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'±5bpm%':>9}{'nyereség (95% CI)':>28}")
    baseline = session.watch.copy()
    for model in make_models(has_sklearn):
        prediction = np.empty(len(session.watch))
        for fold in range(folds):
            test_mask = fold_of == fold
            if test_mask.sum() == 0 or (~test_mask).sum() < 200:
                prediction[test_mask] = session.watch[test_mask]
                continue
            model.fit(session.subset(~test_mask))
            prediction[test_mask] = model.predict(session.subset(test_mask))
        row = score(prediction, session.reference)
        gain, lo, hi = block_bootstrap_gain(
            baseline, prediction, session.reference, blocks
        )
        ci = "—" if math.isnan(gain) else f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(
            f"{model.name:<16}{row['MAE']:>8.3f}{row['RMSE']:>8.3f}"
            f"{row['bias']:>8.3f}{row['p5']:>9.1f}{ci:>28}"
        )


def transfer_scheme(sessions: list[Session], has_sklearn: bool) -> None:
    print("\n### C) Napok közötti transzfer")
    print("Egy napon tanul, a másikon értékel. Ez dönti el, hogy a korrekció")
    print("személyre szabható-e, vagy csak az adott felhelyezésre érvényes.")
    for train in sessions:
        for test in sessions:
            if train.name == test.name:
                continue
            print(f"\n  tanít: {train.name} → tesztel: {test.name}  (teszt n={len(test.time)})")
            print(
                f"  {'modell':<18}{'MAE':>8}{'RMSE':>8}{'bias':>8}{'±5bpm%':>9}"
                f"{'nyereség (95% CI)':>28}"
            )
            for model in make_models(has_sklearn):
                model.fit(train)
                prediction = model.predict(test)
                row = score(prediction, test.reference)
                gain, lo, hi = block_bootstrap_gain(
                    test.watch, prediction, test.reference, test.blocks
                )
                ci = "—" if math.isnan(gain) else f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                print(
                    f"  {model.name:<18}{row['MAE']:>8.3f}{row['RMSE']:>8.3f}"
                    f"{row['bias']:>8.3f}{row['p5']:>9.1f}{ci:>28}"
                )


def pool(sessions: list[Session], name: str) -> Session:
    """Több session egyesítése. A jellemzők sessiononként készültek, ezért a
    nyugalmi horgony és a gördülő ablakok nem lógnak át napok között."""
    order = np.argsort(np.concatenate([s.time for s in sessions]))
    merged = Session(
        name,
        np.concatenate([s.time for s in sessions])[order],
        np.concatenate([s.watch for s in sessions])[order],
        np.concatenate([s.reference for s in sessions])[order],
        np.vstack([s.features for s in sessions])[order],
        sessions[0].feature_names,
    )
    return merged


def leave_one_session_out(sessions: list[Session], has_sklearn: bool) -> None:
    """Ez a séma a valódi cél: a többi napon tanul, egy kihagyotton értékel.

    Két sessionnél ez egybeesik a C) páronkénti transzferrel; háromtól kezdve
    ad új információt, és ez lesz az a szám, ami alapján egy korrekció
    általánosnak nyilvánítható.
    """
    if len(sessions) < 3:
        print(
            f"\n### C2) Leave-one-session-out: {len(sessions)} session, "
            "egybeesik a C) sémával – legalább 3 kell hozzá."
        )
        return
    print("\n### C2) Leave-one-session-out")
    print("A többi nap egyesített anyagán tanul, a kihagyott napon értékel.")
    for index, held in enumerate(sessions):
        others = [s for i, s in enumerate(sessions) if i != index]
        train = pool(others, "+".join(s.name for s in others))
        print(f"\n  kihagyva: {held.name} (n={len(held.time)}), tanít: {train.name}")
        print(f"  {'modell':<18}{'MAE':>8}{'bias':>8}{'±5bpm%':>9}{'nyereség (95% CI)':>28}")
        for model in make_models(has_sklearn):
            model.fit(train)
            prediction = model.predict(held)
            row = score(prediction, held.reference)
            gain, lo, hi = block_bootstrap_gain(
                held.watch, prediction, held.reference, held.blocks
            )
            ci = "—" if math.isnan(gain) else f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            print(
                f"  {model.name:<18}{row['MAE']:>8.3f}{row['bias']:>8.3f}"
                f"{row['p5']:>9.1f}{ci:>28}"
            )


def hybrid_scheme(sessions: list[Session], has_sklearn: bool, minutes: float = 20.0) -> None:
    """Populációs alak a másik napról + rövid in-session eltolás."""
    print(f"\n### D) Populációs modell + {minutes:.0f} perc in-session kalibráció")
    print("A másik napon tanult alakhoz zsugorított session-eltolás adódik.")
    for train in sessions:
        for test in sessions:
            if train.name == test.name:
                continue
            cut = test.time[0] + minutes * 60.0
            calib = test.subset(test.time < cut)
            evaluate = test.subset(test.time >= cut)
            if len(calib.time) < 120 or len(evaluate.time) < 600:
                continue
            print(f"\n  alak: {train.name} → kalibráció+teszt: {test.name} (teszt n={len(evaluate.time)})")
            print(f"  {'variáns':<28}{'MAE':>8}{'nyereség (95% CI)':>28}")
            for population in make_models(has_sklearn):
                if population.name == "identity":
                    continue
                population.fit(train)
                plain = population.predict(evaluate)
                shrunk = Shrunk(population).fit(calib)
                adapted = shrunk.predict(evaluate)
                for label, prediction in (
                    (f"{population.name} (csak transzfer)", plain),
                    (f"{population.name} + eltolás", adapted),
                ):
                    gain, lo, hi = block_bootstrap_gain(
                        evaluate.watch, prediction, evaluate.reference, evaluate.blocks
                    )
                    ci = "—" if math.isnan(gain) else f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                    mae = score(prediction, evaluate.reference)["MAE"]
                    print(f"  {label:<28}{mae:>8.3f}{ci:>28}")
            base_only = Bias().fit(calib)
            prediction = base_only.predict(evaluate)
            gain, lo, hi = block_bootstrap_gain(
                evaluate.watch, prediction, evaluate.reference, evaluate.blocks
            )
            ci = "—" if math.isnan(gain) else f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            mae = score(prediction, evaluate.reference)["MAE"]
            print(f"  {'csak in-session bias':<28}{mae:>8.3f}{ci:>28}")


def learning_curve(
    session: Session, has_sklearn: bool, repeats: int = 25
) -> None:
    """Mennyi kalibrációs anyagtól áll be a nyereség?

    k darab véletlen 5 perces blokkon tanít, a maradékon értékel. A blokkokat
    véletlenszerűen húzza, tehát a kalibráció reprezentatív – ez a prefixnél
    kedvezőbb, felső korlát jellegű eset.
    """
    print(f"\n### F) Tanulási görbe – {session.name}")
    print("k véletlen 5 perces blokkon tanul, a maradékon értékel; 25 ismétlés átlaga.")
    unique = np.unique(session.blocks)
    models = [m for m in make_models(has_sklearn) if m.name != "identity"]
    print(f"{'k blokk':>8}{'perc':>6} " + " ".join(f"{m.name:>14}" for m in models))
    for k in (1, 2, 4, 8, 12, 16):
        if k >= len(unique) - 2:
            break
        gains = {m.name: [] for m in models}
        for repeat in range(repeats):
            rng = np.random.default_rng(1000 + repeat)
            picked = set(rng.choice(unique, size=k, replace=False).tolist())
            mask = np.array([b in picked for b in session.blocks])
            if mask.sum() < 60 or (~mask).sum() < 300:
                continue
            train, test = session.subset(mask), session.subset(~mask)
            base = np.mean(np.abs(test.watch - test.reference))
            for model in models:
                fresh = type(model)()
                fresh.fit(train)
                gains[model.name].append(
                    base - np.mean(np.abs(fresh.predict(test) - test.reference))
                )
        row = f"{k:>8}{k * 5:>6} "
        for model in models:
            values = gains[model.name]
            row += f"{np.mean(values) if values else float('nan'):>+14.3f}"
        print(row)
    print("(értékek: átlagos MAE-nyereség az identityhez képest, bpm; pozitív = javít)")


def high_band(session: Session, has_sklearn: bool, floor: float = 110.0) -> None:
    """Csak a magas-HR tartomány, ahol a hiba tömege keletkezik."""
    mask = session.watch >= floor
    if mask.sum() < 400:
        print(f"\n### G) Magas-HR tartomány – {session.name}: kevés minta ({mask.sum()}), kihagyva")
        return
    part = session.subset(mask)
    print(f"\n### G) Magas-HR tartomány (óra ≥ {floor:.0f} bpm) – {session.name}")
    print(f"n={len(part.time)}, nyers MAE={np.mean(np.abs(part.target)):.3f}")
    print("A tanítás a TELJES sessionön megy, az értékelés csak ezen a sávon.")
    fold_of = blocked_folds(session, 5, seed=5)
    print(f"{'modell':<16}{'MAE':>8}{'bias':>8}{'±5bpm%':>9}{'nyereség (95% CI)':>28}")
    for model in make_models(has_sklearn):
        prediction = np.empty(len(session.watch))
        for fold in range(5):
            test_mask = fold_of == fold
            model.fit(session.subset(~test_mask))
            prediction[test_mask] = model.predict(session.subset(test_mask))
        row = score(prediction[mask], part.reference)
        gain, lo, hi = block_bootstrap_gain(
            part.watch, prediction[mask], part.reference, part.blocks
        )
        ci = "—" if math.isnan(gain) else f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(f"{model.name:<16}{row['MAE']:>8.3f}{row['bias']:>8.3f}{row['p5']:>9.1f}{ci:>28}")


def band_breakdown(sessions: list[Session], has_sklearn: bool) -> None:
    print("\n### E) HR-sávonkénti bontás (blokkolt CV predikcióval)")
    for session in sessions:
        blocks = session.blocks
        unique = np.unique(blocks)
        assignment = {b: i % 5 for i, b in enumerate(RNG.permutation(unique))}
        fold_of = np.array([assignment[b] for b in blocks])
        print(f"\n  {session.name}")
        print(f"  {'sáv':<10}{'n':>7}{'nyers MAE':>11}", end="")
        chosen = [m for m in make_models(has_sklearn) if m.name in {"bias", "hr-piecewise", "gbm"}]
        for model in chosen:
            print(f"{model.name:>14}", end="")
        print()
        predictions = {}
        for model in chosen:
            prediction = np.empty(len(session.watch))
            for fold in range(5):
                mask = fold_of == fold
                model.fit(session.subset(~mask))
                prediction[mask] = model.predict(session.subset(mask))
            predictions[model.name] = prediction
        for band, row in band_table(session.watch, session.reference, session.watch).items():
            print(f"  {band:<10}{row['n']:>7.0f}{row['MAE']:>11.3f}", end="")
            lo, hi = (float(v) for v in band.split("-"))
            mask = (session.watch >= lo) & (session.watch < hi)
            for model in chosen:
                value = score(predictions[model.name][mask], session.reference[mask])["MAE"]
                print(f"{value:>14.3f}", end="")
            print()


# --------------------------------------------------------------------------- #


def context_scheme(sessions: list[Session], has_sklearn: bool) -> None:
    """H) Mennyit érne a MOX testhelyzet- és mozgásinformáció?

    A `personalize.py` modelljei szándékosan csak óraoldali jellemzőket használnak,
    hogy a korrekció Holter nélkül is futtatható legyen. Ez a séma nem bővíti a
    modelleket: azt méri meg, hogy a MOX-kontextus *felső korlátként* mennyit adna
    hozzá. Ha a nyereség nulla, a testhelyzet-információ hiánya nem korlát.
    """
    print("\n### H) MOX testhelyzet és mozgás mint kontextus")
    print("A modellek óraoldaliak maradnak; itt csak a kontextus felső korlátját mérjük.")
    label = {1.0: "fekvő", 2.0: "álló/ülő nyugalom", 3.0: "álló + mozgás"}

    for session in sessions:
        if not len(session.posture) or not np.any(np.isfinite(session.posture)):
            print(f"\n{session.name}: nincs MOX mozgásadat a comparison.csv-ben.")
            continue
        print(f"\n{session.name} – nyers hiba (óra − EKG) testhelyzet szerint")
        print(f"{'testhelyzet':<20}{'n':>7}{'MAE':>8}{'bias':>8}{'EKG HR med':>12}{'mozgás med':>12}")
        for code in (1.0, 2.0, 3.0):
            mask = session.posture == code
            if mask.sum() < 100:
                continue
            error = session.watch[mask] - session.reference[mask]
            print(
                f"{label[code]:<20}{mask.sum():>7d}{np.mean(np.abs(error)):>8.3f}"
                f"{np.mean(error):>8.3f}{np.median(session.reference[mask]):>12.1f}"
                f"{np.nanmedian(session.motion[mask]):>12.1f}"
            )
        finite = np.isfinite(session.motion)
        if np.any(finite):
            print(f"\n{session.name} – nyers hiba mozgás-count sávonként")
            print(f"{'count sáv':<20}{'n':>7}{'MAE':>8}{'bias':>8}{'EKG HR med':>12}")
            edges = [0.0, 15.0, 30.0, 60.0, 120.0, np.inf]
            for low, high in zip(edges[:-1], edges[1:]):
                mask = finite & (session.motion >= low) & (session.motion < high)
                if mask.sum() < 100:
                    continue
                error = session.watch[mask] - session.reference[mask]
                name = f"{low:.0f}–{high:.0f}" if np.isfinite(high) else f"{low:.0f}+"
                print(
                    f"{name:<20}{mask.sum():>7d}{np.mean(np.abs(error)):>8.3f}"
                    f"{np.mean(error):>8.3f}{np.median(session.reference[mask]):>12.1f}"
                )

    print("\nBlokkolt CV: óraoldali jellemzők vs. ugyanazok + MOX mozgás és testhelyzet")
    print(f"{'session':<10}{'modell':<12}{'óra-only':>10}{'+ MOX':>10}{'nyereség (95% CI)':>28}")
    for session in sessions:
        if not len(session.posture) or not np.any(np.isfinite(session.posture)):
            continue
        motion = np.nan_to_num(session.motion, nan=0.0)
        posture = np.nan_to_num(session.posture, nan=2.0)
        extra = np.column_stack(
            [
                motion,
                np.log1p(motion),
                (posture == 1.0).astype(float),
                (posture == 3.0).astype(float),
            ]
        )
        extended = Session(
            session.name, session.time, session.watch, session.reference,
            np.column_stack([session.features, extra]),
            session.feature_names + ("mox_count", "mox_log_count", "mox_fekvő", "mox_mozgás"),
            session.motion, session.posture,
        )
        blocks = session.blocks
        fold_of = blocked_folds(session, 5, seed=3)
        for model_factory in ([Ridge, Boosted] if has_sklearn else [Ridge]):
            predictions = {}
            for tag, data in (("óra", session), ("mox", extended)):
                prediction = np.empty(len(data.watch))
                for fold in range(5):
                    test = fold_of == fold
                    if test.sum() == 0 or (~test).sum() < 200:
                        prediction[test] = data.watch[test]
                        continue
                    model = model_factory()
                    model.fit(data.subset(~test))
                    prediction[test] = model.predict(data.subset(test))
                predictions[tag] = prediction
            gain, lo, hi = block_bootstrap_gain(
                predictions["óra"], predictions["mox"], session.reference, blocks
            )
            ci = "—" if math.isnan(gain) else f"{gain:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            print(
                f"{session.name:<10}{model_factory.name:<12}"
                f"{score(predictions['óra'], session.reference)['MAE']:>10.3f}"
                f"{score(predictions['mox'], session.reference)['MAE']:>10.3f}{ci:>28}"
            )
    print("\n(A nyereség pozitív értéke azt jelentené, hogy a testhelyzet/mozgás olyan")
    print("információt hordoz, amit az óra saját jelzései nem. Ha a CI tartalmazza a")
    print("nullát, a MOX-kontextus nem ad hozzá mérhetőt.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        metavar="NÉV=CSV",
        help="Session neve és comparison.csv útvonala; többször megadható.",
    )
    args = parser.parse_args()
    specs = args.session or [
        "08_28=08_28/results_v2/comparison.csv",
        "09_01=09_01/results/comparison.csv",
    ]

    try:
        import sklearn  # noqa: F401

        has_sklearn = True
    except ImportError:
        has_sklearn = False
        print("Figyelem: sklearn nincs telepítve, a gbm modell kimarad.\n")

    sessions = []
    for spec in specs:
        name, _, path = spec.partition("=")
        sessions.append(load_session(name, Path(path)))

    for session in sessions:
        print(
            f"{session.name}: n={len(session.time)}, "
            f"óra HR {session.watch.min():.0f}-{session.watch.max():.0f}, "
            f"EKG {session.reference.min():.0f}-{session.reference.max():.0f}, "
            f"nyers MAE={np.mean(np.abs(session.target)):.2f}, "
            f"blokkok={len(np.unique(session.blocks))}"
        )

    for session in sessions:
        prefix_scheme(session, [5, 10, 20, 30, 45, 60], has_sklearn)
    for session in sessions:
        blocked_cv(session, has_sklearn)
    transfer_scheme(sessions, has_sklearn)
    leave_one_session_out(sessions, has_sklearn)
    hybrid_scheme(sessions, has_sklearn)
    for session in sessions:
        learning_curve(session, has_sklearn)
    for session in sessions:
        high_band(session, has_sklearn)
    band_breakdown(sessions, has_sklearn)
    context_scheme(sessions, has_sklearn)


if __name__ == "__main__":
    main()
