"""Descriptive checks of season roster values; no fitting, imputation or feature output."""

from collections.abc import Mapping

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from bs4 import BeautifulSoup

from ligat.data.squad_values import money


def parse_probe_table(content: bytes, *, date: str | None = None) -> pd.DataFrame:
    """Read the first items table: season total or the specified dated value column.

    Club names, rather than the page's current-season club links, identify rows.
    No date is inferred from an undated season roster page.
    """
    table = BeautifulSoup(content, "lxml", from_encoding="utf-8").select_one("table.items")
    if table is None:
        raise ValueError("Missing first items table")
    headers = [cell.get_text(" ", strip=True) for cell in table.select("thead th")]
    value_header = "Total market value" if date is None else f"Value {date}"
    if value_header not in headers:
        raise ValueError(f"Missing value column: {value_header}")
    rows = []
    for row in table.select("tbody > tr"):
        links = row.select("a[href*='/startseite/verein/']")
        name = next(
            (link.get_text(" ", strip=True) for link in links if link.get_text(strip=True)), ""
        )
        if not name:
            continue
        cells = row.find_all("td", recursive=False)
        if len(cells) != len(headers):
            raise ValueError(f"Unexpected probe row width: {name}")
        value, _ = money(cells[headers.index(value_header)].get_text(" ", strip=True))
        rows.append({"club_name": name, "value_eur": value})
    result = pd.DataFrame(rows, columns=["club_name", "value_eur"])
    if result.empty or result.club_name.duplicated().any():
        raise ValueError("Probe club names must be nonempty and unique")
    result["value_eur"] = result.value_eur.astype("Float64")
    return result


def _rho(frame: pd.DataFrame, x: str, y: str) -> float:
    pair = frame[[x, y]].dropna().astype(float)
    if len(pair) < 2 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return float("nan")
    return float(pair[x].corr(pair[y], method="spearman"))


def probe_comparison(season: pd.DataFrame, dated: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare all pages on one common club intersection; shares are renormalized."""
    if not dated:
        raise ValueError("At least one dated probe is required")
    pages = {"season": season, **dated}
    common = set(season.club_name)
    for frame in pages.values():
        if frame.club_name.duplicated().any():
            raise ValueError("Duplicate probe club name")
        common &= set(frame.club_name)
    if not common:
        raise ValueError("No clubs common to all probe pages")
    aligned = pd.DataFrame(
        {
            name: frame.set_index("club_name").value_eur.reindex(sorted(common))
            for name, frame in pages.items()
        }
    )
    shares = aligned.div(aligned.sum(min_count=1).where(lambda sums: sums.gt(0)), axis=1)
    return pd.DataFrame(
        [
            {
                "date": date,
                "clubs": len(common),
                "paired_values": int(aligned[["season", date]].notna().all(axis=1).sum()),
                "spearman": _rho(aligned, "season", date),
                "share_mae_pp": float((shares.season - shares[date]).abs().mean() * 100),
                "median_ratio": float(
                    (aligned.season / aligned[date].where(aligned[date].gt(0))).median()
                ),
            }
            for date in dated
        ]
    )


def _development(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy() if "season_id" in frame.columns else frame.reset_index()
    if result.season_id.isna().any() or not set(result.season_id) <= set(range(8, 26)):
        raise ValueError("Value methodology B–E requires development seasons 8–25 only")
    return result


def value_shares(values: pd.DataFrame) -> pd.DataFrame:
    """Compute within-season shares of known totals and averages, retaining missing rows."""
    result = _development(values)
    if result.duplicated(["season_id", "team_id"]).any():
        raise ValueError("Expected unique team-season values")
    for source, target in (("total_value_eur", "share_total"), ("avg_value_eur", "share_avg")):
        result[source] = result[source].astype("Float64")
        denominator = result.groupby("season_id")[source].transform("sum", min_count=1)
        result[target] = result[source] / denominator.where(denominator.gt(0))
    return result


def match_ppm(matches: pd.DataFrame) -> pd.DataFrame:
    """Points per match directly from scores, 3/1/0; no official points or halving."""
    games = _development(matches)
    if games[["home_goals", "away_goals"]].isna().any().any():
        raise ValueError("Completed match scores are required for ppm")
    sides = []
    for side, opponent in (("home", "away"), ("away", "home")):
        difference = games[f"{side}_goals"] - games[f"{opponent}_goals"]
        sides.append(
            pd.DataFrame(
                {
                    "season_id": games.season_id,
                    "team_id": games[f"{side}_team_id"],
                    "points": np.where(difference.gt(0), 3, np.where(difference.eq(0), 1, 0)),
                }
            )
        )
    return (
        pd.concat(sides)
        .groupby(["season_id", "team_id"], as_index=False)
        .agg(points=("points", "sum"), matches=("points", "size"), ppm=("points", "mean"))
    )


def _season_pair(
    source: pd.DataFrame, target: pd.DataFrame, season: int, next_season: int, x: str, y: str
) -> pd.DataFrame:
    left = source.loc[source.season_id.eq(season), ["team_id", x]].rename(columns={x: "x"})
    right = target.loc[target.season_id.eq(next_season), ["team_id", y]].rename(columns={y: "y"})
    return left.merge(right, on="team_id", validate="one_to_one")


def direction_correlations(values: pd.DataFrame, ppm: pd.DataFrame) -> pd.DataFrame:
    """Within-season value/ppm associations at offsets 0,+1,-1, plus ppm persistence."""
    shares, performance = value_shares(values), _development(ppm)
    rows = []
    for season in sorted(shares.season_id.unique()):
        for offset in (0, 1, -1):
            target = int(season + offset)
            if target not in set(performance.season_id):
                continue
            pair = _season_pair(shares, performance, int(season), target, "share_total", "ppm")
            rows.append(
                {
                    "season_id": int(season),
                    "target_season_id": target,
                    "comparison": f"share_total_s vs ppm_s{offset:+d}",
                    "teams": len(pair.dropna(subset=["x", "y"])),
                    "spearman": _rho(pair, "x", "y"),
                }
            )
    for season in sorted(performance.season_id.unique()):
        if season + 1 not in set(performance.season_id):
            continue
        pair = _season_pair(performance, performance, int(season), int(season + 1), "ppm", "ppm")
        rows.append(
            {
                "season_id": int(season),
                "target_season_id": int(season + 1),
                "comparison": "ppm_s vs ppm_s+1",
                "teams": len(pair.dropna(subset=["x", "y"])),
                "spearman": _rho(pair, "x", "y"),
            }
        )
    return pd.DataFrame(rows)


def correlation_summary(correlations: pd.DataFrame) -> pd.DataFrame:
    """Equal-season mean and range; undefined season correlations are not zeros."""
    frame = _development(correlations)
    return frame.groupby("comparison", as_index=False).agg(
        mean_spearman=("spearman", "mean"),
        min_spearman=("spearman", "min"),
        max_spearman=("spearman", "max"),
        seasons=("spearman", "count"),
    )


def squad_correlations(values: pd.DataFrame) -> pd.DataFrame:
    """Within-season squad size and sum/average rank associations, with median squad size."""
    rows = []
    for season, frame in value_shares(values).groupby("season_id"):
        rows.append(
            {
                "season_id": season,
                "teams": len(frame),
                "median_squad_size": float(frame.squad_size.median()),
                "squad_vs_total": _rho(frame, "squad_size", "share_total"),
                "squad_vs_avg": _rho(frame, "squad_size", "share_avg"),
                "total_vs_avg": _rho(frame, "share_total", "share_avg"),
            }
        )
    return pd.DataFrame(rows)


def paired_value_correlations(values: pd.DataFrame, ppm: pd.DataFrame) -> pd.DataFrame:
    """Paired total-minus-average rho against next-season ppm, on identical teams."""
    shares, performance = value_shares(values), _development(ppm)
    rows = []
    for season in sorted(shares.season_id.unique()):
        if season + 1 not in set(performance.season_id):
            continue
        source = shares.loc[shares.season_id.eq(season), ["team_id", "share_total", "share_avg"]]
        target = performance.loc[performance.season_id.eq(season + 1), ["team_id", "ppm"]]
        pair = source.merge(target, on="team_id", validate="one_to_one").dropna()
        total, average = _rho(pair, "share_total", "ppm"), _rho(pair, "share_avg", "ppm")
        rows.append(
            {
                "season_id": season,
                "target_season_id": season + 1,
                "teams": len(pair),
                "rho_total": total,
                "rho_avg": average,
                "difference": total - average,
            }
        )
    return pd.DataFrame(rows)


def paired_difference_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Sample dispersion and standard error across paired season differences."""
    differences = _development(paired).difference.dropna().astype(float)
    count = len(differences)
    std = float(differences.std(ddof=1))
    return pd.DataFrame(
        [
            {
                "mean": differences.mean(),
                "std": std,
                "se": std / np.sqrt(count) if count else np.nan,
                "seasons": count,
            }
        ]
    )


def rank_disagreements(values: pd.DataFrame, *, limit: int = 5) -> pd.DataFrame:
    """Largest absolute sum-vs-average rank gaps; tied values receive average ranks."""
    result = value_shares(values)
    for share, rank in (("share_total", "rank_total"), ("share_avg", "rank_avg")):
        result[rank] = result.groupby("season_id")[share].rank(ascending=False, method="average")
    result["rank_gap"] = (result.rank_total - result.rank_avg).abs()
    return (
        result.dropna(subset=["rank_gap"])
        .sort_values(["rank_gap", "season_id", "team_id"], ascending=[False, True, True])
        .head(limit)
        .reset_index(drop=True)
    )


def missing_values(values: pd.DataFrame, ppm: pd.DataFrame) -> pd.DataFrame:
    """Missing source fields with names and whether the club played the next season."""
    shares, performance = value_shares(values), _development(ppm)
    fields = ["squad_size", "total_value_eur", "avg_value_eur"]
    result = shares.loc[shares[fields].isna().any(axis=1)].copy()
    result["missing_fields"] = result[fields].apply(
        lambda row: ", ".join(row.index[row.isna()]), axis=1
    )
    membership = set(zip(performance.season_id, performance.team_id, strict=True))
    seasons = set(performance.season_id)
    result["played_next_season"] = pd.array(
        [
            (int(row.season_id + 1), row.team_id) in membership
            if row.season_id + 1 in seasons
            else pd.NA
            for row in result.itertuples()
        ],
        dtype="boolean",
    )
    return result.reset_index(drop=True)


def value_coverage(values: pd.DataFrame, ppm: pd.DataFrame) -> pd.DataFrame:
    """Prior-value coverage; promotion comes only from consecutive match rosters.

    At the first available season, previous-season membership is unknown, so
    promotion and other-missing counts remain NA rather than labelling all clubs promoted.
    """
    shares, performance = value_shares(values), _development(ppm)
    membership = set(zip(performance.season_id, performance.team_id, strict=True))
    seasons = set(performance.season_id)
    previous = shares[["season_id", "team_id", "total_value_eur"]].copy()
    previous.season_id += 1
    current = performance[["season_id", "team_id"]].merge(
        previous, on=["season_id", "team_id"], how="left", validate="one_to_one"
    )
    rows = []
    for season, frame in current.groupby("season_id"):
        has_history = season - 1 in seasons
        previous_teams = {
            team for previous_season, team in membership if previous_season == season - 1
        }
        promoted = ~frame.team_id.isin(previous_teams)
        known = frame.total_value_eur.notna()
        rows.append(
            {
                "season_id": season,
                "teams": len(frame),
                "with_previous_value": int(known.sum()),
                "missing_previous_value": int((~known).sum()),
                "previous_roster_available": has_history,
                "promoted": int(promoted.sum()) if has_history else pd.NA,
                "missing_other": int((~known & ~promoted).sum()) if has_history else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def share_stability(values: pd.DataFrame) -> pd.DataFrame:
    """Consecutive-season within-league share rho on continuing clubs only."""
    shares = value_shares(values)
    rows = []
    for season in sorted(shares.season_id.unique()):
        if season + 1 not in set(shares.season_id):
            continue
        pair = _season_pair(
            shares, shares, int(season), int(season + 1), "share_total", "share_total"
        )
        rows.append(
            {
                "season_id": season,
                "target_season_id": season + 1,
                "source_teams": int(shares.season_id.eq(season).sum()),
                "target_teams": int(shares.season_id.eq(season + 1).sum()),
                "teams": len(pair.dropna(subset=["x", "y"])),
                "comparison": "share_total_s vs share_total_s+1",
                "spearman": _rho(pair, "x", "y"),
            }
        )
    return pd.DataFrame(rows)
