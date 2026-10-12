"""CLI entry points; importing this module never starts a download."""

import json
import logging
from pathlib import Path
from typing import Annotated

import requests
import typer

from ligat.data.build import build_datasets
from ligat.data.export import export_excel
from ligat.data.repository import Repository
from ligat.scrape.games import run_games
from ligat.scrape.ifa import AccessBlockedError, IFASource, load_scrape_config

app = typer.Typer(no_args_is_help=True)
scrape_app = typer.Typer(no_args_is_help=True)
app.add_typer(scrape_app, name="scrape")
export_app = typer.Typer(no_args_is_help=True)
app.add_typer(export_app, name="export")
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


def parse_seasons(value: str, *, minimum: int = 8, maximum: int = 27) -> tuple[int, ...]:
    """Accept a single season ID or an inclusive ascending range."""
    parts = value.split("-")
    try:
        if len(parts) == 1:
            start = end = int(parts[0])
        elif len(parts) == 2:
            start, end = map(int, parts)
        else:
            raise ValueError("Invalid range")
        if minimum <= start <= end <= maximum:
            return tuple(range(start, end + 1))
    except ValueError:
        pass
    raise typer.BadParameter("Use one season ID or an ascending range, e.g. 8-27")


def _run(config_path: Path, seasons: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    source = None
    try:
        config = load_scrape_config(config_path)
        requested = (
            parse_seasons(
                seasons, minimum=config.first_season_id, maximum=config.live_season_id - 1
            )
            if seasons is not None
            else None
        )
        source = IFASource(config)
        if seasons is None:
            source.scrape_live()
        else:
            assert requested is not None
            source.scrape_history(requested)
    except (AccessBlockedError, requests.RequestException, ValueError, OSError) as exc:
        logging.getLogger(__name__).error("Collection stopped: %s", exc)
        raise typer.Exit(code=1) from exc
    finally:
        if source is not None:
            source.close()


@scrape_app.command("history")
def history(
    seasons: Annotated[str, typer.Option("--seasons")] = "8-27",
    config: ConfigOption = Path("config/scrape.yaml"),
) -> None:
    """Collect completed seasons once; requires approval before a live network run."""
    _run(config, seasons)


@scrape_app.command("live")
def live(config: ConfigOption = Path("config/scrape.yaml")) -> None:
    """Create a timestamped snapshot of all published current-season rounds."""
    _run(config, None)


@app.command("build")
def build(
    raw_root: Annotated[Path, typer.Option("--raw-root")] = Path("data/raw/ifa/rounds"),
    games_root: Annotated[Path, typer.Option("--games-root")] = Path("data/raw/ifa/games"),
    clean_root: Annotated[Path, typer.Option("--clean-root")] = Path("data/clean"),
    seasons_config: Annotated[
        Path, typer.Option("--seasons-config", exists=True, dir_okay=False)
    ] = Path("config/seasons.yaml"),
    overrides_config: Annotated[
        Path, typer.Option("--overrides-config", exists=True, dir_okay=False)
    ] = Path("config/match_overrides.yaml"),
    teams_config: Annotated[Path, typer.Option("--teams-config")] = Path("config/teams.yaml"),
    exceptions_config: Annotated[
        Path, typer.Option("--exceptions-config", exists=True, dir_okay=False)
    ] = Path("config/game_detail_exceptions.yaml"),
) -> None:
    """Build validated clean Parquet datasets from saved responses; no network access."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        build_datasets(
            raw_root=raw_root,
            games_root=games_root,
            seasons_path=seasons_config,
            overrides_path=overrides_config,
            teams_path=teams_config,
            exceptions_path=exceptions_config,
            repository=Repository(clean_root),
        )
    except (ValueError, OSError, KeyError) as exc:
        logging.getLogger(__name__).error("Build stopped: %s", exc)
        raise typer.Exit(code=1) from exc


@scrape_app.command("games")
def games(
    sample: Annotated[bool, typer.Option("--sample")] = False,
    seasons: Annotated[str | None, typer.Option("--seasons")] = None,
    live: Annotated[bool, typer.Option("--live")] = False,
    report: Annotated[bool, typer.Option("--report")] = False,
    config: ConfigOption = Path("config/scrape.yaml"),
    pages_config: Annotated[
        Path, typer.Option("--pages-config", exists=True, dir_okay=False)
    ] = Path("config/game_pages.yaml"),
) -> None:
    """Collect game pages from raw rounds, or verify existing pages with --report."""
    if sum((sample, seasons is not None, live)) != 1:
        raise typer.BadParameter("Choose exactly one of --seasons, --live or --sample")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        scrape_config = load_scrape_config(config)
        requested = (
            parse_seasons(
                seasons,
                minimum=scrape_config.first_season_id,
                maximum=scrape_config.live_season_id,
            )
            if seasons is not None
            else None
        )
        summary = run_games(
            scrape_config,
            pages_config,
            seasons=requested,
            live=live,
            sample=sample,
            report_only=report,
        )
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        if any(
            row["missing"] or row["invalid"] or row["rejected"] or row["status"] != "completed"
            for row in summary
        ):
            raise typer.Exit(code=1)
    except (AccessBlockedError, requests.RequestException, ValueError, OSError) as exc:
        logging.getLogger(__name__).error("Game collection stopped: %s", exc)
        raise typer.Exit(code=1) from exc


@export_app.command("excel")
def excel(
    clean_root: Annotated[Path, typer.Option("--clean-root")] = Path("data/clean"),
    teams_config: Annotated[Path, typer.Option("--teams-config")] = Path("config/teams.yaml"),
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Export eight validated datasets with canonical Hebrew names for inspection only."""
    try:
        path = export_excel(Repository(clean_root), output, teams_path=teams_config)
        typer.echo(f"Excel: {path.resolve()}")
    except (ValueError, OSError, KeyError) as exc:
        logging.getLogger(__name__).error("Excel export stopped: %s", exc)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
