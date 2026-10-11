"""CLI entry points; importing this module never starts a download."""

import logging
from pathlib import Path
from typing import Annotated

import requests
import typer

from ligat.data.build import build_datasets
from ligat.data.repository import Repository
from ligat.scrape.ifa import AccessBlockedError, IFASource, load_scrape_config

app = typer.Typer(no_args_is_help=True)
scrape_app = typer.Typer(no_args_is_help=True)
app.add_typer(scrape_app, name="scrape")
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
    clean_root: Annotated[Path, typer.Option("--clean-root")] = Path("data/clean"),
    seasons_config: Annotated[
        Path, typer.Option("--seasons-config", exists=True, dir_okay=False)
    ] = Path("config/seasons.yaml"),
    overrides_config: Annotated[
        Path, typer.Option("--overrides-config", exists=True, dir_okay=False)
    ] = Path("config/match_overrides.yaml"),
    teams_config: Annotated[Path, typer.Option("--teams-config")] = Path("config/teams.yaml"),
) -> None:
    """Build validated clean Parquet datasets from saved responses; no network access."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        build_datasets(
            raw_root=raw_root,
            seasons_path=seasons_config,
            overrides_path=overrides_config,
            teams_path=teams_config,
            repository=Repository(clean_root),
        )
    except (ValueError, OSError, KeyError) as exc:
        logging.getLogger(__name__).error("Build stopped: %s", exc)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
