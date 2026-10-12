# EDA-01 C: execution and validation

EDA-01 stage C provides an executed Hebrew notebook telling the story from validated data to a pre-match feature row. Each code cell follows a short Markdown explanation and displays output.

## Scope and implementation

- Seasons 8?25 only; no held-out reads, clean-data writes, Excel export or training.
- Prototype preview_pre_match_row uses strictly earlier timestamps and a last-10-game window. Simultaneous/current/future games are excluded. Venue rates split that same window; unknown/conflicting halftime values remain missing.
- B0 overall and per-season outcome rates with marginal 95% Wilson intervals; full-sample rates are descriptive, not out-of-sample scores.
- Coach quality flags identify left censoring and short A/B/A runs. Flags are retrospective diagnostics, not pre-match features; no source rows or coach identities are changed.
- nbstripout Git filter strips only the staged copy. The hook now verifies staged notebooks without mutating local outputs; regression tests cover clean/dirty staged copies and retain the existing venv/Plotly kernel test.

## Fresh-kernel execution

- 31 Markdown cells + 29 code cells = 60 total; 18 sections; 6 Plotly figures.
- Full Restart & Run All: 161.23 seconds, using the project venv kernel.
- Reopened local notebook: every code cell has outputs and an execution count, six Plotly MIME outputs are present, and tables have HTML/plain-text outputs. Display settings show all 21 warnings and the complete 18-row season summary.
- Local execution evidence: data/reports/eda01c_execution.json. Commit/staged output preservation is verified separately.

## Coach flags by starting season

```text
         is_left_censored  is_flicker
2009/10                16           7
2010/11                 0          11
2011/12                 0           0
2012/13                 0           1
2013/14                 1           1
2014/15                 0           1
2015/16                 0           0
2016/17                 0           1
2017/18                 0           1
2018/19                 0           2
2019/20                 0           3
2020/21                 0           3
2021/22                 0           2
2022/23                 0           9
2023/24                 0           6
```

Totals: 17 left-censored runs and 48 flickers. Counts are descriptive source-quality flags.

## From data to model: game 793896

```text
                             date home_team_name away_team_name  home_goals  \
game_id
793896  2023-02-25 20:00:00+02:00      מכבי חיפה      בית"ר י-ם           2

         away_goals  ht_home_goals  ht_away_goals  home_red_cards  \
game_id
793896            1              1              1               0

         away_red_cards
game_id
793896                1
```

```text
game_id                               ערך פיצ׳ר
feature_home_goals_scored_mean              1.5
feature_home_goals_conceded_mean            0.9
feature_home_points_earned_mean             2.0
feature_home_ht_goals_scored_mean           0.5
feature_home_home_points_per_match          2.6
feature_home_away_points_per_match          1.4
feature_home_prev_season_value_share   0.120988
feature_away_goals_scored_mean              2.1
feature_away_goals_conceded_mean            1.4
feature_away_points_earned_mean             1.8
feature_away_ht_goals_scored_mean           1.2
feature_away_home_points_per_match          2.0
feature_away_away_points_per_match          1.6
feature_away_prev_season_value_share   0.069146
```

```text
game_id           יעד בפועל
target_home_goals         2
target_away_goals         1
target_result      home_win
```

## Validation

- pytest: 248 tests; ruff check, ruff format --check and mypy ligat.
- Leakage, strict timestamps, empty history, Wilson rates/intervals, coach left censoring, flicker length and local-output preservation covered by tests.
- Analysis health checks passed with the same 21 documented warnings.
- EDA task, data dictionary, notebook README and technical guidance updated.
- Stop for Claude Code QA. No merge or follow-up branch.

Post-commit check passed: the local notebook is byte-for-byte unchanged with all outputs; the committed notebook has no outputs or execution counts.
