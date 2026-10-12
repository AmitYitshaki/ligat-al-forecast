# EDA-01 stage B: real-data validation

Builds six in-memory analysis views for seasons 8?25. No clean-data publication, Excel export, held-out analysis, or stage C changes.

## Counts

| View | Rows |
| --- | ---: |
| `match_table` | 4,315 |
| `team_matches` | 8,630 |
| `goal_events` | 10,632 |
| `team_seasons` | 252 |
| `season_summary` | 18 |
| `coach_spells` | 441 |

All schedule, goal/halftime, points, relegation, squad coverage and coach overlap checks passed. Exactly 49 postponed games, all regular-stage. Two relegated clubs each season, except three in 2011/12. Game 864982 has no goal events; its three match goals explain the difference between 10,635 match goals and 10,632 events.

The seven technical-game exclusions in analysis seasons have schedule metadata in match_overrides.yaml. Four held-out exclusions receive only their already-known season identifiers; held-out data was not loaded. Exclusion actions, scores and points policies are unchanged.

## Coach spell sanity check

Counts refer to games with an observed coach. Missing-coach games bridge runs but are not assigned. Spell start flags describe the observed run, not a verified appointment date.

| Coach | Team ID | Seasons | Observed games |
| --- | ---: | --- | ---: |
| ברק בכר | 2171 | 17?21 | 161 |
| סלובודן דראפיץ | 1068 | 19?21 | 105 |
| אלישע לוי | 2171 | 14?16 | 105 |

Shortest spells:

| Coach | Team ID | Season | Observed games |
| --- | ---: | ---: | ---: |
| יצחק שום | 3595 | 11 | 1 |
| נימני אבי | 1061 | 11 | 1 |
| אמסלם דוד | 3595 | 11 | 1 |

## Missing-value inventory

Every nonzero missing count has a documented reason. Thresholds retain warn (>5%) and fail (>30%) labels; a documented fail label is a warning, while an unexplained fail blocks analysis.

| View.column | Missing | Share | Status |
| --- | ---: | ---: | --- |
| `match_table.home_coach_id` | 902 | 20.9% | warn |
| `match_table.home_coach_name` | 854 | 19.8% | warn |
| `match_table.away_coach_id` | 901 | 20.9% | warn |
| `match_table.away_coach_name` | 850 | 19.7% | warn |
| `match_table.referee` | 27 | 0.6% | ok |
| `match_table.home_prev_season_total_value_eur` | 836 | 19.4% | warn |
| `match_table.home_prev_season_avg_value_eur` | 836 | 19.4% | warn |
| `match_table.away_prev_season_total_value_eur` | 845 | 19.6% | warn |
| `match_table.away_prev_season_avg_value_eur` | 845 | 19.6% | warn |
| `match_table.home_goals_first_half` | 14 | 0.3% | ok |
| `match_table.away_goals_first_half` | 14 | 0.3% | ok |
| `match_table.first_goal_side` | 2 | 0.0% | ok |
| `match_table.first_goal_minute` | 411 | 9.5% | warn |
| `team_matches.referee` | 54 | 0.6% | ok |
| `team_matches.first_goal_minute` | 822 | 9.5% | warn |
| `team_matches.goals_first_half` | 28 | 0.3% | ok |
| `team_matches.opponent_goals_first_half` | 28 | 0.3% | ok |
| `team_matches.coach_id` | 1,803 | 20.9% | warn |
| `team_matches.coach_name` | 1,704 | 19.7% | warn |
| `team_matches.opponent_coach_id` | 1,803 | 20.9% | warn |
| `team_matches.opponent_coach_name` | 1,704 | 19.7% | warn |
| `team_matches.prev_season_total_value_eur` | 1,681 | 19.5% | warn |
| `team_matches.prev_season_avg_value_eur` | 1,681 | 19.5% | warn |
| `team_matches.ht_state` | 26 | 0.3% | ok |
| `team_matches.scored_first` | 824 | 9.5% | warn |
| `goal_events.added_time` | 10,632 | 100.0% | fail |
| `goal_events.half` | 38 | 0.4% | ok |
| `goal_events.minute_bucket` | 19 | 0.2% | ok |
| `goal_events.score_before` | 56 | 0.5% | ok |
| `team_seasons.n_coaches` | 36 | 14.3% | warn |
| `team_seasons.promoted_in` | 12 | 4.8% | ok |
| `team_seasons.avg_value_eur` | 1 | 0.4% | ok |
| `team_seasons.total_value_eur` | 1 | 0.4% | ok |
| `team_seasons.value_share_of_league` | 1 | 0.4% | ok |
| `team_seasons.prev_season_value_share` | 50 | 19.8% | warn |
| `coach_spells.coach_id` | 68 | 15.4% | warn |

## Returned warnings (21, all documented)

- match_table.home_coach_id: 902 missing (20.9%); No coaches in 2006/07–2008/09; partial coverage in 2009/10; later source omissions. IDs may be absent when names have no linked identifier (48 matches).
- match_table.home_coach_name: 854 missing (19.8%); No coaches in 2006/07–2008/09; partial coverage in 2009/10; later source omissions. IDs may be absent when names have no linked identifier (48 matches).
- match_table.away_coach_id: 901 missing (20.9%); No coaches in 2006/07–2008/09; partial coverage in 2009/10; later source omissions. IDs may be absent when names have no linked identifier (48 matches).
- match_table.away_coach_name: 850 missing (19.7%); No coaches in 2006/07–2008/09; partial coverage in 2009/10; later source omissions. IDs may be absent when names have no linked identifier (48 matches).
- match_table.home_prev_season_total_value_eur: 836 missing (19.4%); First analysis season has no predecessor; promoted teams; Hapoel Petah Tikva 2008/09 has no source value. No imputation.
- match_table.home_prev_season_avg_value_eur: 836 missing (19.4%); First analysis season has no predecessor; promoted teams; Hapoel Petah Tikva 2008/09 has no source value. No imputation.
- match_table.away_prev_season_total_value_eur: 845 missing (19.6%); First analysis season has no predecessor; promoted teams; Hapoel Petah Tikva 2008/09 has no source value. No imputation.
- match_table.away_prev_season_avg_value_eur: 845 missing (19.6%); First analysis season has no predecessor; promoted teams; Hapoel Petah Tikva 2008/09 has no source value. No imputation.
- match_table.first_goal_minute: 411 missing (9.5%); Unknown first scorer when both teams score in the same minute; unreliable scorers in 864982. First minute also absent for 0:0.
- team_matches.first_goal_minute: 822 missing (9.5%); No first goal in 0:0; unreliable scorers in 864982.
- team_matches.coach_id: 1803 missing (20.9%); Historical coach coverage and source omissions; a name may have no linked identifier.
- team_matches.coach_name: 1704 missing (19.7%); Historical coach coverage and source omissions; a name may have no linked identifier.
- team_matches.opponent_coach_id: 1803 missing (20.9%); Historical coach coverage and source omissions; a name may have no linked identifier.
- team_matches.opponent_coach_name: 1704 missing (19.7%); Historical coach coverage and source omissions; a name may have no linked identifier.
- team_matches.prev_season_total_value_eur: 1681 missing (19.5%); No previous league season for promoted clubs or 2006/07; Hapoel Petah Tikva 2008/09 source value missing.
- team_matches.prev_season_avg_value_eur: 1681 missing (19.5%); No previous league season for promoted clubs or 2006/07; Hapoel Petah Tikva 2008/09 source value missing.
- team_matches.scored_first: 824 missing (9.5%); Not applicable to 0:0; tied first-goal minute or unavailable scorers in 864982.
- goal_events.added_time: 10632 missing (100.0%); Source uses running minutes instead of + notation; first-half stoppage identified using half and minute.
- team_seasons.n_coaches: 36 missing (14.3%); No observed coaches in early seasons; unknown coaches are not invented.
- team_seasons.prev_season_value_share: 50 missing (19.8%); Previous league's share unavailable for first season, promoted teams or missing source value. No imputation.
- coach_spells.coach_id: 68 missing (15.4%); Source can provide a coach name without a linked ID, or an ID without a name.

## Validation

- `pytest`: 241 passed; `ruff check .`, `ruff format --check .`, `mypy ligat`: passed.
- Synthetic complete league plus deliberate corruption covers health checks, stoppage-time buckets, goal ordering, run lengths, coach gaps, previous-league denominators and sporting points benchmarks.
- Loader tested with unfiltered reads prohibited, and season filters rejecting 26?28. Real-data run also prohibited Repository.write.
- SHA256 hashes of all existing clean datasets and Excel files remained unchanged during the real-data run (Excel lock file excluded).
- Full local evidence: `data/reports/eda01b_validation.json` (ignored data artifact).
- Stop for Claude Code QA; no merge or stage C until approved.
