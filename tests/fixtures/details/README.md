# T-005 golden pages

Exact gzip copies from the approved T-004 crawl, without changing raw storage.
JSON expectations include header facts, events, both coaches and availability.
Reviewed facts: 113478 finishes 4:1 (HT 1:0), with two second-yellow dismissals;
211600 finishes 2:1 without lineups; 211773 finishes 3:2 with eleven starters
per team; 240575 finishes 3:1; 1120561 finishes 2:1 with one missing coach.

240618 is excluded from clean datasets, but deliberately retained as a parser
fixture: the actual match ended 3:3 (HT 2:2). The own goal at minute 26 appears in the home header goalers and is
credited directly to home team 1198, not to away team 2171. Coach ID -1 is missing.

Periods: 2006/07, 2009/10 with and without lineups, 2010/11 and 2026/27.
No fixtures from holdout seasons 2024/25 or 2025/26 are used.

428714 covers a source minute 45 assigned to the second half by the header rule.
113512 and 864982 cover documented source exceptions and unsuppressed checks.

268007 is the real minute-46 first-half stoppage fixture requested by Claude QA.
Goal expectations use credited header sides, not the scorer side in the timeline.
