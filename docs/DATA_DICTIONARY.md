# מילון הדאטה

> כל עמודה בכל דאטה סט נקי (`data/clean/`). שינוי סכמה = עדכון הקובץ הזה + סכמת pandera באותו PR.

---

## `matches`: משחק ששוחק (שורה = משחק)

| עמודה | טיפוס | משמעות | מקור / הערה |
| --- | --- | --- | --- |
| `game_id` | int | מזהה המשחק בהתאחדות | מפתח ראשי |
| `season_id` | int | 8–28 | |
| `season` | str | למשל `"2025/26"` | נגזר |
| `round` | int | מספר מחזור רץ בעונה | |
| `stage` | category | `regular` / `upper` / `middle` / `lower` | נגזר מהמחזור ומ-`config/seasons.yaml` |
| `date` | datetime | תאריך ושעת פתיחה, `Asia/Jerusalem` | |
| `home_team_id` | int | | `team_id` של ההתאחדות |
| `away_team_id` | int | | |
| `home_goals` | int | | **אחרי** תיקון ההיפוך |
| `away_goals` | int | | **אחרי** תיקון ההיפוך |
| `venue_raw` | str | שם המגרש כפי שהופיע | |
| `venue` | str | שם מנוקה (בלי "(סגור)") | |
| `is_home_venue` | bool | האם זה האצטדיון הקבוע של קבוצת הבית באותה עונה | נגזר: המגרש השכיח ביותר של הקבוצה בעונה |
| `status` | category | `played` / `awarded` / `void` | `awarded` = נקבע בבית הדין; `void` = משחק ללא נקודות לשתי הקבוצות (סעיף 64, למשל דרבי 2014/15); מוחרג מאימון |
| `source` | str | `ifa` / `soccerway` / ... | |

## `fixtures`: משחק עתידי בעונה הנוכחית (שורה = משחק)

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `season_id`, `round`, `stage` | | כמו ב-`matches` |
| `date` | datetime | מועד מתוכנן (יכול להשתנות) |
| `home_team_id`, `away_team_id` | int | |
| `venue` | str | |
| `scraped_at` | datetime | מתי נמשך; הלוח מתעדכן |

אין `game_id` עד שהמשחק משוחק. מפתח זמני: `(season_id, round, home_team_id, away_team_id)`.

## `round_tables`: הטבלה הרשמית אחרי כל מחזור (שורה = קבוצה × מחזור)

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `season_id`, `round` | int | |
| `team_id` | int | |
| `position` | int | |
| `played`, `won`, `drawn`, `lost` | int | |
| `goals_for`, `goals_against` | int | **אחרי** תיקון ההיפוך |
| `points_model` | int | **הנקודות שבהן משתמשים בכל מקום:** 3×נ + ת, עם חלוקה לפי `seasons.yaml`. בלי הורדות |
| `points_official` | int | נקודות כפי שבאתר. **לאימות בלבד** |
| `deduction` | int | `points_official − points_model`. נרשם, לא משמש במודלים |

## `final_tables`: הטבלה הסופית (שורה = קבוצה × עונה)

אותן עמודות כמו `round_tables` במחזור האחרון, ובנוסף:

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `final_stage` | category | באיזה פלייאוף סיימה |
| `relegated` | bool | |
| `promoted_in` | bool | עלתה לליגה באותה עונה |
| `regular_season_points` | int | נקודות בסוף העונה הסדירה, לפני חלוקה |

## `teams` (`config/teams.yaml`)

| שדה | משמעות |
| --- | --- |
| `team_id` | מזהה ההתאחדות |
| `name_he` | שם קנוני בעברית |
| `name_en` | שם באנגלית |
| `aliases` | כל הכתיבים שנמצאו בדאטה |
| `city` | עיר לתיאור בלבד; אין פיצ׳ר או טיפול מיוחד בדרבי |

## `game_details`: מעמוד המשחק (שורה = משחק)

| עמודה | טיפוס | משמעות | זמין מ- |
| --- | --- | --- | --- |
| `game_id` | int | | |
| `ht_home_goals`, `ht_away_goals` | int | תוצאת מחצית | 2007 |
| `referee` | str | שופט ראשי | 2007 |
| `has_lineups` | bool | האם העמוד כולל הרכבים | |

## `goals` (שורה = שער)

| עמודה | טיפוס | משמעות | זמין מ- |
| --- | --- | --- | --- |
| `game_id`, `team_id` | int | הקבוצה שזכתה בשער | 2007 |
| `minute` | int | דקה (תוספת זמן: 45, 90 + עמודה נפרדת אם מופיע) | 2007 |
| `is_penalty` | bool | | 2007 |
| `is_own_goal` | bool | אם מסומן בעמוד | לבדוק |

## `red_cards` (שורה = כרטיס אדום)

| עמודה | טיפוס | משמעות | זמין מ- |
| --- | --- | --- | --- |
| `game_id`, `team_id` | int | | 2012 |
| `minute` | int | | 2012 |

## `coaches` (שורה = קבוצה × משחק)

| עמודה | טיפוס | משמעות | זמין מ- |
| --- | --- | --- | --- |
| `game_id`, `team_id` | int | | 2012 |
| `coach_name` | str | | 2012 |

בעונות שבהן שדה לא קיים: `null` מפורש, לא 0.
