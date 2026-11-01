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
| `is_home_venue` | bool | האם זה האצטדיון הקבוע של קבוצת הבית באותה עונה | תיאור בדיעבד: המגרש השכיח במשחקי הבית בעונה; שוויון מוכרע לפי שם. אינו פיצ׳ר לחיזוי ללא חישוב מחדש לפני `as_of` |
| `status` | category | `played` בלבד | הכרעות טכניות ומשחקי void מוחרגים מ־matches לפי match_overrides; נשמרים בקבצים הגולמיים |
| `source` | str | `ifa` / `soccerway` / ... | |

## משחקים שהוחרגו מ־matches — החלטת עמית, 11.10.2026

| game_id | עונה | סטטוס | פעולה |
| --- | --- | --- | --- |
| 211602 | 2009/10 | awarded | drop |
| 240618 | 2010/11 | awarded | drop |
| 388645 | 2014/15 | void | drop |
| 525468 | 2017/18 | awarded | drop |
| 732553 | 2021/22 | awarded | drop |
| 732312 | 2021/22 | awarded | drop |
| 954863 | 2024/25 | void | drop |
| 1060422 | 2025/26 | awarded | drop |
| 339766 | 2013/14 | awarded | drop |
| 954984 | 2024/25 | awarded | drop |
| 954988 | 2024/25 | awarded | drop |

כל 11 המשחקים מאומתים בקונפיגורציה. הפענוח וחישוב הטבלאות נעשים לפני
החרגת הרשומות; matches כולל 4,831 שורות מהגרידה הנוכחית, כולן played.
שחזור טבלה מתוך matches לא ייסגר בעונות 11, 12, 15, 16, 19, 23, 26, 27.
כמו כן, שערי הטבלה הרשמית ב־2006/07 לא אמינים.

לפני כתיבת פלט, `ligat build` מצליב לכל משחק ליגה בעל game_id את תוצאת
`.game-time .total` בעמוד הגולמי (בית : חוץ) מול התוצאה המפוענחת מהמחזור (חוץ - בית במקור).
עמודי gzip נבדקים מול הגודל ו־SHA-256 ב־manifest שהושלם, כולל התוכן הלא דחוס.
פער שאינו מופיע ב־override מאומת מכשיל את הבנייה ומציג את רשימת המשחקים;
אין כתיבת Parquet או טיוטת teams במקרה זה. אין אפשרות לדלג על הבדיקה.
משחקים עתידיים ללא game_id ומשחקי מבחן אינם נבדקים. void ב־0:0 בשני המקורות
אינו מזוהה אוטומטית ונשאר החרגה ידנית (למשל 954863).
round_tables ו־final_tables נשמרות בערכים הקיימים ללא שינוי.


חישוב ה־void המאומת של 2014/15 אינו משתנה. בהבהרה נוספת עמית ביקש לשמור כרגע
את כל ערכי round_tables ו־final_tables הקיימים ללא שינוי: לכן `adjust_points: false`
ב־954863 דוחה את תיקון נקודת התיקו שלו. משמעות הדבר היא שהמשחק נעלם מ־matches,
אבל points_model בטבלאות עונת 2024/25 עדיין כולל את הנקודה הזו עד החלטה נוספת.
אין לפרש את האימות החדש כאישור לחישוב הניקוד הקיים בעונה זו.

## `fixtures`: משחק עתידי בעונה הנוכחית (שורה = משחק)

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `season_id`, `round`, `stage` | | כמו ב-`matches` |
| `date` | datetime | מועד מתוכנן ב־`Asia/Jerusalem` (יכול להשתנות); שעה שטרם נקבעה מיוצגת כחצות, ולא כשעת פתיחה מאומתת |
| `home_team_id`, `away_team_id` | int | |
| `venue` | str | |
| `scraped_at` | datetime | חותמת `downloaded_at` של תגובת המחזור, מומרת ל־`Asia/Jerusalem`; מה־snapshot האחרון שהושלם |

אין `game_id` עד שהמשחק משוחק. מפתח זמני: `(season_id, round, home_team_id, away_team_id)`.

## `round_tables`: הטבלה הרשמית אחרי כל מחזור (שורה = קבוצה × מחזור)

> **לאימות בלבד; אינן קלט למודלים. מצב טבלה למודלים מחושב מ-matches לפי as_of (VISION §4.3).**

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `season_id`, `round` | int | |
| `team_id` | int | |
| `position` | int | |
| `played`, `won`, `drawn`, `lost` | int | |
| `goals_for`, `goals_against` | int | **אחרי** תיקון ההיפוך |
| `points_model` | int | **הנקודות שבהן משתמשים בכל מקום:** 3×נ + ת, פחות נקודת תיקו לכל משחק `void` מאומת. בפלייאוף עונות 11–12 בלבד: עיגול מעלה של מחצית הנקודות הסדירות, בתוספת נקודות הפלייאוף המלאות. בלי הורדות |
| `points_official` | int | נקודות כפי שבאתר. **לאימות בלבד** |
| `deduction` | int | `points_official − points_model`. נרשם, לא משמש במודלים |

## `final_tables`: הטבלה הסופית (שורה = קבוצה × עונה)

> **לאימות בלבד; אינן קלט למודלים. מצב טבלה למודלים מחושב מ-matches לפי as_of (VISION §4.3).**

אותן עמודות כמו `round_tables` במחזור האחרון, ובנוסף:

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `final_stage` | category | באיזה פלייאוף סיימה |
| `relegated` | bool nullable | לא נכללת בסגל הקבוצות בעונה העוקבת; נגזר ממזהים בטבלאות המקור, כולל תוצאת מבחני הישרדות. null כאשר אין עונה עוקבת בקלט |
| `promoted_in` | bool nullable | לא נכללה בסגל הקבוצות בעונה הקודמת. null בעונה הראשונה בקלט, שבה אין מידע על העונה הקודמת |
| `regular_season_points` | int | נקודות בסוף העונה הסדירה, לפני חלוקה |

טבלת הסיום נלקחת מהמחזור האחרון שאינו מבחן עלייה, כולל נתוני הקבוצות שכבר סיימו
את הפלייאוף שלהן. העונה החיה מוחרגת. `final_stage` נקבע לפי מקום הקבוצה בטבלת
סיום העונה הסדירה וגדלי הקבוצות ב־`playoff_groups`. תוצאות מועמדות ולא מאומתות
בקובץ ה־overrides אינן משנות סטטוס או נקודות; השלמת האימות היא חלק מ־T-007.

## `teams` (`config/teams.yaml`)

| שדה | משמעות |
| --- | --- |
| `team_id` | מזהה ההתאחדות |
| `name_he` | שם קנוני בעברית |
| `name_en` | שם באנגלית |
| `aliases` | כל הכתיבים שנמצאו בדאטה |
| `city` | עיר לתיאור בלבד; אין פיצ׳ר או טיפול מיוחד בדרבי |

## `game_details`: מעמוד המשחק (שורה = משחק)

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `game_id` | int | מפתח ל־matches |
| `ht_home_goals`, `ht_away_goals` | Int64 nullable | תוצאת המחצית, בית וחוץ; חסר אינו אפס |
| `referee` | string nullable | שופט ראשי |
| `has_timeline` | bool | יש צמתי ציר זמן, לרבות סמני זמן במשחק ללא אירועים |
| `has_lineups` | bool | קיימים שחקני הרכב בשני הצדדים |
| `has_coaches` | bool | קיים שם מאמן בשני הצדדים; זיהוי מספרי אינו תנאי |
| `ht_conflict` | bool | True כאשר תוצאת המחצית בכותרת סותרת את דקות השערים (13 משחקים מתועדים) |
| `goals_available` | bool | False כאשר לא ניתן לפענח שערים עקב שגיאת מקור קיצונית (משחק 864982) |

## `goals` (שורה = שער)

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `game_id`, `team_id` | int | המשחק והקבוצה שהשער נזקף לזכותה |
| `half` | Int64 nullable | 1 למחצית ראשונה, 2 לשנייה; null במשחקי ht_conflict |
| `minute` | int | דקה במקור (ההתאחדות רושמת דקות רצות: 45+2 נרשם כ-47) |
| `added_time` | Int64 nullable | תוספת זמן כשנכתבה במפורש; בפועל תמיד null כיוון שההתאחדות רושמת דקות רצות (למשל 47), וזיהוי תוספת זמן במחצית 1 מתבצע דרך is_first_half_stoppage |
| `is_first_half_stoppage` | bool | True כאשר השער במחצית ראשונה ודקתו גדולה מ-45 |
| `is_penalty`, `is_own_goal` | bool | סוג האירוע לפי ציר הזמן |

השערים נקראים מרשימות המבקיעים בכותרת (.team-home/.team-guest .goalers).
בכותרת שער עצמי כבר מופיע בצד הקבוצה שהרוויחה; team_id נלקח מצד זה בלי היפוך נוסף.
סימוני (פ) ו־(ע) מפוענחים מהכותרת. הכרטיסים האדומים נקראים מציר הזמן.
שיוך שערים למחצית (half): לכל קבוצה, השערים הראשונים לפי סדר הדקות, כמספר שערי המחצית שלה, הם של המחצית הראשונה.
ב-13 משחקי ht_conflict שבהם יש סתירה, השדות נשמרים כ-null ב-half.
במשחק 864982 (6 מבקיעים במשחק של 1:2) goals_available=False ואין שורות ב-goals.
14 החריגים מתועדים ב-`config/game_detail_exceptions.yaml`, עם verified=true וסיבה מפורשת.
במשחקי ht_conflict, is_first_half_stoppage=False כי המחצית אינה ידועה.
השיוך נגזר מכותרת המחצית ואינו אימות עצמאי של נכונותה: נבדקת התאמת ספירת השערים לשיוך,
וכן נכשלים אם שער שסומן למחצית שנייה נמצא לפני דקה 45,
או אם שער שסומן למחצית ראשונה נמצא אחרי דקה 60.
דקה 45 מותרת במחצית השנייה, ודקה 60 מותרת במחצית הראשונה.
פער לא מתועד בסך השערים, במחצית שאינה ניתנת לשיוך או בדגלי הזמינות מכשיל את הבנייה.
חריג מחצית אינו עוקף בדיקות סיום או דגלים; חריג מבקיעים אינו עוקף את בדיקת הכותרת מול matches.

## `red_cards` (שורה = הרחקה)

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `game_id`, `team_id` | int | המשחק והקבוצה שהשחקן שלה הורחק |
| `minute` | Int64 nullable | דקה; ערך המקור -1 נשמר כחסר (null) ומפורט בדוח האימות |
| `is_second_yellow` | bool | אירוע yellow2; אירוע red הוא False |

מידע זה זמין כבר ב־2006/07 דרך ציר הזמן. אין מחיקת כרטיס כשדקתו חסרה (12 מקרים היסטוריים עם minute=null).

## `coaches` (שורה = קבוצה × משחק)

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `game_id`, `team_id` | int | שתי שורות לכל משחק, גם כשאין מאמנים |
| `coach_id` | Int64 nullable | מזהה מה־href; חסר או -1 במקור נשמר כ־null |
| `coach_name` | string nullable | שם מאמן כפי שנמצא; null כשאין שם; זמינות לפי העמוד, גם בחלק ממשחקי 2009/10 |

כל ארבעת הדאטה סטים מוחרגים מאותם 11 משחקים טכניים כמו matches.
הרכבים מופיעים בחלק ממשחקי 2009/10; זמינות נקבעת לפי תוכן ולא לפי שנה.
הבנייה בודקת SHA-256 וגודל של gzip ושל התוכן מול manifest שהושלם.
הצלבות: שערים לכל קבוצה מול הסיום, שיוך מחצית לפי כלל סדר הדקות,
ודגלי has_* מול התוכן. פערים שאינם ברשימת החריגים נשמרים ב־data/reports/game_details_validation.json
ומכשילים את הבנייה לפני כתיבת Parquet. אזהרות על דקות כרטיסים חסרות נשמרות בנפרד.

## `squad_values` (T-005b; שורה = קבוצה × עונה)

קובץ: `data/clean/squad_values.parquet`. מפתח ייחודי: `season_id, team_id`;
גם `season_id, tm_club_id` חייב להיות ייחודי.

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| `season_id` | int64 | מזהה עונת ההתאחדות 8–28; שנת תחילת העונה ב־Transfermarkt פחות 1998 |
| `tm_club_id` | int64 | מזהה המועדון מתוך קישור Transfermarkt |
| `team_id` | int64 | מזהה ההתאחדות לפי מיפוי ידני ב־config/transfermarkt_teams.yaml |
| `squad_size` | Int64 nullable | כל השחקנים שנכללו בסגל במהלך העונה, לא גודל הסגל ביום מסוים |
| `avg_value_eur` | Int64 nullable | השווי הממוצע המוצג בדף, באירו |
| `total_value_eur` | Int64 nullable | סכום שווי כל השחקנים שהיו בסגל במהלך העונה, באירו |

דף העונה אינו שווי בתאריך מסוים. אין כאן תאריך מדידה מוכח ואין אישור
להשתמש בנתון כפיצ'ר t-1 ללא בדיקה מתודולוגית נפרדת. עונת 2026/27 עדיין
פתוחה ונתוניה הם תמונת הדף בזמן האיסוף. סימן `-` נשמר null, ללא השלמות.
העמודות avg_age ו־foreigners_count אינן נכללות בהיקף שאושר לשלב ב'.

`ligat build-squad-values` בודק חתימות וגודל מול manifest, מיפוי חד־חד־ערכי,
כיסוי מדויק של קבוצות כל עונה מול round_tables, וסיכום הליגה עד דיוק העיגול
של ערכי המקור. אם חסרים ערכי קבוצה, נבדק רק שסכום הערכים הידועים אינו
עולה על הסיכום; אין זקיפת ערכים. דוח: `data/reports/t005b_build.json`.
כל הבדיקות מתקיימות לפני כתיבה אטומית דרך Repository.

## Excel לצפייה

כאשר squad_values.parquet קיים, הייצוא כולל גם גיליון squad_values עם שם עברי
ליד team_id, והחתימה שלו מצטרפת ל־JSON הסנכרון. אין קריאה מהאקסל בקוד.

`ligat export excel` מייצא את שמונת קובצי ה־Parquet המאומתים לקובץ ממוספר
תחת data/clean; הגרסה הראשונה בתשתית זו היא ligat_al_data_0.3.xlsx.
עמודות *_name_he נצמדות למזהי הקבוצות לפי name_he ב־config/teams.yaml.
שמות אלה הם שמות קנוניים נוכחיים, ולא שם הקבוצה במועד המשחק.
התאריכים מיוצאים כשעה מקומית בישראל ללא אזור זמן. שום קוד לא קורא מהאקסל.
שינוי בדאטה או בקובץ השמות יוצר גרסה קטנה חדשה; גרסאות קיימות אינן נדרסות.
קובץ JSON צמוד שומר ספירת שורות וחתימות מקור ואקסל לאימות הסנכרון.


### אימות לפני אישור QA

`ligat build --validate-only` מפענח, מצליב ומאמת את שמונת הדאטה סטים בזיכרון,
ושומר את הדוח ב־data/reports/game_details_validation.json בלבד.
אינו כותב Parquet, אינו משנה teams.yaml ואינו מפיק Excel.
הפקה קודמת של Excel 0.3 אינה מהווה אישור QA למיזוג או לפרסום נוסף.
# תצוגות EDA-01 ומחברת 01 (ללא קובצי Parquet חדשים)

המחברת `notebooks/explore/01_prepare_analysis_tables.ipynb` מציגה שש תצוגות
בזיכרון, לעונות 8–27 השלמות, בהתאם לעדכון C2. אין כתיבה ל־data/clean,
השלמה או אימון. עונה 28 חיה ואינה נטענת.

season_role הוא category ב־match_table, team_matches, team_seasons
ו־season_summary: development (8–25), final_validation (26), test (27).
הסטטיסטיקות המחקריות, B0 ו־teams_overview מוגבלים ל־development.
team_profile מציג עונות קבוצה עם תפקיד, מיקום, שערים ונקודות למשחק ושווי;
data_availability מציג אחוזי כיסוי לכל עונה, כולל שני התפקידים הסופיים.
במחצית סתירת מקור אינה זמינה; במאמנים ובשווי מחשבים ממוצע בין שני הצדדים.

תוספות ל־`coach_spells` (אינדקס `team_id, spell_number`):

| עמודה | טיפוס | משמעות |
| --- | --- | --- |
| is_left_censored | boolean | הרצף מתחיל במשחק הראשון בעונה שבו נצפה מאמן, אך אחרי משחק 1; מועד ההתחלה האמיתי אינו ידוע |
| is_flicker | boolean | רצף בן עד שני משחקים בין שני רצפים של אותו מאמן באותה קבוצה, A→B→A; סימון איכות, ללא מחיקה |

הדגלים הם תיאור בדיעבד, לא פיצ׳רים טרום־משחק; ריצוד תלוי גם ברצף הבא.
הדגלים משויכים לעונת תחילת הרצף. משחקי מאמן חסר מחברים רצף אך אינם
משויכים למאמן. is_mid_season_start נשאר דגל של תחילת רצף נצפה; בשילוב
is_left_censored אין להסיק ממנו שהוחלף מאמן. דגלי האיכות אינם חסרים.

`preview_pre_match_row(tables, game_id, window=10)` הוא אב־טיפוס בלבד,
שנשאר בחבילה אך אינו מופעל במחברת 01 לאחר C2:
שורה אחת באינדקס game_id, עם feature_home_* ו־feature_away_* מול
target_home_goals, target_away_goals ו־target_result. ממוצעי שערים, ספיגה,
נקודות ושערי מחצית משתמשים בעד window משחקים שתאריכם קטן ממש מתאריך
המשחק, גם מעבר לגבול עונה. ממוצעי נקודות בית/חוץ מפצלים את אותו חלון.
מחצית סותרת מוחרגת מממוצע המחצית. היסטוריה ריקה נשארת חסרה.
prev_season_value_share הוא נתח הקבוצה בליגה של t−1, ללא השלמה לעולות.
אין שימוש בתוצאת המשחק, במשחקים מאותו זמן או במשחקים עתידיים בפיצ׳רים.

`outcome_base_rates` מחזיר scope (overall/season), season_id (חסר בשורת
overall), outcome (home_win/draw/away_win), n_matches, n_outcomes, rate,
ci_lower, ci_upper. השיעורים ורווחי Wilson של 95% הם תיאור בלבד של הדאטה
שנמסר לפונקציה; בבקטסט נדרש למסור אימון בלבד. הרווחים שוליים לכל תוצאה,
ואינם רווח משותף לשלוש ההסתברויות.

## EDA-02, פעימה A — סטטיסטיקות ליגה תיאוריות

התצוגות הבסיסיות הבאות מחושבות בזיכרון ב-`ligat/analysis/league_stats/core.py`; אינן
פיצ'רים ואינן נכתבות ל-data/clean. בפונקציות אלה `seasons` הוא development
(8–25) כברירת מחדל; בקשת עונה אחרת, רשימה ריקה או כפילות נכשלת.
נקודות הן 3 לניצחון ו-1 לתיקו מכל matches, ללא עונשים וללא חציית נקודות.
המיקום, שלב הסיום ודגלי הירידה מקורם בתוויות הרשמיות ב-team_seasons,
והם הקשר תיאורי בלבד. 11 המשחקים הטכניים אינם ב-matches.

| תצוגה | שורה / אינדקס | עמודות ומשמעות |
| --- | --- | --- |
| team_season_stats | קבוצה×עונה; season_id, team_id | מדדי team_seasons ללא חישוב מחדש, n_teams מ-season_summary ו-era מ-team_matches; win/draw/loss, ניצחונות בית וחוץ, ניצחון גדול (הפרש ≥3), ניצחון צר (1), הפסד כבד (≤−3), BTTS ו-over_2_5; הפרש שערים למשחק ופער נקודות בית–חוץ |
| position_profile | עידן×מיקום רשמי; era, final_position | n וממוצע/חציון/מינימום/מקסימום לנקודות, שערי זכות וחובה למשחק, עם ci_lower/ci_upper לממוצעים. עידני 12/16/14 קבוצות אינם מאוחדים |
| season_persistence | קבוצה×עונה t; season_id, team_id | ppm_t, ppm_prev, position_t, position_prev, value_share_prev ושם; רק קבוצות ששיחקו גם ב-t−1. עונה 8 אינה נכנסת, ואין גישור על עונת היעדרות. t−1 נשאר development גם כשמבקשים רק עונת t |
| persistence_summary | קשר; relationship=ppm/position | n לזוגות מלאים, n_seasons, pearson, spearman, slope, intercept ורווחי 95% למתאמים ולשיפוע. שיפוע ההתמדה אינו אחוז דיוק או הוכחת סיבתיות |
| value_vs_performance | קבוצה×עונה; season_id, team_id | נתח שווי קודם ונתח אותה עונה, ppm, מיקום, squad_size; expected_ppm לפי רגרסיה תיאורית של ppm על נתח קודם; over_under=ppm−expected_ppm. השארית בתוך המדגם, ללא הערכת תחזית |
| value_summary | קשר; relationship=prev_season_value_share/value_share_of_league/squad_size | n לזוגות מלאים, n_seasons, מתאמים, רגרסיה תיאורית ורווחי 95%. נתח אותה עונה כולל מידע סוף עונה ואינו קלט למודל באותה עונה |
| promoted_value_check | עולה×עונה; season_id, team_id | רק promoted_in=True ובעונות 9–25: נתח שווי בעונה t, imputed_share כחציון החצי התחתון של הליגה בעונת t−1; same_season_share_excl_promoted הוא חציון החצי התחתון בעונה t לאחר הסרת כל העולות, להשוואה משנית בלבד. diff=ממשי−כלל t−1, ratio=ממשי/כלל t−1. לא מתבצעת השלמה בפועל |
| promoted_value_summary | סיכום; scope=promoted | n לזוגות מלאים, n_missing, median_diff ו-fraction_below בטווח 0–1 עם רווח Wilson. המקור החסר אינו אפס ואינו נכנס למכנה |
| team_consistency | קבוצה; team_id | seasons, mean_ppm ו-std_ppm בין עונות (חסר בפחות מ-3), רווח ממוצע, מיקומי שיא/שפל רשמיים, titles, top3_seasons, relegations, promotions ו-is_yoyo (לפחות שתי עליות נצפות בפיתוח) |
| team_style_profile | קבוצה עם לפחות 5 עונות; team_id | ממוצעים עונתיים שווי משקל של btts_rate ו-over_2_5_rate, רווחי 95%, n_seasons ו-n_matches. זו אינה חלוקה בסך המשחקים בכל העונות |

כל שיעור משחק ב-team_season_stats מקבל עמודות `<rate>_n`,
`<rate>_ci_lower`, `<rate>_ci_upper` בשיטת Wilson של 95%. שיעורי בית/חוץ
מחולקים רק במשחקי הצד המתאים; יתר השיעורים בכל משחקי הקבוצה בעונה.
אין משחקים במכנה → שיעור ורווח חסרים, לא אפס.

רווחי ממוצעי עונות מבוססים על 2,000 דגימות bootstrap של תצפיות עונה.
רווחי קשרים מבוססים על 1,000 דגימות של בלוקי עונה עם כל הקבוצות באותו בלוק.
אלה ברירות המחדל במחברת; `n_bootstrap` מאפשר לבחור מספר דגימות חיובי
ב-position_profile, persistence_summary, value_summary, team_consistency
ו-team_style_profile. הטסטים משתמשים ב-50 דגימות, ללא שינוי אומדני המרכז.
הזרע קבוע (20261014); הרווחים שוליים, ללא תיקון להשוואות מרובות, ואינם
מבטלים את התלות בין עונות של אותה קבוצה. פחות משתי עונות או שונות אפס
מניבים רווח/מתאם חסר לפי העניין.

בבדיקת העולות, ערכי שווי חסרים נשארים חסרים. החצי התחתון נבחר רק מערכים
זמינים לפי השווי, ובמספר אי-זוגי גודלו floor(n/2). ערך הכלל נלקח מהעונה
הקודמת: עולות עונה 9 משתמשות בעונה 8, גם כאשר מבקשים רק את עונת העלייה.
העולה אינה חלק מליגת t−1. השווי בפועל בעונה t וההשוואה המשנית ללא עולות
הם תיאוריים בדיעבד; זו בדיקת סבירות לכלל prematch ולא השלמה או שינוי בכלל.
בפעימה A אין ניתוחי מחצית או דקות שערים, ולכן ht_conflict ו-goals_available
אינם מסננים תוצאות משחק מלא. אין שינוי בסכמות או בקונפיגורציית היורדות.

### כללי — שיעורים וגבולות הסינון

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `rate_interval` (`core.py`) | ללא; שלשה | קלט `count`, `n`, עם ‎0≤count≤n. מחזירה `(rate, ci_lower, ci_upper)` של 95% Wilson. כאשר n=0 שלושת הערכים NaN; ערכים שליליים או count>n גורמים ל־ValueError. אין פרמטר seed או bootstrap. |

הפונקציה הכללית אינה מקבלת עונות ואינה מסננת נתונים: האחריות על המכנה ועל development היא של הקורא. Wilson הוא רווח שולי לשיעור בינומי; אינו מתקן תלות, בחירת מדגם או ריבוי השוואות. בתיעוד להלן `attrs` מפורטים כאשר הפונקציה קובעת אותם במפורש. טבלאות העזר המקבלות `frame` אינן מקבלות אוטומטית את חוזה הסינון של פונקציות המקבלות `AnalysisTables`.

### A — בדיקות הסקירה (`chapter_a.py`)

בפונקציות המקבלות `tables`, ברירת המחדל היא `ERA14_SEASONS` (14–25), פרט ל־`promoted_rule_comparison` שברירת המחדל שלה היא כל development. `seasons` נבדק כבחירה לא ריקה, ללא כפילות ומתוך 8–25; הבחירה אינה מורחבת אוטומטית לכל עונות עידן 14. יוצאי הדופן המפורשים של היסטוריית t−1/t−2/t−3 מתוארים להלן.

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `correlation_evidence` | `relationship` | קלט `frame`, ורשימת `pairs` של `(name,x,y)`; `method="pearson"` או `"spearman"`; `comparisons=None` פירושו גודל הרשימה, או משפחה מפורשת שגודלה לפחות מספר הזוגות. פלט `n` לזוגות מלאים, `correlation`, `p_value`, `comparisons`, `p_bonferroni`. מתאם ו־p חסרים בפחות משלוש שורות מלאות או באחד המשתנים הקבועים. |
| `adjacent_pairs` | `season_id, team_id` של עונת t | קלט `frame` באינדקס עונה×קבוצה ו־`columns`. מחזירה inner join עם אותה קבוצה ב־t−1, כאשר עמודות העבר מקבלות `_prev`; יתר עמודות t נשמרות. אין גישור על שנת היעדרות, ואין הוספת עונות שלא נמסרו. שני הצדדים חייבים להיות חד־ערכיים לעונה×קבוצה. |
| `era_persistence` | זוג: `pairs` בעונה×קבוצה; סיכום ב־`relationship=ppm/position` | קלט `tables`, `seasons`, `n_bootstrap=1000`. בזוגות נשמרים מדדי הקבוצה, ונוצרים `ppm_prev`, `ppm_t`, `position_prev`, `position_t`, `ppm_change`, `season` כמזהה מספרי. בסיכום `n`, `n_seasons`, `pearson`, `spearman`, `slope`, `intercept`, גבולות `<metric>_ci_lower/upper` למתאמים ולשיפוע, `residual_sd`, `mean_prev`, `mean_t`. שתי העונות חייבות להיות בבחירה. |
| `loso_predictions` | אינדקס השורות המלאות מתוך הקלט | קלט `frame` עם `season_id` באינדקס, עמודת `points_per_match` ורשימת `predictors` לא ריקה וללא כפילויות. דורשת לפחות שתי עונות מלאות מתוך development. OLS עם חותך על כל העונות למעט עונת המטרה; פלט `observed`, `predicted`, `n_train`, `error=observed−predicted`. שורות עם חסרים במטרה או במנבאים מושמטות לפני קביעת הקיפולים. |
| `predictor_comparison` | זוג: סיכום ב־`scope,predictor`; זוגות ב־`season_id,team_id` | קלט `tables`, `seasons`. סיכום `n`, `r_squared` בתוך המדגם ו־`loso_rmse`. `adjacent`: שווי קודם רגיל/לוג, נקודות קודמות, הפרש שערים קודם ושילוביו עם שווי; `history_common`: הפרש קודם ושילוביו עם שווי ו־`history_gd`. טבלת הזוגות מוסיפה `gd_lag2`, `gd_lag3`, `history_gd` ו־`log_prev_value`. ההיסטוריה היא ממוצע זמינים מ־t−2/t−3, מתוך development גם אם מחוץ לבחירת עידן 14; t ו־t−1 עצמם בתוך הבחירה. חסר שווי קודם מוצא, ולוג דורש ערך חיובי; n מדווח לכל מפרט. |
| `home_advantage_diagnostics` | Series של מדדים | קלט `tables`, `seasons`. `n` קבוצות־עונות, `n_pairs` זוגות עונות סמוכות, `mean_gap` ממוצע פער נקודות למשחק בית–חוץ, `persistence` המתאם בין עונות, `observed_sd` הפיזור בין קבוצות־עונות, `luck_sd` ממוצע סטיות התקן מאומדן המזל. `top3_gap`, `places4_8_gap`, `places9_11_gap` הם פערי ביתיות לפי מיקום רשמי; `club_mean_sd` פיזור ממוצעי מועדונים עם לפחות שש עונות, ו־`club_luck_sd` קירוב המזל לאותם ממוצעים לפי מספר עונותיהם. אומדן מזל משתמש בשונות נקודות מאוגמת ובמספר משחקי הבית והחוץ בפועל. |
| `style_diagnostics` | זוג: Series; טבלה ב־`metric` | קלט `tables`, `seasons`. Series: `attack_defence_corr` לאחר מרכוז עונתי, `gd_next_corr`, `ppm_next_corr`, `big_top3`, `big_bottom3`, `narrow_luck_corr`, `gd_next_r2`, `gd_narrow_next_r2`, `league_ppm`. הטבלה מציגה `league_mean`, `persistence`, `observed_sd`, `luck_sd`, `r_squared` עבור נקודות, הפרש שערים, ניצחונות, חלק ניצחונות גדולים, ניצחונות צרים, BTTS, over_2_5 ושאריות שני האחרונים אחרי OLS על שערי זכות וחובה. `luck_sd` הוא קירוב בינומי עם 36 משחקים קבועים, ורק למדדים ששמם מסתיים ב־`rate`; היתר NaN. |
| `promoted_rule_comparison` | זוג: פירוט ב־`season_id,team_id,rule`; סיכום ב־`period,rule` | קלט `tables`, `seasons=8–25`. עולות מעונות אחרי 8; שתי חלופות `median/q30` של נתחי השווי בחצי התחתון של t−1, מתוך ערכים זמינים. עונת הייחוס t−1 מותרת גם מחוץ לבחירה, אך רק בתוך development. פירוט `team_name`, `season`, `value_share_of_league`, `imputed_share`, `diff`, `ratio`. סיכום `n`, `fraction_below`, `median_diff_pp`, `mae_pp`, `ci_lower/upper` של Wilson לשיעור שמתחת לכלל; תקופות `all`, 9–17, 18–25, בחיתוך עם העונות שנבחרו. הפרשים ו־MAE בנקודות אחוז. |
| `promoted_performance` | `promoted_in` | קלט `tables`, `seasons=14–25`. השוואת עולה/ותיקה לפי `n` קבוצות־עונות, ממוצע `points_per_match`, ממוצע `goal_difference_per_match`, `std_ppm`, `relegations`, `best_final_position`. ביצועי הנקודות ספורטיביים; הירידה והמיקום רשמיים. אין דרישה ל־t−1 בתוך עידן 14, כי ההשוואה היא של עונה בודדת. |

`correlation_evidence` ו־`adjacent_pairs` אינם בודקים או מסננים עונות בעצמם; יש למסור frame שכבר סונן. מבחני המתאם מניחים שורות עצמאיות, אף שבפועל יש תלות בין קבוצות ועונות. ב־`era_persistence` הרווחים מדגימת עונות מטרה שלמות, זרע קבוע 20261014, בלי פרמטר seed; שיפוע אינו אחוז דיוק, והמיקומים הם רשמיים. `loso_predictions` מחזיקה עונה שלמה בחוץ אך מאמנת גם על עונות מאוחרות ממנה: זו **EDA בדיעבד, לא walk-forward ולא OOF כרונולוגי**. גם R², התאמות סגנון ובחירת קבוצות לפי מיקום הם תיאוריים בתוך המדגם; אין לפרש שאריות כזיהוי סיבתי של יכולת או מזל. בהשוואת כלל העולות אין נרמול לפי גודל הליגה ואין שינוי בכלל הייצור או בנתוני המקור; שווי t בפירוט הוא יעד בדיקה בדיעבד בלבד.

## EDA-02, פעימה B — הליגה לאורך השנים

תצוגות בזיכרון בלבד, באותו מודול league_stats ובאותו בסיס ספורטיבי של
פעימה A. כל פונקציה המקבלת AnalysisTables מסננת את seasons לעונות פיתוח
8–25 בלבד ודוחה עונות אחרות, כפילויות ורשימה ריקה. המקור הוא match_table,
team_matches ו-team_seasons; אין קריאה ישירה לטבלאות מחזור או סיום.
המיקומים ודגלי הירידה הרשמיים נשמרים כהקשר מהתצוגות הקיימות.
סיכומי סוף עונה אינם קלט לחיזוי באותה עונה.

| תצוגה | שורה / אינדקס | עמודות, בסיס והחרגות |
| --- | --- | --- |
| season_thresholds | עונה; season_id | champion/runner_up לפי final_position 1/2; safety הוא הניקוד הנמוך ביותר של שורדת, relegated_max הגבוה ביותר של יורדת. לכל יעד `_points`, `_ppm`, `_games`. title_margin_points/ppm הם אלופה פחות סגנית. n_teams, games_per_team (ממוצע), games_per_team_min/max, n_relegated ו-era |
| upper_line בתוך season_thresholds | סף מוקדם באותה עונה | צבירת points_earned ממשחקי stage=regular בלבד; הניקוד במקום k בסדר יורד. upper_sizes מועבר במפורש לפונקציה הטהורה, מתוך load_playoff_upper_sizes הקורא config/seasons.yaml: ללא סף בעונות 8–10, k=6 בעונות 11–12 ו-14+, k=8 בעונה 13. אין שוברי שוויון בחישוב הניקוד |
| threshold_ranges | יעד; objective | סיכום לעידן נבחר (ברירת מחדל 14_teams): n עונות עם סף זמין, points_min/median/max, ppm_mean, games_min/max. אינו הבטחה שניקוד מסוים מספיק להישרדות |
| critical_ties | עונה; season_id | title_tie בנקודות האלופה/סגנית הרשמיות; upper_line_tie בין k/k+1 לפי ניקוד השלב הסדיר; relegation_tie בין השורדת האחרונה והיורדת הראשונה לפי המיקום הרשמי. Boolean nullable; קו העליון חסר ללא פיצול |
| critical_ties_summary | גבול; boundary | n_seasons, n_applicable (15 לקו העליון בפיתוח המלא), n_ties. מקבל את טבלת השוויונות שכבר סוננה; אינו קובע איזה שובר שוויון הכריע |
| home_advantage_by_season | עונה; season_id | n משחקים, home_win_rate/draw_rate/away_win_rate עם Wilson; home_goals_pm, away_goals_pm ו-goals_pm_gap; home_ppm/away_ppm = 3×שיעור ניצחון + שיעור תיקו, ppm_gap |
| home_advantage_periods | תקופה; period | 2009/10–2014/15 (עונות 11–16) מול 2015/16–2023/24 (17–25): n_seasons, n_matches, mean_ppm_gap, mean_home_win_rate. ממוצע שווה־משקל של שיעורי העונות, לא איחוד משחקים. תקופה בלי עונות נבחרות נשארת עם n=0 וממוצעים חסרים; אלה נתונים תיאוריים, לא אומדן סיבתי |
| era_comparison | עידן; era | n_matches, n_seasons, goals_per_match משוקלל במשחקים, draw_rate, zero_zero_rate, home_win_rate, away_win_rate, big_margin_rate (הפרש מוחלט ≥3); std_ppm ו-title_margin_ppm הם ממוצעים שווי משקל של עונות, לא חציית נקודות |
| competitive_balance | עונה; season_id | std_ppm בין הקבוצות (סטיית תקן מדגמית), top_bottom_ppm_gap = שיעור הניקוד הגבוה פחות הנמוך; title_margin_ppm לפי האלופה/סגנית הרשמיות, ולכן עשוי להיות שלילי |
| champions_summary | מועדון; team_id | titles לפי מקום ראשון רשמי, team_name מהעונה האחרונה שנבחרה שבה המועדון זכה. סכום 18 בפיתוח המלא |
| calendar_effects | ממד×קטגוריה; dimension, level | weekday: Saturday/Sunday/Monday/other; kickoff_hour: before_18/18_to_20/20_plus; month: 1–12 כמחרוזת. תאריך מומר ל-Asia/Jerusalem, כולל תזוזת יום בחצות. n, n_seasons, goals_per_match ושיעורי תוצאות; תאריך חסר נשמר בקבוצת unknown בכל ממד, אינו מומצא או מושמט |
| postponed_summary | דגל דחייה; is_postponed | n, n_seasons, goals_per_match ושיעורי תוצאות מול שאר המשחקים. משתמש ב-is_postponed המתוקן: המחזור הבא באותו stage בלבד; 49 נדחים בפיתוח |

`flag_inverted` מסומן כש-relegated_max_points > safety_points; אין כישלון
בגלל סדר זה. גם upper_line_points יכול להיות נמוך מ-safety_points הסופי:
הם נמדדים בזמנים שונים. בפיתוח זה קורה בעונות 20 ו-25. בעונה 18 יורדת
צברה 38 נקודות על המגרש לעומת שורדת עם 32; ובעונה 11 האלופה הרשמית
צברה 84 לעומת 87 לסגנית, כאשר כאן לא חוצים את הנקודות ההיסטוריות.

לכל סף הניקוד נלקח מהקבוצה/ות המתאימות. אם כמה קבוצות באותו ניקוד על
הסף, `_games` הוא ממוצע מספר המשחקים שלהן ו-`_ppm` הוא הניקוד חלקי
ממוצע זה; אינו בחירה של שובר שוויון. title_margin_ppm הוא הפרש שני
השיעורים, ולא חלוקת הפרש הנקודות במספר משחקים משותף שרירותי.

כל שיעור בתצוגות B מקבל `<rate>_ci_lower/upper` של 95% Wilson, עם n
משחקים כמכנה. ממוצעי השערים ב-era_comparison, calendar_effects
ו-postponed_summary מקבלים goals_per_match_ci_lower/upper מ-bootstrap
של בלוקי עונה (סכומי שערים ומספר משחקים יחד, לשמירת המשקל במשחקים).
ברירת המחדל n_bootstrap=2000, זרע 20261014; הטסטים משתמשים ב-50.
פחות משתי עונות → רווח ממוצע חסר. אין תיקון להשוואות מרובות, ואין
פרשנות סיבתית ליום, שעה, עידן או דחייה.

season_summary הקיימת משמשת ישירות לתצוגת מגמות שערים, תיקו, 0:0,
שערי פנדל ואדומים; ערכים אלה אינם מחושבים מחדש בפעימה B. ניתוחי
תוצאות מלאות אינם מסננים ht_conflict או goals_available; אין כאן ניתוח
תזמון שערים או מחצית. אין שינוי ב-data/clean או בקונפיגורציית הירידה.

### B — אי־ודאות ובדיקות הסקירה (`history_review.py`)

כל שש הפונקציות מקבלות `tables`, `seasons=DEVELOPMENT_SEASONS` (8–25), עם אימות הבחירה המתואר לעיל. אין בהן פרמטר seed או n_bootstrap: השיטות המפורטות כאן אנליטיות. עידן 14 מסונן במפורש רק היכן שמצוין.

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `threshold_diagnostics` | `objective` | נדרש `upper_sizes` מפורש. מתוך `season_thresholds` משאירה רק `era=14_teams`. עבור `champion/upper_line/safety/relegated_max`: `n` עונות זמינות, `mean`, `std` מדגמית של נקודות למשחק, `slope` לשנה ו־`p` של מגמה ליניארית לפי מזהה העונה; המגמה חסרה כשאין יותר משתי עונות. |
| `home_season_uncertainty` | `season_id` | טבלת `home_advantage_by_season` בתוספת `ppm_gap_se`, `ppm_gap_ci_lower/upper`: שגיאת תקן של ‎+3/0/−3 לכל משחק ורווח נורמלי 95% לפער הביתיות. זו אי־ודאות דגימת משחקים בתוך עונה, בהנחת עצמאות. |
| `era_uncertainty` | `era` | סיכום שיעורי משחק עם Wilson, `n_matches`, `n_seasons`, `goals_per_match`, ממוצעי `std_ppm` ו־`title_margin_ppm`. `goals_per_match_ci_lower/upper` הם רווח t לפי משחקים; `std_ppm_ci_lower/upper` הם רווח t לפי מספר העונות לפיזור הממוצע, עם קצה תחתון לא שלילי ו־NaN בפחות משתי עונות. השערים משוקללים במשחקים; מדדי העונות שווי משקל. |
| `seasonal_trend_tests` | `era,metric` | מתוך `season_summary`: `all` ו־`14_teams`, כל אחד עם draw_rate, zero_zero_rate, penalties_per_match, red_cards_per_match. פלט `n_seasons`, `slope`, `p`, `p_bonferroni=min(1,8p)`; חסרים מושמטים לכל מדד, ופחות משלוש עונות נותנות מגמה חסרה. |
| `balance_decomposition` | `season_id` | `competitive_balance` בתוספת `luck_std`, `latent_std`, `luck_fraction`, `n_teams`. שונות המזל היא ממוצע בין קבוצות של שונות נקודות למשחק המאוגמת מכל שורות קבוצה×משחק בעונה, חלקי מספר משחקי הקבוצה; `latent_std=sqrt(max(0,std_ppm²−luck_variance))`, ו־`luck_fraction=luck_variance/std_ppm²`. |
| `champion_previous_rank` | `season_id` | עבור האלופה הרשמית בכל עונת מטרה: `team_name`, `previous_rank` לפי **נקודות למשחק ספורטיביות** ב־t−1, `previous_available`, `best_previous`, `top3_previous`, `n_teams` בעונת המטרה. t−1 נלקחת רק מ־development, גם מעידן ליגה גדול יותר או מחוץ לבחירה; ללא עונת עבר כלל אין שורת מטרה, ובהיעדר הקבוצה בעבר הדירוג חסר. שוויון בעבר מקבל את הדירוג המינימלי (`method="min"`). |

ספי האליפות והירידה וזהות האלופה הם רשמיים, אך הנקודות על המגרש; ענישה יכולה לשנות מי הקבוצה שנבחרה ואף את סימן פער האלופה–סגנית. רווחי `era_uncertainty` שונים מרווחי bootstrap העונות ב־`era_comparison` ואינם תחליף להם. מגמות בזמן הן OLS תיאורי; שמונת מבחני המגמה מקבלים משפחה אחת של Bonferroni, אך אין מכך ראיה סיבתית או אישור לפיצ׳ר. פירוק המזל מניח שונות נקודות משותפת ומשחקים עצמאיים, ללא נטרול יריבות, לוח או בחירת קבוצות; השונות שנותרת אינה מדידה סיבתית של יכולת.

### B — תנאי משחק ו־Elo של EDA (`conditions.py`)

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `load_match_conditions` | ללא; dict | קלט `path: Path` לקובץ YAML (במחברת `config/match_conditions.yaml`), ללא נתיב ברירת מחדל. מחזירה את התצורה, וממירה start/end לתאריך מקומי. בודקת timezone=Asia/Jerusalem, מקור ורמת אמון בכל רשומה, עונות development כאשר נמסרות, טווחי תאריכים בתוך 1.7.2006–30.6.2024, סוג GHOST/LIMITED ותקופות קהל ללא חפיפה. |
| `classify_conditions` | אינדקס המשחקים שנבחרו | קלט `matches`, `config`, `seasons=8–25`; נדרשים תאריך ואצטדיון ידועים. מחזירה עותק מסונן של המשחקים עם `condition`. אצטדיון בית אופייני נקבע כשכיח בכל עונה×מארחת; בשוויון נבחר הראשון לפי `mode()`. ההיררכיה המעשית: GHOST/LIMITED לפי תאריך מקומי גוברים על SHARED, שגובר על RELOCATED, ואז OTHER/NORMAL. |
| `prematch_elo` | אינדקס המשחקים שנבחרו, בסדר הקלט | קלט `matches`, `seasons=8–25`, תאריכי פתיחה תקינים. מחזירה את עמודות המשחק עם `elo_home`, `elo_away`, `elo_gap` **לפני העדכון**. K=20; יתרון בית 50 בחישוב ציפיית העדכון בלבד, לא בתוספת לפלט elo_gap; סולם הציפייה 400. מועדון שנראה לראשונה מתחיל ב־1450; מועדון שכבר נראה נסוג שליש לכיוון 1500 במשחקו הראשון בעונה חדשה. מכפיל הפרש מוחלט GD: ‏1 ל־0/1, ‏1.5 ל־2, ‏(11+GD)/8 ל־3 ומעלה. תוצאות 1/0.5/0, ועדכוני משחקים באותו מועד מוחלים רק אחרי שנקראו כל דירוגי המצב הקודם של אותה אצווה. אין פרמטרי כיול בחתימה. |
| `elo_residuals` | אינדקס המשחקים שנבחרו | קלט `matches`, `seasons=8–25`. מפעילה prematch_elo ומוסיפה `points_gap=3×sign(home_goals−away_goals)` ו־`residual`, לאחר OLS של הפער על חותך, x ו־x³, כאשר x=elo_gap/100. הרגרסיה מותאמת לכל המדגם שנבחר; אין fit/predict כרונולוגי או החזרת מקדמים. |
| `stratified_effect` | ללא; dict | קלט `groups`: זוג מערכי שאריות `(condition,NORMAL)` לכל **עונה אחת**, לפחות שני משחקים בכל צד בכל שכבה. `n_permutations=10000`, `n_bootstrap=5000`, `seed=20261028`. פלט `effect`, `p`, `ci_lower`, `ci_upper`: הפרש ממוצעים בתוך עונות, עם משקל קבוע לפי מספר משחקי התנאי בכל עונה. p דו־צדדי עם plus-one מתמורות בתוך עונה; רווח 95% BCa מדגימה נפרדת של שני הצדדים בכל עונה. |
| `condition_comparison` | זוג: סיכום ב־`condition`; משחקים באינדקס המקורי המסונן | קלט `matches`, `config`, `seasons=8–25`, `n_permutations=10000`, `n_bootstrap=5000`. מסווגת ומנטרלת Elo. שש שורות NORMAL/GHOST/LIMITED/SHARED/RELOCATED/OTHER: `n`, `raw_gap`; בארבע ההשוואות ל־NORMAL גם `n_seasons`, `normal_n`, `overlap_n`, `effect`, `p`, `ci_lower/upper`, `p_bonferroni`, `p_bh` כשיש שכבות השוואה. משפחת התיקון נשארת בגודל ארבע גם אם מבחן חסר. זרע קבוע 20261028 דרך stratified_effect, ללא פרמטר seed נפרד. הטבלה השנייה כוללת condition, דירוגי Elo, פער ושאריות. |
| `home_period_comparison` | RangeIndex | קלט `matches`, `seasons=8–25`, `normal_only=False`; אם True נדרשת עמודת condition וסינון NORMAL. מחזירה שתי תקופות 11–16 ו־17–25: `start_season`, `end_season`, `n_seasons`, `mean_gap`, `se`, `std`. הממוצע שווה משקל לעונות, לאחר ממוצע ‎+3/0/−3 במשחקי כל עונה. |

**Elo של EDA בפרק B בלבד. אינו Elo הייצור של M0. הפרמטרים הם ערכי ספרות שלא כוילו (ה-ס15). אתחול העולות (1450 / דירוג ישן) פתוח למחברת 03.** מועדון שחוזר לאחר היעדרות משתמש בדירוג שנשמר לו, עם נסיגה אחת במשחק הראשון בעונת חזרתו; אין נסיגה חוזרת לכל שנת היעדרות. הקלט מסונן לעונות שנמסרו, ולכן אין היסטוריית Elo אוטומטית לפני תחילת הבחירה.

סיווג האצטדיון השכיח משתמש בעונה שלמה, והגדרות התנאים הן **בדיעבד** למחקר בלבד. SHARED נבדק ובוטל כמועמד; סיווגו הקיים אינו פיצ׳ר prematch. התאמת השאריות נעשית בתוך המדגם; מבחן התנאים אינו אומדן סיבתי של קהל או אצטדיון. האומדן, התמורות וה־BCa מתייחסים לאותו הפרש משוקלל בתוך עונות; `raw_gap` הוא תיאור גולמי של התנאי, ואינו אפקט מתוקנן. SciPy מחשבת את תאוצת BCa בעזרת jackknife נפרד לכל מדגם ושכבה, עם התאמת גודל המדגם; העונות עצמן וההתאמה ל־Elo נחשבות קבועות. אין כאן bootstrap של עונות או אי־ודאות מלאה של מודל ה־Elo. חיתוך תקופות הביתיות ב־2015/2016 נבחר בדיעבד.

## EDA-02, פעימה C — מהלך העונה והטבלה

מודול league_stats פוצל לחבילה לפי פרקים, עם אותו API ציבורי:
`from ligat.analysis.league_stats import ...`. core.py שומר את חישובי A/B
ללא שינוי; season_course.py מוסיף את C. התצוגות טהורות ובזיכרון, development
8–25 בלבד, בלי קריאת טבלאות רשמיות נוספת ובלי כתיבה ל-clean/raw.
המיקום, שלב הסיום והירידה נלקחים מהתוויות הקיימות ב-team_seasons.

| תצוגה | שורה / אינדקס | בסיס, עמודות והחרגות |
| --- | --- | --- |
| standings_by_match_number | עונה×k×קבוצה; season_id, k, team_id | stage=regular כברירת מחדל; אפשר upper/middle/lower או all לכל העונה. מסדרים כל קבוצה לפי date ואז game_id, צוברים points (3/1/0), goal_difference, goals_scored. games_played הוא מספר משחקיה שנצברו עד k. קבוצה עם פחות מ-k נשארת בערכים האחרונים עם games_short=True. rank יורד בנקודות → הפרש → שערי זכות, ושוויון מלא מקבל ממוצע המקומות. מזהי האינדקס Int64; אין שוברי שוויון אחרים, עונשים או חצייה |
| table_settling | עידן×k; era, k | spearman_mean בין rank ל-final_position הרשמי, ממוצע בעונות עם מתאם מוגדר, spearman_n ו-n_seasons. leader_won_title_rate = ממוצע קרדיט אליפות (מספר אלופות בין המובילות / מספר המובילות); bottom_relegated_rate = ממוצע קרדיט ירידה (מספר יורדות בין האחרונות / מספר האחרונות). כל העונות במכנה. n_tied_leader/n_tied_bottom מדווחים עונות עם שוויון. לכל מדד n, credit_sum ו-ci_method; count נשאר כינוי לסכום קרדיטים ועלול להיות לא שלם. leader_won_title_excl_ties / bottom_relegated_excl_ties הם רגישות בלבד, עם n ורווח Wilson. מתאם בטבלה שכולה שווה חסר |
| settling_sensitivity | מדד; metric | leader_won_title/bottom_relegated: n_points (צמדי era×k), n_comparable לגרסת ללא שוויונות, mean_abs_diff ו-max_abs_diff בין הגרסאות; פערים בטווח 0–1, למשל 0.0227 = כ־2.3 נקודות אחוז. סיכום תיאורי; צמדי k אינם דוגמאות עצמאיות |
| in_upper_line_stayed_rate בתוך table_settling | ממוצע עונות עם פיצול | הקבוצות שמעל הקו מקבלות משקל 1; קבוצת שוויון על הקו מקבלת לכל חברה משקל מספר המקומות הפנויים / מספר החברות. סכום משקל × I(final_stage=upper), חלקי גודל העליון, ואז ממוצע שווה־משקל של העונות. n כאן עונות, credit_sum סכום שיעורי העונות, team_weight סכום המקומות ו-weighted_team_count סכום הצלחות משוקללות. tied_seasons מדווח שוויונות על הקו; אין מחיקת עונה בשוויון. NA ללא פיצול; upper_sizes כולל 8 בעונה 13 |
| split_movement | עונה עם פיצול×קבוצה; season_id, team_id | סוף regular לפי k המרבי שנצפה; rank_at_split, final_position, move=rank_at_split−final_position, group (upper/middle/lower מה-final_stage), team_name, games_short, points_halved (הקשר היסטורי בעונות 11–12; אינו משנה את הניקוד). move חיובי הוא שיפור, ויכול להיות חצי מקום בעקבות שוויון ממוצע |
| split_summary | קבוצת פלייאוף; group | n קבוצות־עונות, move_mean/median/min/max ו-improved/unchanged/declined. leader_at_split_won_title_rate לפי קרדיט חלקי, leader_n כל העונות, leader_credit_sum סכום הקרדיטים (leader_won_count כינוי לאותו סכום), leader_tied_seasons. leader_at_split_won_title_excl_ties ו-n לרגישות. n_seasons/n_halved_seasons להקשר; השדות העונתיים חוזרים בכל קבוצת תצוגה |
| stage_comparison | שלב; stage | n_matches, n_seasons, goals_per_match עם רווח bootstrap של בלוקי עונה; draw/zero_zero/one_one/btts/home_win/away_win/big_margin_rate עם Wilson. one_one=תוצאה 1:1, btts=שער לשני הצדדים, big_margin=הפרש מוחלט ≥3. כל משחק מלא נכלל פעם אחת בלבד |
| playoff_form | קבוצה×עונה עם פלייאוף; season_id, team_id | ppm_regular ו-ppm_playoff (כל הפלייאוף של הקבוצה), n_regular/n_playoff, ppm_change=פלייאוף−סדיר, team_name ו-final_stage. אין חציית נקודות או השלמה. קבוצה ללא פלייאוף אינה שורה בתצוגה; שינוי יריבות הוא הסתייגות להשוואת הקצב |
| playoff_form_summary | קבוצת פלייאוף; final_stage | n קבוצות־עונות, n_seasons, mean_ppm_change, median_ppm_change ורווח 95% mean_ppm_change_ci_lower/upper. דגימת בלוקים של עונות שלמות שומרת את הקבוצות יחד; הממוצע משוקלל לפי מספר קבוצות־העונות, לא ממוצע עונות שווה. בפחות משתי עונות הרווח חסר. קשר תיאורי, לא סיבה |
| final_rounds_check | שלב×תקופה; stage, period | last/rest לפי שלושת מספרי המחזור האחרונים (last_n=3) בכל עונה ובאותו stage. n, שיעורי תוצאות ו-goals_per_match עם אותם רווחים; upset_count, upset_n ו-upset_rate עם Wilson. אין קבוצת rest באמצעי: כל 12 משחקיו בשלושת המחזורים האחרונים. תקופה שאין בה משחקים אינה מופיעה |
| season_points_race | עונה×k×קבוצה; season_id, k, team_id | standings(stage=all) יחד עם team_name ו-regular_games. נקודות כל העונה ללא חצייה/עונשים. גרף לעונה אחת, עם קו פיצול ב-regular_games המרבי; games_short מוחרג מציור הקווים כדי לא להציג משחקים מדומים לאחר השלמת 33/36 משחקים |

טבלה אחרי k היא תיאור לפי **מספר משחקים לכל קבוצה**, לא לפי תאריך
משותף. הקבוצות עשויות להגיע ל-k בימים שונים עקב דחיות. אין להשתמש בדירוג
הזה כקלט prematch; שחזור טבלה as_of לפי תאריך נפרד במחברת 03.

במדד upset של final_rounds_check, לכל צד סופרים רק את משחקיו בעונה
שתאריכם **קודם ממש** לתאריך המשחק. מתוך standings(stage=all) נשלפות
נקודות הקבוצה עצמה אחרי k הקודם, חלקי k; אין שימוש בדירוג שאר הקבוצות
שעלול לכלול משחקים מתאריך מאוחר יותר. k=0 נשאר חסר. משחק נכלל ב-upset_n
רק כשהיסטוריית שני הצדדים קיימת וה-ppm שונה; ניצחון של הצד בעל ה-ppm
הנמוך הוא upset. תיקו עם ppm שונה הוא אי־הפתעה; ppm שווה/חסר מוחרג.
זהו סינון תיאורי פשוט, לא זיהוי של קבוצה חסרת עניין מתמטית.

כל פונקציה ציבורית של season_course מקבלת seasons=DEVELOPMENT_SEASONS ודוחה עונה
אחרת, רשימה ריקה וכפילות. פונקציות המשתמשות ב-bootstrap מקבלות
n_bootstrap=2000 (50 בטסטים): table_settling, settling_sensitivity,
split_summary, stage_comparison, playoff_form_summary, final_rounds_check. לרווחי ממוצע קרדיט
עם ערכים חלקיים משתמשים ב-bootstrap של עונות; כשהקרדיטים כולם 0/1
משתמשים ב-Wilson. ci_method מפרט את השיטה (bootstrap_seasons/wilson),
ו-unavailable כשאין עונות בפורמט. פחות משתי עונות נותנות רווח bootstrap
חסר. לגרסת הרגישות הבינרית ללא שוויונות משתמשים ב-Wilson תמיד. רווחי המתאם מדגימת עונות; ממוצעי השערים מבלוקי
עונה עם סכומי שערים ומספר משחקים יחד. אלה רווחים תיאוריים, ללא תיקון
להשוואות מרובות וללא ביטול תלות בין קבוצות/עונות. בעידני 12/16 רק 3
עונות; בפלייאוף האמצעי רק 12 משחקים, ולכן אין ממנו מסקנה חזקה.

11 המשחקים הטכניים כבר הוחרגו; הם יכולים לגרום ל-games_short. אין
ב-C ניתוח מחצית או דקות שערים, ולכן ht_conflict ו-goals_available אינם
מסננים תוצאות משחק מלא. תזוזה ממיקום ספורטיבי למיקום רשמי כוללת גם
השפעת חציית נקודות ועונשים; אין לשייך את כל התזוזה לשינוי בכוח הקבוצה.
מחקר מוטיבציה מלא לפי טבלה as_of נדחה למחברת 04, VISION §7 שאלה 6א.

### C — סטיות בפלייאוף ותיקו לפי מרחק (`season_course.py`)

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `playoff_deviations` | `season_id,team_id` | קלט `tables`, `seasons=8–25`. כל עמודות `playoff_form`, בתוספת `regular_deviation` ו־`playoff_deviation`: נקודות למשחק בסדיר/פלייאוף פחות ממוצע **אותן קבוצות** בעונה ובקבוצת הפלייאוף שלהן. גם האמצעי נשמר כשהוא קיים. |
| `playoff_persistence` | `final_stage` | קלט `tables`, `seasons=8–25`, `n_bootstrap=2000` חיובי. רגרסיה של סטיית הפלייאוף על סטיית הסדיר: `n`, `n_seasons`, `pearson`, `spearman`, `slope`, `intercept`, ו־`<metric>_ci_lower/upper` למתאמים ולשיפוע. bootstrap של עונות שלמות עם זרע קבוע 20261014; ללא פרמטר seed. |
| `draw_rates_by_distance` | `stage,distance_bin` | קלט `tables`, `seasons=8–25`. distance הוא מספר המחזור האחרון שנצפה בעונה×שלב פחות מחזור המשחק. דליים מסודרים 0–2, 3–5, 6–8, 9–11, 12–14, 15+; פלט `n`, `draw_count`, `n_seasons`, `draw_rate`, `draw_rate_ci_lower/upper` של Wilson. שורות רק לדליים שנצפו; distance_bin הוא categorical מסודר. |

מרכוז הפלייאוף משווה כוח יחסי בתוך הקבוצה שנבחרה אליו, אך אינו מבטל את הבחירה בדיעבד, את הקיצור בתחתון או את שינוי היריבות. רווחי קשר חסרים כשאין מספיק עונות/שונות לפי החוזה של חישוב הקשרים; מדגם האמצעי קטן במיוחד. מרחק מסוף שלב כאן הוא **מחזורים ממוספרים לפי הסיום שנצפה**, לא ימים או מספר משחקים שנותרו לקבוצה בזמן אמת. לשימוש ב־M0 צריך לחשבו מלוח שהיה ידוע ב־as_of, ולא להעתיק את התצוגה הזאת.

### C — בדיקות הסקירה (`course_review.py`)

ברירת המחדל של פונקציות המקבלות `tables` היא `seasons=ERA14_SEASONS` (14–25); אפשר למסור בחירת development אחרת, בכפוף להגבלות המבניות המתוארות בכל רשומה. כל בחירה נבדקת כלא ריקה, ללא כפילות ומתוך 8–25. דירוגי הסדיר ספורטיביים, ואילו מיקום הסיום, זהות האלופה והירידה הם תוויות רשמיות; זה אינו שחזור טבלה לפי תאריך as_of.

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `playoff_noise_diagnostics` | `final_stage=upper/lower` | קלט `tables`, `seasons`. פלט `regular_sd` — ממוצע סטיות תקן עונתיות של סטיות הסדיר בתוך קבוצת הפלייאוף; `noise_sd` — שורש ממוצע שונות נקודות הסדיר המאוגמת בעונה חלקי n_regular לכל קבוצה; `n` קבוצות־עונות. אין רווח או bootstrap ואין תיקון לבחירה לפלייאוף. |
| `random_mean_movement` | ללא; float | קלט `n` שלם חיובי (לא bool). מחזירה `(n²−1)/(3n)`, ממוצע תזוזה מוחלטת תחת סדר אקראי אחיד של n קבוצות. אין נתוני משחקים, סינון עונות או seed: זה חישוב סגור. |
| `position_credits` | RangeIndex, שורות מורחבות | קלט frame שטוח עם `season_id`, `k`, ושורת קבוצה אחת בכל צילום; `rank="rank"` הוא שם עמודת דירוג ממוצע. בודקת שמזהי העונות ב־development. מחזירה את כל העמודות בתוספת `position`, `weight`: קבוצת שוויון בגודל m מקבלת 1/m לכל אחד מכל m המקומות שהיא תופסת. אין הסתפקות ב־floor/ceil; תחילת רצועה לא שלמה גורמת ל־ValueError. |
| `rank_snapshots` | RangeIndex; `season_id,k,team_id` כעמודות | קלט `tables`, `seasons`. צילומי `standings_by_match_number` בסדיר, כולל נקודות, שערים, משחקים ודירוג, עם `team_name`, `final_position`, `final_stage`, `relegated` מ־team_seasons. כולל games_short; k הוא מספר משחקים של כל קבוצה, לא תאריך משותף. |
| `rank_heatmap` | זוג: מטריצה `position × final_position`; פירוט ב־RangeIndex | קלט `tables`, `seasons`, `k=10`, `at_split=False`. דורשת 14 קבוצות בכל עונה ושנקודת הבדיקה קיימת בכולן; at_split=True בוחר k מרבי נצפה בכל עונה. מטריצת **ספירות משוקללות 14×14**, ללא נרמול להסתברויות; השוויונות מורחבים באמצעות position_credits. הפירוט כולל את השורות והמשקלים שבנו אותה. |
| `region_persistence` | `region,k` | קלט `tables`, `seasons`; דורשת 14 קבוצות. אזורים top=1–3, upper_line=4–6, middle_table=7–10, bottom=11–14; k≤26. פלט `n` כסכום משקלי המקומות, `retained`, `movement`, `n_seasons`, `same_region_rate=retained/n`, `mean_movement=movement/n`. שיוך לאזור מתפצל בשוויון; התזוזה נמדדת מהדירוג הממוצע המקורי אל המיקום הרשמי. |
| `settling_review` | `era,k` | קלט `tables`, `seasons`, ו־`upper_sizes` מפורש. מחזירה את מלוא סכמת `table_settling` הקיימת, עם `min_bootstrap_seasons=5`, n_bootstrap=2000 וזרע פנימי 20261014. אין פרמטרי resampling בחתימה הזאת; רווחים המבוססים על bootstrap עונות אינם מוצגים בפחות מחמש עונות. |
| `leader_benchmarks` | `benchmark,k` | קלט `tables`, `seasons`. k=10/20, benchmark=leader/bottom; `n_seasons`, `success` כסכום קרדיטים עונתיים בשוויון, `rate`, `chance_rate` קבוע 1/14 לאלופה או 2/14 ליורדת. משקל שווה לעונה, אין רווחי סמך. קווי המקרה מיועדים ל־14 קבוצות ואינם מותאמים אוטומטית לבחירה בגודל ליגה אחר. |
| `split_diagnostics` | זוג: סיכום ב־`group`; עובדות ב־`benchmark` | קלט `tables`, `seasons`, `upper_sizes` מפורש. סיכום upper/lower: `n`, `unchanged_rate`, `mean_movement`, `random_movement` לפי גודל כל פלייאוף, `spearman_mean` בין דירוג הפיצול לסיום. עובדות: leader_title, last_two_relegated, places_11_12_relegated, עם `success`, `n`, `rate` מקרדיטי שוויון בסוף הסדיר. הגבולות 13–14 ו־11–12 מיועדים לעידן 14. |
| `regular_strength_expectation` | אינדקס המשחקים | קלט `tables`, `seasons`. כל עמודות המשחקים שנבחרו עם `home_gf`, `home_ga`, `away_gf`, `away_ga` מממוצעי הסדיר המלא לכל קבוצה, ו־`expected_goals=(home_gf×away_ga+away_gf×home_ga)/league_mean`. league_mean הוא ממוצע שערי זכות בשורות קבוצה×משחק בסדיר; חייב להיות חיובי. הצפי מתייחס לסך שערי המשחק, ללא רכיב ביתיות נפרד. |
| `stage_strength_comparison` | זוג: `stage`; `stage,contrast` | קלט `tables`, `seasons`. טבלת regular/regular_late/upper/lower: `n_matches`, `n_seasons`, `goals_per_match`, שיעורי תוצאות עם Wilson, `expected_goals`, `btts_rate` עם Wilson; regular_late הוא מחזורים 20–26 בסדיר. `goals_per_match_ci_lower/upper` מדגימת ממוצעי עונות כשיש לפחות חמש עונות, 2000 דגימות וזרע 20261014. טבלת effects: `mean`, `se`, `positive`, `n_seasons`, עם excess_strength=נצפה−צפוי או minus_regular לכל פלייאוף, ו־minus_early=מחזורים 20–26 פחות עד 19. |
| `draw_strata` | `season_id,stage` | קלט `tables`, `seasons`. לכל שכבת **עונה×שלב** regular/upper/lower: תאי 2×2, ‏a=תיקו בשלושת המחזורים האחרונים, b=יתר התוצאות שם, c=תיקו בשאר, d=יתר התוצאות בשאר. סוף השלב הוא המחזור המרבי שנצפה; שכבה בלי משחקי last או rest אינה נכללת. האמצעי מוחרג. |
| `mantel_haenszel` | ללא; dict | קלט `cells` עם עמודות a,b,c,d ואינדקס season_id,stage; בודקת שעונות האינדקס ב־development. `odds_ratio`, `ci_lower/upper` של יחס הסיכויים המאוגם, `p` למבחן יחס סיכויים 1 ללא תיקון רציפות, `p_bonferroni_12=min(1,12p)`, `last_n`, `rest_n`, `last_draw_rate`, `rest_draw_rate`, `n_strata`. הכיוון הוא סיכוי לתיקו בסוף שלב מול שאר השלב, בריבוד עונה×שלב. |
| `combined_draw_test` | `stage=all/regular/upper/lower` | קלט `tables`, `seasons`. בונה draw_strata ומפעילה mantel_haenszel על כל השכבות יחד ובנפרד לכל שלב זמין. עמודות זהות למילון של המבחן. attrs: `same_direction_seasons` — מספר עונות שבהן השיעור המאוגם בסוף נמוך מבשאר, ו־`n_seasons`; אלה תיאורים לפי עונה, לא מבחן נוסף. |

בקבוצת שוויון מלאה המשקל הכולל הוא יחידה לקבוצה ויחידה למקום בכל עונה. מפות החום מתארות מעבר מדירוג ספורטיבי למיקום רשמי; ענישה ושוברי שוויון רשמיים יכולים להשפיע על הפער. גבולות האזורים והעובדות על סוף הסדיר הוגדרו לעידן 14. אקראיות מלאה של סדר הקבוצות היא קו השוואה מתמטי, לא מודל כדורגל.

צפי הכוח של הסדיר משתמש **בסדיר המלא**, לרבות משחקים מאוחרים מהמשחק הנבדק; בסדיר עצמו ההתאמה בתוך המדגם. הוא מיועד לנטרול תיאורי של יריבות במחברת 02, ואינו פיצ׳ר prematch או תחזית M2. בטבלת השלבים, מרכז goals_per_match משוקלל במשחקים, בעוד שרווח ה־bootstrap מחושב מממוצעי עונות שווי משקל; אין לבלבל בין שני בסיסי המשקל כשמספר המשחקים שונה בין עונות. ההפרשים ב־effects מחושבים ברמת עונה; se הוא שגיאת תקן בין עונות ולא סטיית תקן של משחק בודד.

**מבחן Mantel–Haenszel נבחר אחרי שראינו את הנתונים.** גם combined_draw_test וגם התצוגה של mantel_haenszel הם בדיקה תיאורית בדיעבד, ללא bootstrap או seed. הריבוד בעונה×שלב אינו מוכיח שהסיבה היא מוטיבציה ואינו מתקן אוטומטית את חיפוש המבחן. השדה p_bonferroni_12 הוא פלט של משפחה בגודל 12 בקוד, ואינו הופך את המבחן למאשר שנקבע מראש. ברירת המחדל היא עידן 14; השוואת כל הפיתוח דורשת seasons מפורש, והיא מדגם שונה.

## מחברת 02 — פעימה D: המשחק והשערים

תצוגות תיאוריות בזיכרון ב־`ligat.analysis.league_stats.match_flow`, ללא כתיבה ל-clean.
כל הפונקציות מקבלות `seasons=DEVELOPMENT_SEASONS` ומסרבות לעונות מחוץ ל־8–25,
לרשימה ריקה או כפולה. אינדקסים נוצרים עם `drop=True`.

| פונקציה / טבלה | אינדקס | עמודות ומשמעות |
| --- | --- | --- |
| scoreline_distribution | home_goals; עמודות away_goals | קלט tables ו־seasons=8–25; tail_start=5 כברירת מחדל וחייב להיות חיובי. מטריצת שיעורים **6×6** עם תוויות 0,1,2,3,4,"5+". כל השערים מ־5 ומעלה מאוגדים בתא הזנב; כל המשחקים נשמרים, ללא חיתוך וללא נרמול מחדש. attrs: n_matches, n_tail (משחקים שבהם לפחות צד אחד בזנב), counts, ci_lower/upper לכל תא בשיטת Wilson. שינוי tail_start משנה את גודל המטריצה ל־tail_start+1 בכל ציר |
| top_scorelines | home_goals, away_goals | 10 התוצאות השכיחות בפועל; n הוא מספר הופעות, rate ו-ci_lower/upper מתוך כל המשחקים |
| poisson_reference | home_goals; עמודות away_goals | קלט tables, seasons=8–25, tail_start=5 חיובי. מכפלת Poisson עצמאיים עם ממוצעי שערי הבית והחוץ במדגם שנבחר; מטריצת **6×6** עם "5+", כאשר כל שולי הזנב נכללים דרך ‎1−P(0..4). סכומה 1 ללא נרמול מחדש. attrs: lambda_home/away, tail_mass (המסה שמחוץ לריבוע התוצאות המדויקות 0..4, וכבר כלולה במפה), n_matches. קו בסיס תיאורי בתוך המדגם, ללא התאמה לכוח קבוצות |
| observed_vs_poisson | scoreline | קלט tables, seasons=8–25. שורות 0:0, 1:0, 0:1, 1:1, 2:2, all_draws; n משחקים, count, observed ורווח Wilson, expected, ratio=נצפה/צפוי. תוצאות נמוכות נלקחות ממטריצת ה־6×6. **all_draws הוא שוויון שערים מדויק בכל הזנב**, עד דיוק נומרי; אינו סכום אלכסון מטריצת הקטגוריות, כי "5+:5+" כולל גם תוצאות שאינן תיקו |
| goal_count_dispersion | metric: home_goals/away_goals/total_goals | n משחקים, n_seasons, mean, variance מדגמית (ddof=1), dispersion=variance/mean, home_away_correlation פירסון. לכל מדד ci_lower/upper מ-bootstrap של בלוקי עונות תוך שמירת צמדי תוצאות. יחס עם ממוצע אפס או מתאם בשונות אפס חסר |
| draw_types | season_id | n כל משחקי העונה; zero_zero/one_one/two_two_plus: n, rate ורווחי Wilson. כל השיעורים מתוך המשחקים, לא מתוך התיקו |
| timeline_exclusions | reason | n_matches לפי סיבות זרות: ht_conflict; no_goals; unreliable_scorers_864982. הסיבה המפורשת של 864982 קודמת ל-no_goals, כדי לא לספור פעמיים |
| game_timelines | game_id, team_id | season_id, is_home, minutes_leading/level/trailing (סכום 90), led_at_any_point, trailed_at_any_point, points_earned, points_dropped_from_lead, points_won_from_behind, equalized_within_10, late_goal_for/against (ספירת שערים), first_half_stoppage_goals |
| timeline_team_summary | season_id, team_id | team_name, n משחקים, ppm, ממוצעי דקות בשלושת המצבים; סך נקודות שאבדו מיתרון והושגו מפיגור; n_led/n_trailed; equalized_within_10_n הוא מספר המשחקים עם ספיגה שהשאירה בפיגור, ושיעור הצלחה עם רווח Wilson. סכומים וממוצעים של העונה הם תיאור מדויק, ללא רווח תחזית |
| timeline_resilience | team_id | מועדונים עם min_seasons=5 לפחות; השם האחרון, n משחקים, n_seasons, n_led/n_trailed, סך נקודות שאבדו מיתרון והושגו מפיגור; points_won_per_trailed_game=השגה/n_trailed, points_dropped_per_led_game=איבוד/n_led, עם ci_lower/upper מדגימת עונות לכל יחס. אפס משחקי חשיפה נותן NA. נרמול לחשיפה אינו תקנון לחוזק או הוכחת סיבתיות |
| halftime_transitions | is_home, ht_state | n שורות קבוצה×משחק במצב leading/level/trailing; win/draw/loss_n/rate ו-ci_lower/upper; ppm ורווח bootstrap עונות. attrs.n_excluded_rows מדווח גם חוסר ht_state וגם החרגות ציר זמן |
| zero_zero_at_60 | outcome | zero_zero/home_win/away_win/scoring_draw; n משחקים ללא שער עד דקה 60 כולל, count, rate ורווח Wilson. scoring_draw משלים את החלוקה (67 משחקים אמיתיים) |
| first_goal_effect | is_home, scored_first | n, שיעורי W/D/L ורווחי Wilson, ppm ורווח bootstrap. 0:0 וזהות ראשונה עמומה (שני צדדים באותה דקה ראשונה) מוחרגים; attrs.n_zero_zero_or_ambiguous_rows מציין את מספר השורות |
| first_goal_minute_distribution | minute_bucket | n משחקים תקינים, count משחקים עם שער ראשון בדלי, rate ורווח Wilson; no_goal מציין 0:0. השיעורים כולל no_goal מסתכמים ל־1 |
| scoring_rate_by_state | is_home, state | state=leading/level/trailing; n_goals עם score_before ידוע, minutes חשיפה מ-game_timelines, n_matches, n_seasons, goals_per_90=90×שערים/דקות, goals_per_90_ci_lower/upper מ-bootstrap של בלוקי עונות. בית וחוץ נפרדים; אפס חשיפה נותן NA |
| goal_timing | minute_bucket, score_before | מצב המבקיעה leading/level/trailing לפני השער בסדר half/minute/goal_number; n כל השערים בדלי, count במצב, rate ורווח Wilson. זה חלק מהשערים, לא קצב לדקת חשיפה |
| team_late_goals | team_id | מועדונים עם min_seasons=5; השם האחרון, n_seasons, n_matches, n_goals; second_half_goal_share וחלק שערי מחצית שנייה מדקה 76 (late_goal_share), ספירת שערים ורווחי Wilson. מכנה שני המדדים הוא כל שערי הקבוצה התקינים |
| penalty_owngoal_stats | season_id, team_id | n משחקים עם מבקיעים זמינים, n_goals לזכות; penalty/own_goal_for/against_n ו-per_match; penalty_goal_share ורווח Wilson. שער עצמי לטובת הקבוצה הנהנית, לא קבוצת השחקן שהבקיע עצמי. סתירות מחצית אינן מוחרגות |
| penalty_owngoal_by_season | season_id | n משחקים, n_goals; penalty/own_goal_n, per_match, goal_share ורווח Wilson. כל שער נספר פעם אחת בסיכום הליגה. per_match יכול לעלות על 1 ולכן אינו שיעור בינומי |
| red_card_stats | part, group, season_id | טבלה ארוכה: cards עם group=אדום לקבוצה:אדום ליריבה (0/1; 1 פירושו ≥1); first_card_minute עם up_to_60/after_60; second_yellow לפי עונה. בשני החלקים הראשונים n שורות קבוצה×משחק, W/D/L ורווחי Wilson, ppm ורווח עונות; בשלישי n הוא אדומים, second_yellow_n/share ורווח Wilson. season_id=0 מסמן סיכום כל עונות הבחירה, לא עונה אמיתית; עמודות לא רלוונטיות לחלק נשארות חסרות |

### כללי ציר הזמן והנקודות

- מוחרגים 13 משחקי ht_conflict ומשחק 864982 ללא מבקיעים אמינים: 14 מתוך 4,315.
  נשארים 4,301 משחקים ו־8,602 שורות. שאר תוצאות המשחקים אינן נמחקות מניתוח התוצאות.
- בכל תצוגת מחצית/תזמון/ציר זמן נשמרים attrs: n_total_matches,
  n_included_matches, n_excluded_matches ו-exclusions עם סיבות זרות. סיכום ההחרגות
  עצמו הוא timeline_exclusions. אין מחיקה שקטה של שער חסר שיוך מחצית במשחק תקין:
  הפונקציה נכשלת, וכך גם באי־התאמה בין שערי הציר לתוצאה.
- הסדר הוא half, minute, goal_number. תוספת הראשונה נצמדת לשעון 45 ותוספת השנייה
  ל־90; minute המקורי אינו משתנה. משך אפס מותר. סדר שערים של שתי קבוצות באותה דקה
  הוא מוסכמת תיאור בלבד; first_goal_effect שומר עמימות לפי זיהוי השער הראשון הקיים.
- **הבהרת עמית:** points_dropped_from_lead=3−points_earned רק אם הייתה הובלה;
  ערכים 0/2/3, אחרת NA. points_won_from_behind=points_earned רק אם היה פיגור;
  ערכים 0/1/3, אחרת NA. משחק שבו הקבוצה גם הובילה וגם פיגרה מופיע בשניהם.
- equalized_within_10: לפחות שוויון אחד בתוך 10 דקות שעון מספיגה שהשאירה בפיגור;
  הספיגות נבחנות בתוך פרק פיגור עד לשוויון. בלי ספיגה כזו NA; מכנה השיעור הוא משחקים
  עם חשיפה, לא מספר אירועי הספיגה. תוספת זמן מקוצרת עלולה להשפיע על המרווח.
- שערי פנדל הם רק בעיטות שהובקעו. אין נתוני החמצות או שריקות. ספירת סוגי השערים
  משתמשת ב־4,314 משחקים, ורק 864982 מוחרג. שערי עצמי כבר נזקפו לקבוצה הנהנית ב-goals.
- red_card_stats מקבלת `red_cards=` כקלט מפורש שנקרא ב-Repository.read_filtered לפי
  game_id של משחקי הבחירה. אין I/O בפונקציה ואין הרחבה של AnalysisTables או loader.
  12 כרטיסים ללא דקה נשמרים בספירות ובחלק הצהוב השני; שורת קבוצה עם דקה חסרה לאדום
  כלשהו מוחרגת מפירוק האדום הראשון, מפני שלא ניתן לדעת אם האדום החסר היה הראשון.
  attrs: n_cards, n_missing_minute, n_timing_excluded_team_games.
- ממוצעי ppm ב-halftime_transitions/first_goal_effect/red_card_stats ופיזור השערים מקבלים
  n_bootstrap=2000; בטסטים 50. רווחים דורשים שתי עונות לפחות. מדגם n<30 לא מצדיק מסקנה;
  השוואות מרובות ורווחי שערים תלויים אינן מתוקנות אוטומטית. קשר, לא סיבה, במיוחד באדומים.

### הבהרות QA לפעימה D: קצב לפי חשיפה וחוסן

- scoring_rate_by_state משתמשת ב-score_before של goal_events. באצוות שערים
  באותה דקה שבהן המצב אינו ידוע, השערים אינם משויכים למצב בכוח. בדאטה:
  18 שערים כאלה (8 בית, 10 חוץ). attrs.n_missing_score_before ו-
  missing_score_before_by_venue מדווחים אותם. דקות החשיפה אינן נמחקות.
  10,576 שערים משויכים ו־18 חסרים משלימים 10,594 שערים בציר הזמן התקין.
- החשיפה נלקחת מאותו game_timelines בדיוק, עם אותן 14 החרגות; לכל צד
  387,090 דקות (4,301×90). לכל מצב מצורפות כל עונות הבחירה שיש בהן משחקים,
  גם אם באותה עונה אין שערים או אין חשיפה במצב המסוים.
- רווחי היחסים דוגמים בלוקי עונות שלמים ומחשבים מחדש סך מונה / סך מכנה
  בכל דגימה, ולא ממוצע לא־משוקלל של שיעורי עונות. n_bootstrap=2000,
  זרע 20261014, ו־50 בטסטים. בפחות משתי עונות אין רווח; מכנה אפס
  נותן NA. דגימות bootstrap עם מכנה אפס אינן נכנסות לרווח.
- timeline_resilience שומרת את הסכומים ואת n_led/n_trailed לצד השיעורים.
  השיעור הראשון מתוך משחקי פיגור בלבד, והשני מתוך משחקי יתרון בלבד.
  כל משחק יכול להופיע בשני המכנים, בהתאם למהלך המשחק. min_seasons=5 נשמר.
- D6 מציג קצב הבקעה לפי בית/חוץ ומצב, לא את התפלגות השערים בתוך דלי דקה;
  goal_timing נשארת כטבלת עזר. D8 מציג שיעורי נקודות עם רווחים בשני הצירים.
- המתאם הכולל בית/חוץ מערב הבדלי חוזק עם התנהגות בתוך המשחק. תלות מותנית
  תיבדק במחברת 06 רק לאחר ש-goals יקבע λ לכל צד; המתאם הגולמי אינו בדיקה כזאת.


### D — חצאים אקראיים להשוואת Poisson (`poisson_check.py`)

שתי הפונקציות מקבלות `tables`, `seasons=DEVELOPMENT_SEASONS` (8–25), `seed=0`, `iterations=100` חיובי. הבחירה נבדקת כלא ריקה, ללא כפילות ומתוך development; נדרשים משחקים ששוחקו עם שני ציוני שערים מלאים. **זו EDA עם שני חצאים אקראיים משלימים בתוך כל עונה, לא OOF כרונולוגי, לא נתוני prematch ולא מדידת ביצועי M2.**

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `poisson_crossfit_predictions` | אינדקס המשחקים, ממוין | כל משחק מקבל fold=0/1 בהגרלה בינומית, בלי הבטחה לחצאים שווי גודל; שני החצאים בכל עונה חייבים להיות לא ריקים. בכל fold מתאימים כוח התקפה והגנה ב־100 איטרציות כברירת מחדל על החצי המשלים, עם pseudocount=0.5. הפלט: `season_id`, `home_team_id`, `away_team_id`, `home_goals`, `away_goals`, `fold`, `lambda_home`, `lambda_away`. attrs: `method`, `seed`, `iterations`, `chronological=False`. קבוצה שלא הופיעה באימון מקבלת כוח ניטרלי 1. |
| `poisson_crossfit_comparison` | `scoreline` | אותם קלטים ופרמטרים. שורות 0:0, 1:0, 0:1, 1:1, 2:2, all_draws: `n`, `count`, `observed`, `expected` כממוצע ההסתברות הפרטנית למשחק, `observed_ci_lower/upper` של Wilson, `ratio`. attrs זהים ל־predictions. תיקו צפוי הוא Skellam ב־0 לכל λ, עם טיפול נפרד באפס; כולל את הזנב המלא. |

שערי המשחק המוחזק בחוץ אינם נכנסים לאומדן ה־λ של **אותו משחק**, ושני צדי משחק תמיד באותו fold. לעומת זאת מותרות כאן תוצאות מאוחרות יותר באותה עונה, ואותו משחק יכול להשתתף באומדן למשחקי החצי האחר. הפרמטרים לא נבחרו בבקטסט כרונולוגי. אומדני כוח מחצי עונה רועשים; יחס נצפה/צפוי לתוצאה מסוימת אינו לבדו הצדקה לתיקון הסתברות שלה.

### D — יציבות וכיול תהליך האפס (`score_review.py`)

פונקציות המקבלות `tables` או מסגרת משחקים ו־`seasons` מסננות את הבחירה ל־8–25 ודוחות בחירה מחוץ ל־development, כפולה או ריקה. ברירת המחדל היא כל development, לא עידן 14. `summarize_poisson_null` מקבלת מדדים מסוכמים בלבד ואינה יכולה לבדוק עונות: הקורא אחראי לסינון. `R` להלן הוא שם תיאורי למספר השחזורים; שם הפרמטר בקוד הוא `replicates`.

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `poisson_baseline_comparison` | `scoreline` | קלט `tables`, `seasons=8–25`. שורות התוצאות הנמוכות ו־all_draws, עם `observed`, `global_expected`, `global_ratio`, `per_game_expected`, `per_game_ratio`. משווה קו בסיס של ממוצעי הליגה בתוך המדגם לקו בסיס עם כוח קבוצות מחצאים משלימים. חצאי ההשוואה משתמשים ב־seed=0, iterations=100, ללא פרמטרים כאלה בחתימה. |
| `score_coverage` | `maximum=3..7` | קלט `tables`, `seasons=8–25`. `n`, `outside` (לפחות צד אחד מעל maximum), `coverage=1−outside/n` של ריבוע תוצאות מדויקות **לפני** איגום הזנב. attrs: `maximum` כשיא השערים שנצפה בצד כלשהו, `outside_scores` כמילון `(home_goals,away_goals)→count` למשחקים שבהם צד הגיע ל־5, `strong_tail_5`, `strong_tail_6` כהסתברויות שערי Poisson עם λ=3 להגיע לפחות ל־5/6. |
| `score_stability` | זוג: סטיות ב־`season_id × category`; מגמות ב־`scoreline` | קלט `tables`, `seasons=8–25`, `permutations=5000` חיובי, `seed=20261029`. קטגוריות 1:1,1:0,0:0,0:1,2:0,2:1,1:2,0:2,2:2,other. נדרשות לפחות שתי עונות וכל הקטגוריות חייבות להופיע במדגם המאוגם. הסטיות הן שיעור עונתי פחות שיעור הליגה שנבחרה, ביחידות 0–1. attrs של טבלת הסטיות: `counts`, `chi2`, `dof`, `p_asymptotic`, `p_permutation`, `permutations`, `seed`, `seasonal_standard_error` (עוגן קבוע sqrt(0.12×0.88/240)). טבלת תשע המגמות: `slope_pp` בנקודות אחוז לעונה, `p`. |
| `poisson_truth` | אינדקס המשחקים, ממוין | קלט `tables`, `seasons=8–25`, `iterations=100` חיובי; נדרשות תוצאות מלאות. אומדת התקפה/הגנה מכל משחקי **העונה השלמה**, באותה שיטת pseudocount=0.5. מחזירה רק `season_id`, `home_team_id`, `away_team_id`, `lambda_home`, `lambda_away`; עמודות השערים האמיתיים אינן מוחזרות. זו אמת plug-in לתהליך האפס, לא חיזוי מחוץ למדגם. |
| `simulate_poisson_goals` | אינדקס המשחקים שנבחרו, ממוין | קלט `truth` עם מזהי העונה והקבוצות ושתי עמודות λ, `seasons=8–25`, `seed=20261029`. λ חייבים להיות סופיים ולא שליליים. מחזירה מזהים ו־`home_goals`, `away_goals` שהוגרלו כשערי Poisson עצמאיים. משתמשת **רק ב־λ** להגרלה, ואינה קוראת שערים אמיתיים גם אם הוספו לקלט. λ אינם מועתקים לפלט. |
| `poisson_diagnostic_metrics` | Series בשם `observed`, אינדקס מדדים | קלט `predictions` הכולל מזהים, שערים ושתי עמודות λ, ו־`seasons=8–25`; λ חייבים להיות סופיים וחיוביים ממש. `dispersion_home/away=Σ(y−λ)²/Σλ`, `residual_correlation` בין `(y−λ)/sqrt(λ)` של שני הצדדים; יחסי נצפה/צפוי ל־0:0,1:0,0:1,1:1,2:2 ו־`all_draws`; `draw_balanced`, `draw_middle`, `draw_unbalanced` בשלישים שווי מספר משחקים בקירוב לפי abs(lambda_home−lambda_away). שוויון נשבר בסדר האינדקס הממוין. תיקו מחושב לפי Skellam מלא, ולא לפי קטגוריית "5+:5+". |
| `summarize_poisson_null` | אינדקס המדדים | קלט `observed: Series`, `simulations: DataFrame` ששורותיה שחזורים ועמודותיה אותם מדדים. מחזירה `observed`, `null_mean`, `null_sd` מדגמית, `z=(observed−null_mean)/null_sd`, `p_mc`, `null_lower`, `null_upper` כאחוזונים 2.5/97.5. p דו־צדדי סופר מרחק מוחלט מממוצע הסימולציות, עם plus-one: ‏(1+מספר חריגים)/(R+1). אין bootstrap או פרמטר seed בפונקציה זו. |
| `calibrated_poisson_check` | זוג: סיכום באינדקס מדדים; שחזורים ב־RangeIndex | קלט `tables`, `seasons=8–25`, `replicates=200` (לפחות 2), `seed=20261029`, `observed_seed=0`, `iterations=100`, `progress=None`. מחשבת מדדים אמיתיים מחצאי seed=observed_seed, בונה אמת מעונה שלמה, ומגרילה R ליגות מתוצאות Poisson; בכל שחזור מתאימה **מחדש** כוחות בשני החצאים האקראיים ומחשבת אותם מדדים. SeedSequence(seed).spawn(R) מספקת לכל שחזור זרעים נפרדים לשערים ולחלוקה לחצאים; progress(i) נקראת אחרי כל שחזור, i=1..R. סיכום כסכמת summarize_poisson_null; טבלת R השחזורים עם מדד בכל עמודה. attrs בסיכום: `replicates`, `seed`, `observed_seed`, `iterations`, `algorithm="score-review-1"`, `split`, `calibrated_draw_excess=observed(all_draws)/null_mean(all_draws)−1`. |

מבחן היציבות שומר במדויק את מספר המשחקים בכל עונה ואת שכיחות הקטגוריות המאוגמת, דרך הגרלת טבלאות שכיחות מותנות; p המונטה־קרלו משתמש ב־plus-one. תשע מגמות התוצאות הן חקרניות, בלי תיקון ריבוי השוואות מאשר. נתון seasonal_standard_error הוא קנה מידה להמחשה, לא SE שחושב לכל אחת מעונות הקלט.

בבדיקה המכוילת **האמת היא כוח שנאמד מעונה שלמה**. אומדן האמת והמדדים הנצפים כן קוראים שערי עבר אמיתיים; ההגרלה עצמה קוראת רק λ, וההתאמה המחודשת בכל שחזור רואה רק את השערים המדומים. כך שומרים באפס גם את הרעש שנוצר משיטת אומדן הכוח בחצאים. טווח 95% null_lower/null_upper הוא **טווח של תהליך האפס, לא רווח סמך לאפקט הנצפה**; z הוא סטייה מהאפס המכויל ולא מבחן נורמלי אוטומטי. האמת המותאמת מעונה שלמה עצמה רועשת, ולכן רעש השיטה עשוי להיות מוערך ביתר. זו EDA בדיעבד: אין לפרש חצאים אקראיים כ־OOF כרונולוגי, או לבחור תיקון ייעודי ל־1:1/2:2 לפי יחס גולמי. שימוש עתידי ב־M3 ייבדק עם λ כרונולוגי של M2.

### D — מטמון הבדיקה (`ligat/analysis/score_cache.py`)

| פונקציה / טבלה | אינדקס | קלט, פלט ומשמעות |
| --- | --- | --- |
| `cached_poisson_check` | זוג: סיכום באינדקס מדדים; שחזורים ב־RangeIndex | קלט `tables`, `cache_path: Path` מפורש, `seasons=8–25`, `replicates=200`, `seed=20261029`, `observed_seed=0`, `iterations=100`, `progress=None`. מחזירה אותו זוג טבלאות כמו calibrated_poisson_check. attrs הסיכום כוללים את attrs הכיול ובנוסף `cache_key`, `simulation_seconds` של החישוב המקורי ו־`cache_hit`. בקריאת מטמון תקין cache_hit=True; בחישוב חדש False. |

מפתח SHA-256 כולל hash של עמודות season_id, home_team_id, away_team_id, home_goals, away_goals והאינדקס, ממוין לפי אינדקס, לאחר סינון development; תוכן המקור של `poisson_check.py` ושל `score_review.py`; ורשומת פרמטרים עם העונות, R, שני הזרעים, iterations, pseudocount=0.5, גרסת האלגוריתם וגרסאות NumPy/SciPy/pandas. אין במפתח גרסת Python או hash של כל קובצי הפרויקט. `progress` והנתיב עצמם אינם חלק מהמפתח.

קובץ JSON שומר key, parameters, attrs, summary ו־samples בפורמט split; אין pickle. בכשל קריאה/פענוח נתמך או בשינוי המפתח מחשבים מחדש. הכתיבה לקובץ אח זמני עם הסיומת `.tmp` והחלפה אטומית; ההורה של cache_path חייב להתקיים, ואינו נוצר אוטומטית. הנתיב נבחר בידי הקורא למטמון נגזר ומוחרג מ־Git, ללא כתיבה ל־clean/raw או לתוצרי מודל ייצור. הפונקציה היא מעטפת I/O ל־EDA המכוילת; אין להפעילה כדי לייצר פיצ׳רים או ציוני M2. הסתייגויות האמת הרועשת וטווח תהליך האפס חלות גם על פלט שנקרא מהמטמון.

## מחברת 02, פרק E: הקשר תיאורי

הטבלאות הבאות נוצרות בזיכרון ב־`ligat.analysis.league_stats.context`; אינן נשמרות
ב־`data/clean` ואינן פיצ'רים. עונות 8–25 בלבד, ו־ValueError לבחירה מחוץ לטווח.
נקודות תמיד 3/1/0 על המגרש. מיקום רשמי קודם משמש להשוואה תיאורית בלבד.
כל האינדקסים נוצרים עם `drop=True`.

| פונקציה | אינדקס | תוכן ומשמעות |
| --- | --- | --- |
| upset_rates | metric, gap_bin, weaker_side | metric=value_ratio/rank_gap; n, פירוק home/draw/away עם Wilson, ובכוח ידוע גם weak_win/strong_win_rate עם Wilson. עולות מול ותיקות מקבלות promoted_win/veteran_win_rate. עמודות שלא ניתן לפרש נשארות NA |
| t1_strength_comparison | model | LOSO על אותו מדגם development: baseline/value/rank_difference/rank_log/value_rank_log; n, n_seasons, log_loss, RPS. מוציא עולות, עונה 8 ושווי או מיקום חסר/לא חיובי. attrs: correlation_lv_rk, comparisons (הפרשי log loss, bootstrap עונות עם CI ומספר עונות טובות יותר), seasonal_losses, seed ו־n_bootstrap. זו EDA בדיעבד, לא walk-forward או בחירת מודל |

### אי־ודאות

- רווחי ממוצעים דוגמים עונות שלמות, n_bootstrap=2000 כברירת מחדל, זריעה קבועה
  כמו בפונקציות הקיימות; בטסטים משתמשים ב־50. פחות משתי עונות נותן רווח חסר.
  Wilson הוא רווח שולי תיאורי, לא רווח עם תיקון לתלות בין משחקים.

### קבוצות השוואה, קטגוריות וחוסרים

- פער שווי: יחס הגדול לקטן של prev_season_value_share, ללא השלמה, בחמישונים
  הנקבעים מכל המשחקים בעלי שני ערכים חיוביים ולא שווים. value_ratio_edges ב־attrs.
  פער מיקום קודם: 1–3, 4–7, 8+ מקומות. weaker_side מציין home_weaker/away_weaker;
  בכל מקרה סכום n בכל metric הוא מספר המשחקים, אין לחבר את שני ה־metric למדגם אחד.
- equal_strength: שוויון מלא במדד; unknown_both: לשתיהן אין t−1; בשניהם רק בית/תיקו/חוץ.
  veteran_vs_promoted: רק לאחת t−1, והשנייה עולה מזוהה; פירוק לוותיקה/עולה בבית ובחוץ,
  בלי להגדיר את העולה אוטומטית כחלשה. missing_source: חסר ערך גם כשלא מדובר בעולה.
  בקטגוריות אלה weak_win/strong_win נשארים NA. אין מילוי חוסרים ואין שכבת prematch.

בכל ההשוואות מציגים n ורווחים; לא מסיקים מ־n<30.
זהות הקבוצות ולוח המשחקים עלולים להסביר את הקשרים. חפיפת רווחים
אינה הוכחת שוויון; קשר אינו סיבה.
