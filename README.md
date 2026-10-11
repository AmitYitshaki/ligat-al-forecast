# Ligat Al Forecast

חיזוי הסתברותי של הטבלה הסופית בליגת העל בכדורגל: לכל קבוצה הסיכוי לכל מקום, לאליפות ולירידה, מתעדכן אחרי כל מחזור.

**הגישה המתוכננת:** חיזוי משחקים וסימולציית יתרת העונה. בחירת המודלים תיעשה לפי הדאטה שייאסף; שילוב XGBoost ו-Dixon-Coles הוא כיוון לבדיקה. כרגע מתמקדים באיסוף מידע.

**הדאטה:** כל משחקי ליגת העל מ-2006/07, מהאתר הרשמי של ההתאחדות לכדורגל בישראל.

## מסמכים

| מסמך | מה יש בו |
| --- | --- |
| [עקרונות הפרויקט](docs/PROJECT_PRINCIPLES.md) | מטרה, דאטה, שיטה, יומן החלטות |
| [הנחיות טכניות](docs/TECHNICAL_GUIDELINES.md) | ארכיטקטורה, OOP, patterns, בדיקות |
| [מבנה הצוות](docs/TEAM_STRUCTURE.md) | מי עושה מה, ופרוטוקול האישורים |
| [מקורות דאטה](docs/DATA_SOURCES.md) | נקודות קצה, מלכודות, שאלות פתוחות |
| [מילון הדאטה](docs/DATA_DICTIONARY.md) | כל עמודה בכל דאטה סט |
| [מילון מושגים](docs/GLOSSARY.md) | הסברים למושגים |
| [ויזואליזציה](docs/VISUALIZATION.md) | איך גרפים נבנים, וקטלוג הגרפים |
| [מפת דרכים](docs/ROADMAP.md) | שלבים ושערי מעבר |
| [משימות](docs/tasks/README.md) | סדר המשימות ותלויות |

## התקנה

סביבת העבודה היא `venv`, עם Python 3.13 בהקמה הנוכחית. ה-CI בודק Windows ו-Linux.

ב-PowerShell, מתוך תיקיית הפרויקט:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m pip check
```

בדיקות:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy ligat
.\.venv\Scripts\python.exe -m pytest
```

ב-Linux/macOS מחליפים את הנתיב ל-Python ב-`.venv/bin/python`.
להפעלת hooks יש להפעיל את הסביבה ואז להריץ `python -m pre_commit install`.
`requirements-dev.lock` נוצר עם pip-tools על Python 3.13/Windows; CI בודק גם Linux.
לרענון הנעילה בסביבה הפעילה: `python -m piptools compile --extra dev --strip-extras --allow-unsafe --no-emit-index-url --no-emit-trusted-host --output-file requirements-dev.lock pyproject.toml`.
התלויות החיצוניות נעולות; החבילה המקומית מותקנת בנפרד במצב editable.

לפי אישור עמית, מותקנות ספריות האיסוף, הדאטה, המודלים, הגרפים ואיכות הקוד המפורטות ב-`pyproject.toml`. התקנת ספריות מודלים אינה החלטה על בחירת המודל. Kaleido דורשת Chrome/Chromium לצורך ייצוא תמונות.
מימוש T-002 מוסיף סקרייפר מחזורים; עדיין אין מודל פעיל. טווח האיסוף המאושר הוא 2006/07–2026/27 (`season_id` 8–28).

לאחר ההתקנה, ניתן לבדוק את ממשק הפקודות **בלי פנייה לרשת**:

```powershell
.\.venv\Scripts\ligat.exe scrape history --help
.\.venv\Scripts\ligat.exe scrape live --help
```

**הפקודות הבאות פונות לרשת ודורשות אישור עמית לפני הרצה:**

```powershell
.\.venv\Scripts\ligat.exe scrape history --seasons 8-27
.\.venv\Scripts\ligat.exe scrape live
```

הפקודות בדוגמאות חייבות לרוץ **משורש הריפו**: גם `--config` וגם `raw_root` יחסיים לתיקיית העבודה הנוכחית.
ההגדרות נמצאות ב-`config/scrape.yaml` (ניתן לבחור קובץ עם `--config`).
היסטוריה נשמרת ב-`data/raw/ifa/rounds/{season_id}/`; ריצות live בתיקיית `snapshot/{timestamp}/` באותה עונה.
תגובה נכתבת תחילה לקובץ זמני באותה תיקייה ונבדקת לפני פרסום בשם הסופי, ללא דריסת קובץ קיים. תגובה שגויה נשמרת ב-`rejected/{timestamp}_{unique_id}_{filename}`, ותיעוד הדחייה נשמר ב-manifest. הרצה הבאה יכולה לנסות שוב בלי למחוק תגובות גולמיות.
רק קובצי staging זמניים של הריצה מנוקים; קבצים סופיים ותגובות שנדחו נשמרים ללא שינוי. קבצים זמניים שנותרו מקריסה אינם משמשים כמטמון.
403 או 429 עוצרים את המופע, ללא retry ופניות נוספות.
ל-T-003: ב-live יש לבחור רק את ה-snapshot האחרון שה-manifest שלו מסומן `completed`; snapshots עם `failed` או `in_progress` אינם דאטה שלם. אין לטעון קבצים מתיקיית `rejected`.
משחקי מבחן עלייה נאספים; ההחרגה שלהם תיעשה ב-T-003.
הבדיקות סינתטיות ואינן אישור לתאימות מול האתר החי.

## בניית הדאטה המקומי (T-003)

לאחר אישור עמית, מתוך שורש הריפו:

```powershell
.\.venv\Scripts\ligat.exe build
```

הבנייה פועלת ללא רשת וללא שינוי בקבצים הגולמיים. היא דורשת manifest שהושלם לכל עונה
בקונפיגורציה; בעונת live נבחר ה־snapshot האחרון שהושלם לפי `completed_at`.
גודל ו־SHA-256 של כל XML נבדקים לפני פענוח.
נוצרים `matches.parquet`, `fixtures.parquet`, `round_tables.parquet`, `final_tables.parquet`
תחת `data/clean/`, וטיוטת שמות ו־aliases תחת `config/teams.yaml`.
כל Parquet מאומת ב־Pandera לפני ואחרי הכתיבה ובכל קריאה דרך `Repository`.
כל קובץ מתפרסם בהחלפה אטומית; ארבעת הקבצים אינם טרנזקציה משותפת. אחרי קריסה בזמן הפרסום יש להריץ את הבנייה שוב לפני צריכת הפלט.

אפשר לבחור `--raw-root`, `--clean-root`, `--seasons-config`, `--overrides-config`,
ו־`--teams-config`. הנתיבים יחסיים לתיקיית העבודה. `ligat build --help` אינו בונה קבצים.
הרצה חוזרת מחליפה את הדאטה הנקי ושומרת העשרות ידניות בשדות השמות, האנגלית והעיר,
וכן aliases קיימים בטיוטת הקבוצות.

משחקי מבחן מוחרגים לפי שם הסבב ב־manifest. משחקים עתידיים נבחרים בנפרד בכל מחזור,
כולל מחזור ששוחק בחלקו. שעה עתידית שטרם נקבעה מיוצגת כחצות בתאריך המתוכנן,
ואין לפרש זאת כשעת פתיחה מאומתת. תאריכי משחקים ששוחקו כוללים את השעה מהמקור.
כל התאריכים, כולל `scraped_at`, נשמרים באזור הזמן `Asia/Jerusalem`.

`config/match_overrides.yaml` מכיל שישה סטטוסים מאומתים ושני מועמדים שלא מופעלים.
הרשימה אינה בהכרח מלאה; ההצלבה מול עמודי המשחק תבוצע בהמשך. משחקי `void` אינם מקבלים
נקודות גם כשהאתר סופר אותם כתיקו. `points_model` אינו כולל ענישה; הנקודות מתחילת
הפלייאוף הן קיזוז הנקודות הסדירות בעיגול למעלה, בתוספת נקודות הפלייאוף ללא קיזוז.
הקיזוז פעיל רק בעונות 11–12. טבלת הסיום אינה כוללת את העונה החיה.

`is_home_venue` הוא תיאור בדיעבד לפי האצטדיון השכיח בעונה, בהתאם למילון הדאטה.
הוא אינו פיצ׳ר מותר לחיזוי בזמן אמת; פיצ׳ר כזה ידרוש חישוב מחדש רק ממידע שקדם ל־`as_of`.
בדיקות T-003 משתמשות גם בקבצים אמיתיים שמורים, ללא פנייה לרשת.

## צוות

עמית יצחקי (מנהל הפרויקט) · Gemini (תכנון ורכזות) · Codex (כתיבת קוד) · Claude Code (QA ואנליזה)


## גרידת עמודי משחק — T-004

לאחר QA ואישור עמית בלבד, משורש הריפו וב־venv:

```powershell
ligat scrape games --seasons 8-27
ligat scrape games --live
```

המזהים נקראים רק ממחזורי raw, כולל המשחקים הטכניים, וללא משחקי מבחן.
בלייב נקרא רק ה־snapshot האחרון שהושלם ונשלפים רק עמודים שחסרים במטמון המאומת.
הפלט: `data/raw/ifa/games/{season_id}/{game_id}.html.gz` ו־manifest לכל עונה.
gzip משמר את תוכן התגובה; קובצי HTML ישנים אינם נמחקים או משתנים.
403/429 עוצרים מיד ללא retry; בכשל אחר ניתן להריץ שוב להמשך מהמטמון.
לפני ריצה ארוכה יש להשבית מצב שינה זמנית ב־Windows ולשחזר את ההגדרה המקורית בסיום.

דוח מקומי בלבד (ללא רשת), גם לפני אישור הריצה:

```powershell
ligat scrape games --seasons 8-27 --report
ligat scrape games --live --report
```

הדוח מציג צפי מהמחזורים מול עמודים מאומתים, missing/invalid, rejected וסטטוס לכל עונה.
היעד הנוכחי 4,842 עמודים: 4,802 היסטוריים ועוד 40 בלייב. חריגה מסיימת ב־exit code 1.
`--sample` נשאר זמין למדגם של עשר עונות; כל פנייה לרשת דורשת אישור.
