# Ligat Al Forecast

חיזוי הסתברותי של הטבלה הסופית בליגת העל בכדורגל: לכל קבוצה הסיכוי לכל מקום, לאליפות ולירידה, מתעדכן אחרי כל מחזור.

**הגישה המתוכננת:** חיזוי משחקים וסימולציית יתרת העונה. בחירת המודלים תיעשה לפי הדאטה שייאסף; שילוב XGBoost ו-Dixon-Coles הוא כיוון לבדיקה. כרגע מתמקדים באיסוף מידע.

**הדאטה:** כל משחקי ליגת העל מ-2007/08, מהאתר הרשמי של ההתאחדות לכדורגל בישראל.

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

ההגדרות נמצאות ב-`config/scrape.yaml` (ניתן לבחור קובץ עם `--config`).
היסטוריה נשמרת ב-`data/raw/ifa/rounds/{season_id}/`; ריצות live בתיקיית `snapshot/{timestamp}/` באותה עונה.
כל תגובה נשמרת ללא שינוי, לצד `manifest.json` ושמות הסבבים. קבצים קיימים בהיסטוריה אינם נמשכים מחדש.
משחקי מבחן עלייה נאספים; ההחרגה שלהם תיעשה ב-T-003.
הבדיקות סינתטיות ואינן אישור לתאימות מול האתר החי.

## צוות

עמית יצחקי (מנהל הפרויקט) · Gemini (תכנון ורכזות) · Codex (כתיבת קוד) · Claude Code (QA ואנליזה)
