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

כרגע מותקנים כלי איסוף ואיכות קוד בלבד. ספריות דאטה, מודלים וגרפים יתווספו במשימות המתאימות.
עדיין אין סקרייפר או מודל פעיל.

## צוות

עמית יצחקי (מנהל הפרויקט) · Gemini (תכנון ורכזות) · Codex (כתיבת קוד) · Claude Code (QA ואנליזה)
