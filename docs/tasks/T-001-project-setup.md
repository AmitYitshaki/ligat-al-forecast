# T-001: הקמת שלד הפרויקט

**סטטוס:** בבדיקה
**מבצע:** Codex
**תלויות:** —

## מטרה
ריפו עובד עם מבנה התיקיות מ-`TECHNICAL_GUIDELINES.md`, סביבת Python, כלי איכות ו-CI.

## הסבר כדורגלני
אין. זו תשתית: בלעדיה כל שלב אחר נבנה על חול.

## קלט ופלט
- קלט: `docs/TECHNICAL_GUIDELINES.md` (סעיפים 1, 2, 7, 11).
- פלט: `pyproject.toml`, מבנה `ligat/` עם קבצי `__init__.py` ריקים, `tests/` עם טסט דמה, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`.

## קריטריוני קבלה
- [x] `venv` + `pip install -r requirements-dev.lock` + `pip install --no-deps --no-build-isolation -e .` עובד על סביבה נקייה.
- [x] `ruff check`, `mypy ligat` ו-`pytest` עוברים.
- [ ] CI רץ על PR ועובר.
- [x] `README.md` מעודכן עם הוראות התקנה.

## מחוץ להיקף
כל קוד לוגי. רק שלד.

## מצב ביצוע: 9.10.2026

השלד הוקם ב-venv עם Python 3.13. התלויות נעולות, והבדיקות המקומיות עברו (9 בדיקות ייבוא). ה-CI על Windows ו-Linux ייבדק ב-PR. המשימה ממתינה ל-QA של Claude Code ולאישור עמית לפני מיזוג; לא בוצעה גרידה.
