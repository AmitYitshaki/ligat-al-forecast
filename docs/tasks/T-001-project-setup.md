# T-001: הקמת שלד הפרויקט

**סטטוס:** פתוחה
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
- [ ] `pip install -e .` (או `uv sync`) עובד על סביבה נקייה.
- [ ] `ruff check`, `mypy ligat` ו-`pytest` עוברים.
- [ ] CI רץ על PR ועובר.
- [ ] `README.md` מעודכן עם הוראות התקנה.

## מחוץ להיקף
כל קוד לוגי. רק שלד.
