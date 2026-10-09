# הנחיות טכניות

> מסמך חי. כל סוכן שכותב או בודק קוד קורא אותו לפני שמתחיל.
> עודכן לאחרונה: 8.10.2026

---

## 1. סטאק

| תחום | כלים |
| --- | --- |
| שפה | Python 3.11+ |
| דאטה | pandas, numpy, pyarrow (Parquet) |
| איסוף | requests, beautifulsoup4, lxml |
| מודלים | xgboost, scikit-learn, scipy, statsmodels (לבנצ'מרקים) |
| הסבר מודל | shap |
| ויזואליזציה | plotly, kaleido (ייצוא PNG) |
| גרידה דינמית | playwright (רק אם T-004 יחליט שצריך) |
| כיוונון | optuna |
| ולידציה של דאטה | pandera |
| קונפיגורציה | PyYAML + pydantic |
| איכות קוד | ruff (lint + format), mypy, pre-commit |
| בדיקות | pytest, pytest-cov |
| CLI | typer |
| CI | GitHub Actions: ruff + mypy + pytest על כל PR |

ניהול סביבה עם `uv` או `venv` + `pyproject.toml`. גרסאות התלויות נעולות.

## 2. מבנה הריפו

```
ligat-al-forecast/
├── ligat/                      # החבילה
│   ├── config.py               # נתיבים והגדרות; טוען config/*.yaml
│   ├── domain.py               # טיפוסי הדומיין: Match, TeamSeason, SeasonFormat
│   ├── scrape/
│   │   ├── base.py             # Source (ממשק)
│   │   └── ifa.py              # IFASource
│   ├── data/
│   │   ├── parse.py            # HTML → רשומות; היפוך תוצאה; ניקוי אצטדיון
│   │   ├── build.py            # בניית matches / round_tables / final_tables
│   │   ├── repository.py       # קריאה וכתיבה של כל הדאטה סטים
│   │   ├── schemas.py          # סכמות pandera
│   │   └── validate.py         # טבלה מחושבת מול רשמית
│   ├── features/
│   │   ├── base.py             # FeatureBlock (ממשק)
│   │   ├── attack_defense.py
│   │   ├── home_away.py
│   │   ├── momentum.py
│   │   ├── motivation.py
│   │   ├── matchup.py
│   │   ├── elo.py
│   │   └── pipeline.py         # מרכיב FeatureBlocks לטבלת פיצ'רים
│   ├── models/
│   │   ├── base.py             # MatchModel (ממשק)
│   │   ├── registry.py         # יצירת מודל לפי שם מהקונפיגורציה
│   │   ├── baselines.py        # B0–B3
│   │   ├── xgb_goals.py        # מודל 1
│   │   ├── dixon_coles.py      # חלופה
│   │   └── points_corrector.py # מודל 2
│   ├── simulate/
│   │   ├── rules.py            # שוברי שוויון, פיצול לפלייאוף, חלוקת נקודות
│   │   └── season.py           # SeasonSimulator
│   ├── analysis/               # חישובים לגרפים ולדוחות (טהורים, נבדקים)
│   ├── viz/                    # תשתית גרפים: ראו docs/VISUALIZATION.md
│   ├── evaluate/
│   │   ├── metrics.py          # RPS, log loss, כיול, RMSE
│   │   └── backtest.py         # walk-forward בשתי קומות
│   └── cli.py                  # scrape / build / features / train / backtest / forecast
├── config/
│   ├── seasons.yaml            # פורמט כל עונה
│   ├── teams.yaml              # שם קנוני לכל team_id
│   └── experiment.yaml         # היפר-פרמטרים, טווחי עונות, זרעים
├── data/
│   ├── raw/                    # לא נוגעים. לעולם.
│   ├── clean/
│   └── features/
├── notebooks/                  # הצגה וחקירה בלבד
├── tests/
├── reports/                    # figures/ (נבנה מחדש, לא ב-Git) + gallery/
├── forecasts/                  # append-only
├── docs/
└── pyproject.toml
```

## 3. עקרונות OOP

**מחלקה רק כשיש מצב או ממשק.** ניקוי ופיצ'רים הם בעיקר פונקציות טהורות שמקבלות DataFrame ומחזירות DataFrame. מחלקות מתאימות ל:
- רכיבים עם מצב: `IFASource` (session, השהיה, מטמון), `EloRating`, `SeasonSimulator`.
- ממשקים שיש להם כמה מימושים: `MatchModel`, `FeatureBlock`, `Source`.

**SOLID כפי שהוא חל אצלנו:**

| עיקרון | מה זה אומר בפרויקט |
| --- | --- |
| Single Responsibility | המודול שמושך HTML לא מפענח אותו; המודול שמפענח לא שומר לדיסק |
| Open/Closed | מודל חדש = מחלקה חדשה שממשת `MatchModel`. הסימולטור והבקטסט לא משתנים |
| Liskov | כל `MatchModel` מחזיר את אותו פורמט פלט בדיוק; אפשר להחליף אחד באחר |
| Interface Segregation | ממשקים קטנים: `fit`, `predict_goals`. לא "מחלקת-על" עם 20 מתודות |
| Dependency Inversion | הסימולטור מקבל `MatchModel` בבנאי, לא יוצר XGBoost בעצמו |

**הרכבה עדיפה על ירושה.** ירושה רק מממשק מופשט (`ABC` או `Protocol`), לא שרשראות ירושה.

**Dataclasses לישויות דומיין** (`Match`, `SeasonFormat`): `frozen=True` כשאפשר.

## 4. Design Patterns בשימוש

| Pattern | איפה | למה |
| --- | --- | --- |
| **Strategy** | `MatchModel` עם B0–B3, XGBoost, Dixon-Coles | החלפת מודל בלי לגעת בסימולטור או בבקטסט |
| **Registry / Factory** | `models/registry.py` | יצירת מודל לפי שם מ-`experiment.yaml` |
| **Repository** | `data/repository.py` | נקודת גישה אחת לדאטה; אף מודול לא קורא Parquet ישירות |
| **Adapter** | `scrape/base.py` → `ifa.py` | מקור חדש (למשל FBref) = מתאם חדש, אותו פלט |
| **Composite / Pipeline** | `features/pipeline.py` מרכיב `FeatureBlock`s | הוספה או הסרה של קבוצת פיצ'רים בשורה אחת; מאפשר ablation |
| **Template Method** | `evaluate/backtest.py` | לולאת walk-forward קבועה; המודל והפיצ'רים משתנים |
| **Registry (Decorator)** | `viz/registry.py`: `@chart(...)` | כל גרף נרשם פעם אחת; בנייה, גלריה וטסטים מוצאים אותו אוטומטית |

לא משתמשים ב-Singleton. הגדרות מועברות כאובייקט config.

### ממשקים מרכזיים (חוזה)

```python
class MatchModel(ABC):
    name: str

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.Series,
            sample_weight: pd.Series | None = None) -> "MatchModel": ...

    @abstractmethod
    def predict_goals(self, features: pd.DataFrame) -> pd.DataFrame:
        """מחזיר עמודות: match_id, team_id, expected_goals (λ)."""


class FeatureBlock(ABC):
    name: str            # שם המושג הכדורגלני: "attack_defense", "motivation"...
    football_rationale: str   # משפט אחד: מה זה תופס על המגרש

    @abstractmethod
    def compute(self, matches: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame: ...
```

`football_rationale` הוא שדה חובה. פיצ'ר בלי הסבר כדורגלני לא נכנס.

## 5. נכונות בזמן (מניעת דליפה)

זה כלל הברזל של הפרויקט.

- כל פונקציית פיצ'ר מקבלת `as_of` ומשתמשת **רק** במשחקים שהתאריך שלהם קטן ממש מ-`as_of`.
- Elo נשמר לפני המשחק, לא אחריו.
- פיצ'רי "עונה קודמת" לוקחים את `final_tables` של העונה הקודמת בלבד.
- Early stopping וכיוונון: על עונה מאוחרת יותר, לעולם לא על דגימה אקראית.
- עונות ההחזקה (2024/25, 2025/26) לא נטענות בשום קוד כיוונון. ה-backtest מקבל אותן רק עם דגל מפורש `--final-eval`.
- טסט חובה: שינוי תוצאה של משחק לא משנה אף פיצ'ר של אותו משחק.

## 6. דאטה

- `data/raw/` בלתי ניתן לשינוי. סקרייפר כותב; אף אחד אחר לא.
- כל שלב קורא מהשכבה הקודמת וכותב לשכבה שלו, בפורמט Parquet.
- כל דאטה סט עובר ולידציה של סכמת pandera בכתיבה ובקריאה.
- מזהים: `team_id` ו-`game_id` של ההתאחדות הם המפתחות. שמות קבוצות לתצוגה בלבד, מ-`config/teams.yaml`.
- עמודות ב-`snake_case`, באנגלית. תאריכים כ-`datetime64` עם אזור זמן `Asia/Jerusalem`.

## 7. איכות קוד

- Type hints בכל פונקציה ציבורית; `mypy` עובר.
- Docstring לכל פונקציה ציבורית: מה היא עושה, ובפונקציות דומיין גם המשמעות הכדורגלנית.
- פונקציות קצרות; אם פונקציה עושה שני דברים, מפצלים.
- `logging` ולא `print`. שגיאות נכשלות בקול: לא בולעים exceptions.
- אין מספרי קסם בקוד. כל קבוע (השהיה, מספר סימולציות, זמן מחצית) ב-config.
- זרעים קבועים לכל אקראיות (`numpy.random.Generator` עם seed מה-config).
- הסימולציה מוקטרת ב-NumPy: לולאה רק על מחזורים, לא על סימולציות.

## 8. מחברות

- מחברות **לא מכילות לוגיקה.** הן מייבאות מ-`ligat` ומציגות.
- כל מחברת עוברת **Restart & Run All** לפני commit.
- פלטים מנוקים לפני commit (`nbstripout`), מלבד מחברת התוצאות הסופיות.
- שמות ממוספרים: `01_raw_data.ipynb`, `02_eda.ipynb`...

## 9. בדיקות

| סוג | דוגמאות חובה |
| --- | --- |
| יחידה | פענוח `"0 - 3"` → בית 3, חוץ 0; ניקוי "(סגור)"; RPS על מקרים ידועים |
| חוקי ליגה | חלוקת נקודות בעונה 2009/10 משחזרת 77 → 39; שוברי שוויון |
| שחזור | הזנת כל התוצאות האמיתיות של עונה → הטבלה הרשמית בדיוק, לכל עונה |
| דליפה | שינוי תוצאת משחק לא משנה את הפיצ'רים שלו |
| סימולטור | כל הקבוצות שוות → ≈ 1/14 לאליפות; סכום הסתברויות בכל שורה ועמודה = 1 |
| מודל | כל `MatchModel` מחזיר פלט בפורמט החוזה |
| Golden files | קובץ HTML שמור של מחזור אחד מכל פורמט → פלט מפוענח קבוע |

טסטים לא ניגשים לרשת. הסקרייפר נבדק מול קבצי HTML שמורים ב-`tests/fixtures/`.

## 10. סקרייפר

- השהיה ≥ 2 שניות בין בקשות; בלי מקביליות.
- ניסיון חוזר עם backoff אקספוננציאלי (עד 3 ניסיונות) על שגיאות רשת ו-5xx; עצירה על 403.
- כל תגובה נשמרת ל-`data/raw/ifa/{season_id}/round_{round_id}.xml` עם חותמת זמן.
- אם הקובץ קיים ועונה סגורה: לא מושכים שוב.
- User-Agent מזוהה.
- **רץ על המחשב המקומי** (הסביבה בענן חסומה לאתר).

## 11. Git

- ענף `main` מוגן; כל שינוי ב-PR.
- ענפים: `feat/...`, `fix/...`, `test/...`, `docs/...`.
- Conventional Commits: `feat(scrape): parse round tables`.
- PR קטן ומפוקס: משימה אחת, עד כמה מאות שורות.
- `data/` לא ב-Git (מלבד דוגמאות קטנות ל-fixtures).
- `forecasts/` **כן** ב-Git, בכוונה: commit עם תאריך לפני כל מחזור מוכיח שהתחזית נעשתה מראש.

## 12. Definition of Done

משימה גמורה רק כאשר:
- [ ] הקוד עובר ruff, mypy ו-pytest ב-CI.
- [ ] יש טסטים לפונקציונליות החדשה.
- [ ] אין דליפת מידע (סעיף 5).
- [ ] פיצ'ר חדש כולל `football_rationale`; גרף חדש כולל `football_question`.
- [ ] אם הדאטה השתנה: `ligat viz build` עובר והגרפים המושפעים נבדקו בגלריה.
- [ ] המסמכים ב-`docs/` עודכנו אם השתנתה החלטה.
- [ ] Claude Code (QA) אישר את ה-PR.
