# EDA-01 C: דוח הרצה ואימות

שלב C של EDA-01 מוסיף מחברת בעברית, מהדאטה המאומת לשורת פיצ׳רים לפני המשחק. כל תא קוד מלווה בהסבר קצר ובפלט גלוי.

## המימוש וההיקף

- עונות 8–25 בלבד; ללא טעינת עונות ההחזקה, כתיבה לדאטה הנקי, Excel או אימון.
- preview_pre_match_row משתמש בחלון עשרת המשחקים שתאריכם קודם ממש למשחק. משחקים באותו זמן, המשחק עצמו והעתיד מוחרגים. ממוצעי בית/חוץ מפצלים את אותו חלון, ומחצית חסרה או סותרת אינה מושלמת.
- בנצ׳מרק B0: שיעורי תוצאות כוללים ולפי עונה, עם רווחי Wilson נפרדים של 95%. השיעורים הכוללים מתארים את התקופה; בבקטסט נדרש לחשב מאימון בלבד.
- דגלי קדנציות: כיסוי שמתחיל מאוחר וריצוד קצר A→B→A. אלה דגלי איכות בדיעבד, לא פיצ׳רים טרום־משחק. אין שינוי בזהויות מאמנים או מחיקת שורות.
- filter של nbstripout מנקה רק את העותק הנשמר ב־Git. ה־hook מאמת את העותק המוכן ל־commit ללא שינוי בקובץ המקומי. טסטי הרגרסיה בודקים גם עותק נקי וגם עותק עם פלטים; בדיקת kernel/Plotly הקיימת נשמרה.

## Restart & Run All

- 31 תאי Markdown ו־29 תאי קוד: 60 תאים, 18 חלקים ושישה גרפי Plotly.
- ריצה מלאה ב־kernel של ה־venv: 161.23 שניות.
- הקובץ המקומי נפתח מחדש לבדיקה: לכל תא קוד יש פלט ומספר הרצה, שש תצוגות Plotly נשמרו, ולטבלאות נשמרו פלטי HTML וטקסט. כל 21 האזהרות וטבלת 18 העונות מוצגות במלואן.
- גם אחרי commit הקובץ המקומי נשאר זהה לפי SHA256; בעותק ב־Git אין פלטים או מספרי הרצה.
- דוח ההרצה המקומי: data/reports/eda01c_execution.json.

## דגלי מאמנים לפי עונת תחילת הקדנציה

```text
         is_left_censored  is_flicker
2009/10                16           7
2010/11                 0          11
2011/12                 0           0
2012/13                 0           1
2013/14                 1           1
2014/15                 0           1
2015/16                 0           0
2016/17                 0           1
2017/18                 0           1
2018/19                 0           2
2019/20                 0           3
2020/21                 0           3
2021/22                 0           2
2022/23                 0           9
2023/24                 0           6
```

בסך הכול: 17 קדנציות עם תחילת כיסוי מאוחרת ו־48 ריצודים. הספירות הן תיאור של איכות המקור, ללא תיקון אוטומטי.

## מהדאטה למודל: משחק 793896

```text
                             date home_team_name away_team_name  home_goals  \
game_id
793896  2023-02-25 20:00:00+02:00      מכבי חיפה      בית"ר י-ם           2

         away_goals  ht_home_goals  ht_away_goals  home_red_cards  \
game_id
793896            1              1              1               0

         away_red_cards
game_id
793896                1
```

שורת הפיצ׳רים (נתחי השווי מוצגים בעיגול של שש ספרות):

```text
game_id                               ערך פיצ׳ר
feature_home_goals_scored_mean              1.5
feature_home_goals_conceded_mean            0.9
feature_home_points_earned_mean             2.0
feature_home_ht_goals_scored_mean           0.5
feature_home_home_points_per_match          2.6
feature_home_away_points_per_match          1.4
feature_home_prev_season_value_share   0.120988
feature_away_goals_scored_mean              2.1
feature_away_goals_conceded_mean            1.4
feature_away_points_earned_mean             1.8
feature_away_ht_goals_scored_mean           1.2
feature_away_home_points_per_match          2.0
feature_away_away_points_per_match          1.6
feature_away_prev_season_value_share   0.069146
```

היעדים בפועל, המוצגים בנפרד מהפיצ׳רים:

```text
game_id           יעד בפועל
target_home_goals         2
target_away_goals         1
target_result      home_win
```

## בדיקות

- pytest: 248 טסטים עברו; ruff check, ruff format --check ו־mypy ligat עברו.
- טסטים לדליפה, גבול זמן קשיח, היסטוריה ריקה, שיעורי Wilson, תחילת כיסוי מאוחרת, אורך ריצוד ושמירת הפלטים המקומיים.
- בדיקות הבריאות עברו עם אותן 21 אזהרות מתועדות.
- עודכנו EDA-01, מילון הדאטה, README המחברות וההנחיות הטכניות.
- עצירה לסקירת QA של קלוד. ללא מיזוג או ענף נוסף.
