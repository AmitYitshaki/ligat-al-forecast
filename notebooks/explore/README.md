# מחברות חקירה

כאן נשמרות מחברות חקירה ו־EDA. לוגיקה זמנית מותרת כדי לבדוק רעיונות.
חישוב שמכריע החלטה עובר ל־`ligat/analysis/`, עם טסט שמאמת אותו.
מחברות ההצגה מחוץ לתיקייה זו מייבאות לוגיקה מהחבילה ואינן מיישמות אותה בעצמן.

לפני commit: Restart & Run All בסביבת ה־venv. פלטים ומספרי הרצה מנוקים
באמצעות nbstripout, גם ב־Git filter וגם ב־pre-commit.

להוספת kernel של ה־venv ל־Jupyter:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name ligat-al-forecast --display-name "Python (ligat-al-forecast venv)"
```

בחרו את ה־kernel הזה למחברות הפרויקט. שלוש ספריות המחברות הן תלויות dev בלבד.
ב־clone חדש יש להתקין את ה־filter המקומי מתוך ה־venv:

```powershell
.\.venv\Scripts\nbstripout.exe --install --attributes .gitattributes
```
