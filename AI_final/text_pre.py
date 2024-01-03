import pandas as pd

def setSex(sex):
    if sex == 'Male':
        return 0.0
    return 1.0

def setAge(age):
    if age >= 0 and age < 10:
        return 0.0
    elif age >= 10 and age < 20:
        return 1.0
    elif age >= 20 and age < 30:
        return 2.0
    elif age >= 30 and age < 40:
        return 3.0
    elif age >= 40 and age < 50:
        return 4.0
    elif age >= 50 and age < 60:
        return 5.0
    elif age >= 60 and age < 70:
        return 6.0
    elif age >= 70 and age < 80:
        return 7.0
    elif age >= 80 and age < 90:
        return 8.0
    else:
        return 9.0
import pandas as pd

def onehot(df):
    sex_encoded = pd.get_dummies(df['sex'], prefix='sex')
    age_encoded = pd.get_dummies(df['age'], prefix='age')

    new_df = pd.concat([df['ID'], sex_encoded, age_encoded], axis=1)

    return new_df
