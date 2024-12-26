import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

model = linear_model.LinearRegression()

df = pd.read_csv("data\\homeprices_onehoten.csv")

y = df.pop("price")


dummies = pd.get_dummies(df.town, dtype=int)
print(dummies)
df = pd.concat([df, dummies], axis="columns")
df = df.drop(["west windsor", "town"], axis="columns")
print(df)


model.fit(df, y)

print(model.predict(pd.DataFrame([[3400, 0, 0]], columns=["area", "monroe township", "robinsville"])))
print(model.score(df, y))