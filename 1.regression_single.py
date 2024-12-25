import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

model = linear_model.LinearRegression()

df = pd.read_csv("data/homeprices.csv")

print(type(df["area"]), type(df["price"]), type(df[["area"]]))

plt.scatter(1,2,10)
# plt.show()

model.fit(df[["area"]], df["price"])

print(model.coef_)
print(model.intercept_)

# print(model.predict([[3300]]))
print(model.predict(pd.DataFrame([[3300]], columns=["area"])))

# df = pd.read_csv("data\\area.csv")
# ans = model.predict(df)
# print(ans)