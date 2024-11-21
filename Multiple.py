import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("./datasets/FuelConsumptionCo2.csv")
print(df.head())

print(df.columns)
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(5))

#code showcases the scatter plot between engine size and co2 emissions
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='purple')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# let's check the train data distribution

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#plt.show()

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train['CO2EMISSIONS'])
regr.fit(x, y)
#Print the coefficents
print("coeffiecents : ",regr.coef_)


#Ordinary least square
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test['CO2EMISSIONS'])
print("Mean Squared Error (MSE) : %.2f"  % np.mean((y_hat - y) ** 2))
var = regr.score(x, y)
print('varience: ', var)