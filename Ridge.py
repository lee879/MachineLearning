import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

np.random.seed(666)
x = np.random.uniform(-3,3,size=100)
x = x.reshape(-1,1)
y = 0.5 * x  + 3. + np.random.normal(size=(100,1))

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=666)

def Poly_rige(degree,alpha):
    return Pipeline([
    ("poly",PolynomialFeatures(degree=degree)),
    ("dt_scalar",StandardScaler()),
    ("rige_reg",Ridge(alpha)) #这里使用的是岭回归

])
def Poly_line(degree):
    return Pipeline([
    ("poly",PolynomialFeatures(degree=degree)),
    ("dt_scalar",StandardScaler()),
    ("line_reg",LinearRegression()) #这里使用的线性回归
])
def plot_model(model):
    x_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_plot = model.predict(x_plot)

    plt.scatter(x, y)
    plt.plot(x_plot[:, 0], y_plot, color="r")
    plt.show()

poly_line = Poly_line(degree=20)
poly_line.fit(x_train,y_train)
p = poly_line.predict(x_test)
print(mean_squared_error(y_test,p))
#plot_model(poly_line)

#使用岭回归(正则化) 让曲线更加的平缓
poly_rige = Poly_rige(degree=20,alpha=50)
poly_rige.fit(x_train,y_train)
p = poly_rige.predict(x_test)
print(mean_squared_error(y_test,p))
plot_model(poly_rige)







