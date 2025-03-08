import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.linear_models import RANSAC
from sega_learn.utils import make_regression
from sega_learn.utils import Metrics
r2_score = Metrics.r_squared

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = RANSAC(n=20, k=300, t=0.01, d=10, model=None, 
                 auto_scale_t=True, scale_t_factor=2,
                 auto_scale_n=False, scale_n_factor=2                
                 )
reg.fit(X, y)

r2 = round(r2_score(y, reg.predict(X)), 2)
coef = [round(c, 2) for c in reg.best_fit.coef_]
intercept = round(reg.best_fit.intercept_, 2)
formula = reg.get_formula()

print(f"R^2 Score: {r2}")
print(f"Regression Coefficients: {coef}")
print(f"Regression Intercept: {intercept}")
print(f"Regression Formula: {formula}")
