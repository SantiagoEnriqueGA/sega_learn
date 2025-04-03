import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.linear_models import Lasso, OrdinaryLeastSquares, Ridge
from sega_learn.utils import Metrics, make_regression
from sega_learn.utils.voting import VotingRegressor

r2_score = Metrics.r_squared

# Create a synthetic dataset
X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)

# Create and fit the models
ols = OrdinaryLeastSquares(fit_intercept=True)
ols.fit(X, y)

lasso = Lasso(alpha=1.0, fit_intercept=True)
lasso.fit(X, y)

ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(X, y)

# Create the voter
voter = VotingRegressor(models=[ols, lasso, ridge], model_weights=[0.3, 0.3, 0.4])

print(f"Voter R^2 Score: {r2_score(y, voter.predict(X)):.2f}")
voter.show_models(formula=True)
