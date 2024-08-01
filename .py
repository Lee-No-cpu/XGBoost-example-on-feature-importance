import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
# Load the data
data_path = 'Your datapath'
train = pd.read_csv(data_path + 'yourfle.csv')

# Define features and target
features = ['var_1','var_2', 'var_3', 'var_4']
target = 'Tar_1'

X = train[features]
y = train[target]

# Remove rows where y is NaN or infinite
mask = y.isna() | np.isinf(y)
X = X.loc[~mask]
y = y.loc[~mask]

# Define the parameter grid
param_grid = {
    'max_depth': [6,8, 10,12, 15],
    'n_estimators': [100, 500, 1000, 1500],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Create a base model
xgb = XGBRegressor(random_state=20220)

# Instantiate the grid search model
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='neg_mean_squared_error'  # Use negative MSE as the scoring metric
)

# Fit the grid search to the data
grid_search.fit(X, y)

# Save the best parameters
best_params = grid_search.best_params_
df_best_params = pd.DataFrame([best_params])
df_best_params.to_csv(data_path + 'best_parameters.csv', index=False)

# Train the model with the best parameters
xgb_best = XGBRegressor(**best_params, random_state=20220)
xgb_best.fit(X, y)

y_pred_best = xgb_best.predict(X)
r2_best = r2_score(y, y_pred_best)


xgb_new = XGBRegressor(max_depth=9, colsample_bytree=0.8, learning_rate=0.05,
                       min_child_weight=1, subsample=0.7, n_estimators=2000,
                       random_state=20220)
xgb_new.fit(X, y)
y_pred_new = xgb_new.predict(X)
r2_new = r2_score(y, y_pred_new)


plt.figure(figsize=(12, 6))


plt.scatter(y, y_pred_best, alpha=0.5, label=f'Best Parameters (R² = {r2_best:.4f})')
m_best, b_best = np.polyfit(y, y_pred_best, 1)
plt.plot(y, m_best*y + b_best, color='blue')



plt.title('Comparison of Model Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

feature_importances_best = xgb_best.feature_importances_
df_feature_importance_best = pd.DataFrame({'Feature': features, 'Importance': feature_importances_best})
df_feature_importance_best.to_csv('feature_importance_best.csv', index=False)



# Save actual vs predicted values for the best parameters model
df_actual_vs_predicted_best = pd.DataFrame({'Actual': y, 'Predicted_Best': y_pred_best})
df_actual_vs_predicted_best.to_csv('actual_vs_predicted_best.csv', index=False)

df_feature_importance_best.to_csv('E:/PCM article/PCM_ZLW/Division/' + 'feature_importance_best.csv', index=False)

