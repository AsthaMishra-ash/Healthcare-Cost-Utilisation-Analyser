import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import os

def train_and_save():
    base = os.path.dirname(os.path.abspath(__file__))
    df   = pd.read_csv(os.path.join(base, '../data/cms_inpatient.csv'))

    # Encode categoricals
    le_state    = LabelEncoder()
    le_drg      = LabelEncoder()
    le_category = LabelEncoder()

    df['state_enc']    = le_state.fit_transform(df['provider_state'])
    df['drg_enc']      = le_drg.fit_transform(df['drg_definition'])
    df['category_enc'] = le_category.fit_transform(df['drg_category'])

    features = ['state_enc', 'category_enc', 'drg_enc',
                'total_discharges', 'average_covered_charges', 'cost_to_charge_ratio']
    X = df[features]
    y = df['average_total_payments']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    results = {
        'lr_r2':   round(r2_score(y_test, lr_preds), 4),
        'lr_rmse': round(np.sqrt(mean_squared_error(y_test, lr_preds)), 2),
        'lr_mae':  round(mean_absolute_error(y_test, lr_preds), 2),
        'rf_r2':   round(r2_score(y_test, rf_preds), 4),
        'rf_rmse': round(np.sqrt(mean_squared_error(y_test, rf_preds)), 2),
        'rf_mae':  round(mean_absolute_error(y_test, rf_preds), 2),
        'features': features,
        'feature_importance': dict(zip(features, rf.feature_importances_.tolist())),
        'y_test':   y_test.tolist(),
        'rf_preds': rf_preds.tolist(),
        'lr_preds': lr_preds.tolist(),
    }

    # Save
    model_dir = base
    with open(os.path.join(model_dir, 'rf_model.pkl'),  'wb') as f: pickle.dump(rf, f)
    with open(os.path.join(model_dir, 'lr_model.pkl'),  'wb') as f: pickle.dump(lr, f)
    with open(os.path.join(model_dir, 'results.pkl'),   'wb') as f: pickle.dump(results, f)
    with open(os.path.join(model_dir, 'encoders.pkl'),  'wb') as f:
        pickle.dump({'state': le_state, 'drg': le_drg, 'category': le_category}, f)

    print("✅ Models trained and saved.")
    print(f"Linear Regression — R²: {results['lr_r2']}, RMSE: ${results['lr_rmse']:,.0f}")
    print(f"Random Forest     — R²: {results['rf_r2']}, RMSE: ${results['rf_rmse']:,.0f}")
    return results

if __name__ == '__main__':
    train_and_save()
