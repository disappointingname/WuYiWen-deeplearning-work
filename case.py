import pandas as pd
import numpy as np
import gc
import time
import re
import datetime
import warnings
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

def process_employment_length(value):
    if pd.isna(value) or str(value).lower() in ['nan', 'n/a', 'null']:
        return np.nan
    if isinstance(value, (int, float)):
        return min(max(value, 0), 50)
    value = str(value).lower()
    mapping = {
        '<1 year': 0.5, '< 1 year': 0.5,
        '1 year': 1.0, '2 years': 2.0,
        '3 years': 3.0, '4 years': 4.0,
        '5 years': 5.0, '6 years': 6.0,
        '7 years': 7.0, '8 years': 8.0,
        '9 years': 9.0, '10+years': 11.0,
        '10+ years': 11.0, '15+ years': 16.0,
        '20+ years': 21.0
    }
    if value in mapping:
        return mapping[value]
    match = re.search(r'(\d+)', value)
    return float(match.group(1)) if match else np.nan

def extract_year(value):
    try:
        match = re.search(r'(\d{4})', str(value))
        return int(match.group(1)) if match else np.nan
    except:
        return np.nan

class RobustFeatureEngineer:
    def __init__(self):
        self.imputation_values = {}
        self.scaler = None
        self.categorical_cols = ['grade', 'subGrade', 'purpose', 'regionCode', 'homeOwnership']
        self.label_encoders = {}

    def fit(self, df):
        numeric_cols = self._get_numeric_cols(df)
        for col in numeric_cols:
            if df[col].isna().any():
                self.imputation_values[col] = df[col].median()
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str).fillna('unknown'))
                self.label_encoders[col] = le

    def add_stat_features(self, df):
        if 'annualIncome' in df and 'loanAmnt' in df:
            df['income_loan_ratio'] = df['annualIncome'] / (df['loanAmnt'] + 1)
        if 'revolBal' in df and 'annualIncome' in df:
            df['revol_income_ratio'] = df['revolBal'] / (df['annualIncome'] + 1)
        if 'openAcc' in df and 'totalAcc' in df:
            df['open_total_ratio'] = df['openAcc'] / (df['totalAcc'] + 1)
        if 'dti' in df and 'annualIncome' in df:
            df['dti_income'] = df['dti'] * df['annualIncome']
        if 'interestRate' in df:
            df['interest_bin'] = pd.qcut(df['interestRate'], 15, labels=False, duplicates='drop')
        for col in ['regionCode', 'grade', 'purpose']:
            if col in df and 'annualIncome' in df:
                group_mean = df.groupby(col)['annualIncome'].transform('mean')
                df[f'{col}_income_diff'] = df['annualIncome'] - group_mean
            if col in df and 'loanAmnt' in df:
                group_mean = df.groupby(col)['loanAmnt'].transform('mean')
                df[f'{col}_loan_diff'] = df['loanAmnt'] - group_mean
        if 'annualIncome' in df:
            df['income_rank'] = df['annualIncome'].rank(pct=True)
        if 'loanAmnt' in df:
            df['loan_rank'] = df['loanAmnt'].rank(pct=True)
        return df

    def add_interaction_features(self, df):
        if 'grade_enc' in df and 'purpose_enc' in df:
            df['grade_purpose'] = df['grade_enc'] * 10 + df['purpose_enc']
        if 'regionCode_enc' in df and 'homeOwnership_enc' in df:
            df['region_home'] = df['regionCode_enc'] * 10 + df['homeOwnership_enc']
        if 'fico_avg' in df and 'interestRate' in df:
            df['fico_interest'] = df['fico_avg'] * df['interestRate']
        if 'annualIncome' in df and 'dti' in df:
            df['income_dti'] = df['annualIncome'] * df['dti']
        if 'income_loan_ratio' in df and 'dti' in df:
            df['income_loan_dti'] = df['income_loan_ratio'] * df['dti']
        if 'revol_income_ratio' in df and 'fico_avg' in df:
            df['revol_fico'] = df['revol_income_ratio'] * df['fico_avg']
        return df

    def transform(self, df):
        df = df.copy().drop(columns=['id'], errors='ignore')
        df = self._process_dates(df)
        if 'employmentLength' in df.columns:
            df['employmentLength'] = df['employmentLength'].apply(process_employment_length)
            df['employmentLength'] = df['employmentLength'].fillna(df['employmentLength'].median())
        if 'dti' in df and 'annualIncome' in df:
            df['debt_burden'] = df['dti'] * df['annualIncome'] / 1000
        if 'annualIncome' in df and 'loanAmnt' in df:
            df['income_to_loan'] = df['annualIncome'] / (df['loanAmnt'] + 1)
        if 'ficoRangeLow' in df and 'ficoRangeHigh' in df:
            df['fico_avg'] = (df['ficoRangeLow'] + df['ficoRangeHigh']) / 2
        if 'revolUtil' in df:
            df['high_utilization'] = (df['revolUtil'] > 80).astype(int)
        if 'openAcc' in df and 'totalAcc' in df:
            df['closed_account_ratio'] = (df['totalAcc'] - df['openAcc']) / (df['totalAcc'] + 1)
        for col, value in self.imputation_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        for col in self.categorical_cols:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le is not None:
                    df[col] = df[col].astype(str).fillna('unknown')
                    unknown_mask = ~df[col].isin(le.classes_)
                    if unknown_mask.any():
                        df.loc[unknown_mask, col] = 'unknown'
                        if 'unknown' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'unknown')
                    df[f'{col}_enc'] = le.transform(df[col])
                    df.drop(col, axis=1, inplace=True)
        numeric_cols = self._get_numeric_cols(df)
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
            q1, q3 = df[col].quantile([0.05, 0.95])
            if q3 > q1:
                iqr = q3 - q1
                lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df[col] = np.clip(df[col], lb, ub)
        if len(numeric_cols) > 0:
            if self.scaler is None:
                self.scaler = StandardScaler()
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            for col in object_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = pd.factorize(df[col])[0]
        non_numeric = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            df = df.drop(columns=non_numeric)
        df = self.add_stat_features(df)
        df = self.add_interaction_features(df)
        return df.fillna(0)

    def _get_numeric_cols(self, df):
        return [col for col in df.select_dtypes(include=np.number).columns if col not in ['isDefault', 'y']]

    def _process_dates(self, df):
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df.drop(col, axis=1, inplace=True)
                    else:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df.drop(col, axis=1, inplace=True)
                except:
                    df[col] = df[col].apply(extract_year)
        for col in df.columns:
            if 'credit' in col.lower():
                df[col] = df[col].apply(extract_year)
                if col.lower() in ['earliestcreditline', 'earliescreditline']:
                    current_year = datetime.datetime.now().year
                    df['credit_history_years'] = current_year - df[col]
                    df.drop(col, axis=1, inplace=True)
        return df

    def select_important_features(self, X, y, threshold='median'):
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        rf.fit(X, y)
        selector = SelectFromModel(rf, threshold=threshold, prefit=True)
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features


def robust_cross_validation(X, y, test_df, n_folds=5):
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{n_folds}")
        start_time = time.time()
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        fe = RobustFeatureEngineer()
        fe.fit(X_train)
        X_train_fe = fe.transform(X_train)
        X_val_fe = fe.transform(X_val)
        test_fold_fe = fe.transform(test_df)
        selected_features = fe.select_important_features(X_train_fe, y_train, threshold='median')
        X_train_fe = X_train_fe[selected_features]
        X_val_fe = X_val_fe[selected_features]
        test_fold_fe = test_fold_fe[selected_features]
        base_models = [
            ('lgb', lgb.LGBMClassifier(device='gpu', gpu_platform_id=0, gpu_device_id=0, n_estimators=800, learning_rate=0.05, random_state=42)),
            ('xgb', xgb.XGBClassifier(tree_method='hist', device='cuda', n_estimators=800, learning_rate=0.05, random_state=42, use_label_encoder=False)),
            ('cb', cb.CatBoostClassifier(task_type='GPU', iterations=800, learning_rate=0.05, random_state=42, silent=True))
        ]
        stack_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=200),
            n_jobs=1,
            passthrough=True
        )
        stack_model.fit(X_train_fe, y_train)
        val_preds = stack_model.predict_proba(X_val_fe)[:, 1]
        oof_preds[val_idx] = val_preds
        test_fold_preds = stack_model.predict_proba(test_fold_fe)[:, 1]
        test_preds += test_fold_preds / n_folds
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))
        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}, Acc: {fold_acc:.5f}, Time: {time.time() - start_time:.1f}s")
    return test_preds, oof_preds
