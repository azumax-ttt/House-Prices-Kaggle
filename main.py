import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score

# 学習データで前処理、特徴スケーリング

df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# =====================================
# 学習データで前処理、特徴スケーリング¶
# =====================================

# YearBuiltが0のものを削除する
df = df[df['YearBuilt'] != 0]

# 欠損値を埋める
meaningful_missing_cols = [
    'Alley',
    'FireplaceQu',
    'Fence',
    'PoolQC',
    'MiscFeature',
    'MasVnrType',
    'BsmtFinType2',
    'BsmtCond',
    'BsmtQual',
    'BsmtFinType1',
    'BsmtExposure',
    'BsmtFinType2',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageCond',
    'GarageQual',
]

df[meaningful_missing_cols] = df[meaningful_missing_cols].fillna('None')

df.loc[(df['MasVnrType'] == 'None') & (df['MasVnrArea'] > 0), 'MasVnrType'] = 'Unknown'
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
imputer = SimpleImputer(strategy='most_frequent')
df[['Electrical']] = imputer.fit_transform(df[['Electrical']])

imputer = SimpleImputer(strategy ="constant", fill_value=0)
df[['GarageYrBlt']] = imputer.fit_transform(df[['GarageYrBlt']])

# 数値カラムをカテゴリカル変数に変更する
df['MSSubClass'] = df['MSSubClass'].astype('category')

# LotFrontageをknnで欠損値代入
target = 'LotFrontage'
X = df.drop([target, 'Id'], axis=1)
y = df[target]

# 数値カラムのリスト取得(標準化の対象)
num_cols = X.select_dtypes(include=np.number).columns.to_list()

# ダミー変数
X = pd.get_dummies(X, drop_first=True)
# 標準化
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# LotFrontageがNaNのデータはテストデータ，そうでなければ学習データ
test_indexes = df[df[target].isna()].index
train_indexes = df[~df[target].isna()].index
X_train, X_test = X.loc[train_indexes], X.loc[test_indexes]
y_train, y_test = y.loc[train_indexes], y.loc[test_indexes]

# kNNのモデルを作って予測値を代入する
knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
y_pred = knn.predict(X_test)
df.loc[test_indexes, target] = y_pred

# MSSubClassはカテゴリ化してダミー変数化する
df['MSSubClass'] = df['MSSubClass'].astype('category')
X_scaled = df.drop(['Id', 'SalePrice'], axis=1)

# 月日に関する特徴量エンジニアリング
# 築年数を新しいカラムとして作成
X_scaled['HouseAge'] = X_scaled['YrSold'] - X_scaled['YearBuilt']
# リフォーム後の年数カラム作成
X_scaled['RemodAge'] = X_scaled['YrSold'] - X_scaled['YearRemodAdd']
# リフォーム済みかのカラム作成
X_scaled['IsRemodeled'] = (X_scaled['YearBuilt'] != X_scaled['YearRemodAdd'])
# ガレージ年数カラム作成
X_scaled['GarageAge'] = X_scaled['YrSold'] - X_scaled['GarageYrBlt']
# 売却時の季節カラム作成
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"

X_scaled["SeasonSold"] = X_scaled["MoSold"].apply(get_season)

# ================================================-
# 学習データで特徴量選択
# =================================================

## RepeatedKFold × CatBoostのFeatureImportanceで特徴量選択
y = df['SalePrice']
k = 5
n_repeats = 3
cat_cols = X_scaled.select_dtypes(exclude=np.number).columns.to_list()

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

feature_importances = np.zeros(X_scaled.shape[1])

for fold, (train_idx, valid_idx) in enumerate(cv.split(X_scaled, y)):
    print(f"Fold {fold+1}")
    X_train, X_valid = X_scaled.iloc[train_idx], X_scaled.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=0
    )

    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

    feature_importances += model.get_feature_importance()

# 平均化
feature_importances /= cv.get_n_splits()

importance_df = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

print(importance_df)

# GreedyFeatureSelectionを使って特徴量選択
class GreedyFeatureSelection():
    
    def __init__(self, pipeline, cv):
        self.pipeline = pipeline
        self.cv = cv
        self.selected_features = []
        self.scores = [0]
    
    def select_feature(self, X, y):
        
        all_features = X.columns
        
        while True:
            # print('greedy selection started')
            best_score = self.scores[-1]
            candidate_feature = None
            for feature in all_features:
                if feature in self.selected_features:
                    continue
                # print(f'{feature} started')
                features = self.selected_features + [feature]
                X_train = X[features]
                # 評価
                score = cross_val_score(self.pipeline, X_train, y, cv=self.cv).mean()
                # print(f'{features} score: {score}')
                if score > best_score:
                    # print(f'best score updated {best_score} -> {score}')
                    best_score = score
                    candidate_feature = feature
            
            if candidate_feature is not None:
                # print(f'========{candidate_feature} is selected=============')
                self.scores.append(best_score)
                self.selected_features.append(candidate_feature)
            else:
                break

pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('model', LinearRegression())
])
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
importace_cols = importance_df.head(30)['feature'].to_list()
importance_X = X_scaled[importace_cols]
importance_X = pd.get_dummies(importance_X, drop_first=True)

# Greedy feature selection

gfs = GreedyFeatureSelection(pipeline=pipeline, cv=cv)
gfs.select_feature(importance_X, y)
# スコアの結果と選択された特徴量を確認
print(gfs.scores)
print(gfs.selected_features)

plt.plot(range(len(gfs.scores)), gfs.scores)
plt.xlabel("Number of Features")
plt.ylabel("CV Score")
plt.title("Greedy Feature Selection Progress")
plt.show()


# 10個・15個・20個の特徴量でCatBoostを学習してCVスコア比較
importace_cols = importance_df.head(10)['feature'].to_list()
importance_X = X_scaled[importace_cols]
cat_cols = importance_X.select_dtypes(exclude=np.number).columns.to_list()

cv = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=42)
model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=0
    )

scores = cross_val_score(model, importance_X, y, cv=cv, scoring='neg_mean_squared_error')

rmse_scores = np.sqrt(-scores)
print(f"RMSE({k}FoldCV): {rmse_scores.mean():.0f}")
print(f"std: {rmse_scores.std():.0f}")

# 15個の特徴量でCatBoostを学習してCVスコア比較
importace_cols = importance_df.head(15)['feature'].to_list()
importance_X = X_scaled[importace_cols]
cat_cols = importance_X.select_dtypes(exclude=np.number).columns.to_list()

cv = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=42)
model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=0
    )

scores = cross_val_score(model, importance_X, y, cv=cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"RMSE({k}FoldCV): {rmse_scores.mean():.0f}")
print(f"std: {rmse_scores.std():.0f}")

# 20個の特徴量でCatBoostを学習してCVスコア比較
importace_cols = importance_df.head(15)['feature'].to_list()
importance_X = X_scaled[importace_cols]
cat_cols = importance_X.select_dtypes(exclude=np.number).columns.to_list()

cv = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=42)
model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=0
    )

scores = cross_val_score(model, importance_X, y, cv=cv, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)
print(f"RMSE({k}FoldCV): {rmse_scores.mean():.0f}")
print(f"std: {rmse_scores.std():.0f}")

# 15個の特徴量でCatBoostでモデル学習学習
importace_cols = importance_df.head(15)['feature'].to_list()
importance_X = X_scaled[importace_cols]
cat_cols = importance_X.select_dtypes(exclude=np.number).columns.to_list()


model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        cat_features=cat_cols,
        early_stopping_rounds=50,
        verbose=0
    )
model.fit(importance_X, y)

# =============================================
# テストデータの前処理と特徴スケーリング
# =============================================

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_df.info()

# 前処理
# 欠損値を埋める
meaningful_missing_cols = [
    'Alley',
    'FireplaceQu',
    'Fence',
    'PoolQC',
    'MiscFeature',
    'MasVnrType',
    'BsmtFinType2',
    'BsmtCond',
    'BsmtQual',
    'BsmtFinType1',
    'BsmtExposure',
    'BsmtFinType2',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageCond',
    'GarageQual',
]

test_df[meaningful_missing_cols] = test_df[meaningful_missing_cols].fillna('None')

imputer = SimpleImputer(strategy='most_frequent')
test_df[['MSZoning']] = imputer.fit_transform(test_df[['MSZoning']])
test_df[['Utilities']] = imputer.fit_transform(test_df[['Utilities']])
test_df[['Exterior1st', 'Exterior2nd']] = imputer.fit_transform(test_df[['Exterior1st', 'Exterior2nd']])
test_df[['Functional']] = imputer.fit_transform(test_df[['Functional']])
test_df[['KitchenQual']] = imputer.fit_transform(test_df[['KitchenQual']])
test_df[['SaleType']] = imputer.fit_transform(test_df[['SaleType']])

test_df.loc[(test_df['MasVnrType'] == 'None') & (test_df['MasVnrArea'] > 0), 'MasVnrType'] = 'Unknown'
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(0)
test_df['GarageCars'] = test_df['GarageCars'].fillna(0)
test_df['GarageArea'] = test_df['GarageArea'].fillna(0)

imputer = SimpleImputer(strategy ="constant", fill_value=0)
test_df[['GarageYrBlt']] = imputer.fit_transform(test_df[['GarageYrBlt']])
test_df[['BsmtFullBath', 'BsmtHalfBath']] = imputer.fit_transform(test_df[['BsmtFullBath', 'BsmtHalfBath']])

bsm_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
test_df[bsm_cols] = imputer.fit_transform(test_df[bsm_cols])

# 数値カラムをカテゴリカル変数に変更する
test_df['MSSubClass'] = test_df['MSSubClass'].astype('category')

test_df.info()

# LotFrontageをknnで欠損値代入
target = 'LotFrontage'
test_X = test_df.drop([target, 'Id'], axis=1)
test_y = test_df[target]

# 数値カラムのリスト取得(標準化の対象)
num_cols = test_X.select_dtypes(include=np.number).columns.to_list()

# ダミー変数
test_X = pd.get_dummies(test_X, drop_first=True)
# 標準化
test_X[num_cols] = scaler.fit_transform(test_X[num_cols])

# LotFrontageがNaNのデータはテストデータ，そうでなければ学習データ
test_indexes = test_df[test_df[target].isna()].index
train_indexes = test_df[~test_df[target].isna()].index
X_train, X_test = test_X.loc[train_indexes], test_X.loc[test_indexes]
y_train, y_test = test_y.loc[train_indexes], test_y.loc[test_indexes]

# kNNのモデルを作って予測値を代入する
knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
y_pred = knn.predict(X_test)
test_df.loc[test_indexes, target] = y_pred
test_X.info()

# MSSubClassはカテゴリ化してダミー変数化する
test_df['MSSubClass'] = test_df['MSSubClass'].astype('category')
test_X_scaled = test_df.drop(['Id'], axis=1)

# 月日に関する特徴量エンジニアリング
# 築年数を新しいカラムとして作成
test_X_scaled['HouseAge'] = test_X_scaled['YrSold'] - test_X_scaled['YearBuilt']
# リフォーム後の年数カラム作成
test_X_scaled['RemodAge'] = test_X_scaled['YrSold'] - test_X_scaled['YearRemodAdd']
# リフォーム済みかのカラム作成
test_X_scaled['IsRemodeled'] = (test_X_scaled['YearBuilt'] != test_X_scaled['YearRemodAdd'])
# ガレージ年数カラム作成
test_X_scaled['GarageAge'] = test_X_scaled['YrSold'] - test_X_scaled['GarageYrBlt']
# 売却時の季節カラム作成
def get_season(month):
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"

test_X_scaled["SeasonSold"] = test_X_scaled["MoSold"].apply(get_season)

# =================================
# テストデータでpredict
# =================================
importance_test_X = test_X_scaled[importace_cols]
y_pred = model.predict(importance_test_X)

submission = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': y_pred
})

submission.to_csv('submission.csv', index=False)