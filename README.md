# House Prices - (Kaggle)

## 概要

このリポジトリは、Kaggle の「House Prices」コンペティションへの初参加時のコードをまとめたものです。  
初めてのデータ分析・Kaggle 挑戦として、データの前処理から特徴量エンジニアリング、モデル構築、提出までの一連の流れを実装しています。

- 1 回目の提出スコア: **0.14873**
- 今後はハイパーパラメータチューニングや特徴量の追加・選択などでスコア向上を目指します

## 特徴

- **前処理**: 欠損値補完、カテゴリ変数の処理、特徴量エンジニアリング（築年数・リフォーム有無・季節など）
- **特徴量選択**: CatBoost による Feature Importance と Greedy Feature Selection を組み合わせて重要な特徴量を選択
- **モデル**: CatBoostRegressor を用いた回帰モデル
- **評価**: RepeatedKFold による交差検証で安定した評価
- **提出**: テストデータに対して予測し、Kaggle 提出用 CSV を出力

## ファイル構成

- `main.py` : データ前処理・特徴量エンジニアリング・モデル学習・予測・提出までの一連のコード
- `README.md` : 本ファイル

## 今後の改善予定

- ハイパーパラメータチューニング（CatBoost のパラメータ最適化）
- 新たな特徴量の追加や選択手法の工夫
- アンサンブル学習の導入
- EDA（探索的データ分析）の充実

## 参考

- [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---
