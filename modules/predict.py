import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import random

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim

from modules.constants import local_paths
import os
import pickle
import itertools


class PredBase:
  def __init__(
    self,
    returns_df: pd.DataFrame, 
    bet_type: str = 'umaren', 
    threshold: float = None, 
    stochastic_variation: bool = True
  ):
    
    self.returns_df = returns_df  # 払戻データの初期化
    self.bet_type = bet_type  # 賭け方の初期化
    self.threshold = threshold  # 閾値の初期化
    self.stochastic_variation = stochastic_variation  # 確率による賭け金額の調整の有無

  def process_missing_values(self, df):
    df_p = df.copy()
    """欠損値の補完"""
    # 数値カラムを選択
    num_cols = df_p.select_dtypes(include=['int64', 'float64']).columns
    # カテゴリカラムを選択
    cat_cols = df_p.select_dtypes(include=['object']).columns

    # 数値カラムの欠損値を平均値で補完
    for col in num_cols:
        df_p[col] = df_p[col].fillna(df_p[col].mean())
    # カテゴリカラムの欠損値を最頻値で補完
    for col in cat_cols:
      df_p[col] = df_p[col].fillna(df_p[col].mode().iloc[0])

    return df_p
  

  def preprocess_df(self, df, train=False):
    df_copy = df.copy()

    if train:
      if self.bet_type in ['umaren', 'umatan']:
        df_copy['target'] = df_copy['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
      elif self.bet_type in ['sanrenpuku','sanrentan']:
        df_copy['target'] = df_copy['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
      else:
        raise RuntimeError(f"{self.bet_type} is not supported.")
    
    # 欠損値の補完
    df_p = self.process_missing_values(df_copy)
    df_p_copy = df_p.copy()

    # カウントエンコーディング
    count_enc = ce.CountEncoder(cols=['date'])
    df_ce = count_enc.fit_transform(df_p_copy[['date']])
    
    # ラベルエンコーディング
    df_le = pd.DataFrame()
    for col in ['horse_id', 'jockey_id', 'trainer_id', 'owner_id']:
      le = LabelEncoder()
      df_le[col] = le.fit_transform(df_p_copy[col])


    # 元のデータフレームにエンコードされた列を追加
    df_p_encoded = df_p_copy.reset_index(drop=True).join(
      pd.concat([df_ce.reset_index(drop=True), df_le.reset_index(drop=True)], axis=1), rsuffix='_encoded'
    )

    return df_p_encoded.loc[:, ~df_p_encoded.columns.duplicated()]

  

  def drop_columns(self, df):
    df_p = df.copy()

    self.drop_features = ['race_id', 'rank', 'horse_id', 'jockey_id', 'trainer_id', 'owner_id', 'date', 'reference_date']

    # 予測に不要なカラムを削除
    df_p = df_p.drop(self.drop_features, axis=1, errors='ignore')

    return df_p

  
  
  def save_model(self, output_dir=local_paths.MODELS_DIR, modelname='model'):
    with open(os.path.join(output_dir ,f'{modelname}.pickle'), mode='wb') as f:
      pickle.dump(self.model, f, protocol=2)



  def calc_bet_amount(self, group):
    if self.stochastic_variation == True:
      # 確率に応じて賭け金額を調整
      for idx in group.index:
        proba = group.loc[idx, 'predicted_proba']
        if proba >= 0.9:
          bet_amount = 300
        elif proba >= 0.8:
          bet_amount = 200
        else:
            bet_amount = 100
    else:
      bet_amount = 100

    return bet_amount
  


  def process_bet_type_combinations(self, bet_type, predict_num, race_group):
    """馬連、馬単、三連複、三連単の組み合わせと払戻の種類を生成"""
    if bet_type == 'umaren' and predict_num >= 2:
      combinations = list(itertools.combinations(race_group.index, 2))
      bet_type_return = '馬連'
    elif bet_type == 'umatan' and predict_num >= 2:
      combinations = list(itertools.permutations(race_group.index, 2))
      bet_type_return = '馬単'
    elif bet_type == 'sanrenpuku' and predict_num >= 3:
      combinations = list(itertools.combinations(race_group.index, 3))
      bet_type_return = '三連複'
    elif bet_type == 'sanrentan' and predict_num >= 3:
      combinations = list(itertools.permutations(race_group.index, 3))
      bet_type_return = '三連単'

    return combinations, bet_type_return



  def calc_returns(self, race_group, df_add_returns, combinations, bet_type_return, rank_threshold):
    """組み合わせ数に応じて払い戻し額と賭け金額を計算"""
    bet_amount = self.calc_bet_amount(race_group)
    bet_sum = bet_amount * len(combinations)
    df_add_returns.loc[race_group.index, 'bet_sum'] = bet_sum

    # 正しい結果の場合、そうでない場合の払い戻し額を設定  
    correct = race_group[race_group['rank'] <= rank_threshold]
    if len(correct) == rank_threshold:
      returns_value = race_group[f'{bet_type_return}_returns'].iloc[0].replace("'", '').replace("[", '').replace("]", '')
      try:
        df_add_returns.loc[race_group.index, 'returns'] = int(returns_value)
      except ValueError:
        df_add_returns.loc[race_group.index, 'returns'] = np.nan
    else:
      df_add_returns.loc[race_group.index, 'returns'] = 0

    return df_add_returns
  

  
  def returns_against_pred_bet(self, pred_df):
    """予測に基づく払戻額を計算し、賭け金額を追加"""
    df = self.predict_target(pred_df)

    df = df.loc[df['predicted_target'] == 1, ['race_id', 'number', 'rank', 'predicted_target', 'predicted_proba']]

    df_add_returns = df.merge(self.returns_df, on='race_id', how='left')
    df_add_returns['returns'] = 0
    df_add_returns['bet_sum'] = 0  # 賭け金額カラムを追加

    # 各レースで払い戻し額と賭け金額を計算
    for race_id, race_group in df_add_returns.groupby('race_id'):
      predict_num = race_group['predicted_target'].sum()
      combinations = []  # 組み合わせを初期化
        

      if self.bet_type in ['umaren', 'umatan']:
        predict_min = 2

      elif self.bet_type in ['sanrenpuku', 'sanrentan']:
        predict_min = 3

      else:
        raise RuntimeError(f"{self.bet_type} is not supported.")
      
      # 予想が2つまたは3つ未満の場合スキップ
      if predict_num < predict_min:
        df_add_returns.loc[race_group.index, ['returns']] = np.nan
        continue

      # 組み合わせ数に応じて払い戻し額と賭け金額を計算
      else:     
        combinations, bet_type = self.process_bet_type_combinations(self.bet_type, predict_num, race_group)
        df_add_returns = self.calc_returns(race_group, df_add_returns, combinations, bet_type, predict_min)

    return df_add_returns[['race_id', 'returns', 'bet_sum']].drop_duplicates()
  

  def calc_returns_rate(self, pred_df):
    df_add_returns = self.returns_against_pred_bet(pred_df)
    df = df_add_returns.dropna(subset=['returns']).reset_index(drop=True)

    # 累積ベット金額と払い戻しを計算
    df['total_bet'] = df['bet_sum'].cumsum()
    df['total_returns'] = df['returns'].cumsum()
    df['returns_rate'] = df['total_returns'] / df['total_bet']

    return df


  def plot_returns_rate(self, pred_df):
    """回収率を計算し、グラフを生成"""
    df = self.calc_returns_rate(pred_df)

    # 賭けた回数と払い戻しの総額を計算
    betting_count = len(df)
    total_returns = df['returns'].sum()


    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'MS Gothic'

    # グラフの作成
    plt.figure(figsize=(6, 4), layout='constrained')
    plt.plot(df.index + 1, (df['returns_rate'] * 100), marker='.', color='b')
    start_point = int(betting_count * 0.1)
    plt.xlim([start_point, betting_count])
    plt.xticks(np.arange(start_point, betting_count + 1, step=200))
    plt.title('賭けた回数と回収率の推移', fontsize=16)
    plt.xlabel('賭けた回数', fontsize=12)
    plt.ylabel('回収率 (%)', fontsize=12)
    plt.grid(True)
    plt.show()

    # 払い戻し金額と賭けた回数を出力
    print(f"総払い戻し金額: {total_returns}円")
    print(f"賭けた回数: {betting_count}回")

    return df
  

class RFModel(PredBase):
  def __init__(
    self, 
    train_df: pd.DataFrame | None,
    returns_df: pd.DataFrame | None, 
    bet_type='umaren', 
    threshold=None, 
    random_: bool = False,
    stochastic_variation=True,
    model=None
  ):

    # 学習データと払戻データを初期化
    super().__init__(returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'rf'
    self.df = train_df
    self.random_ = random_
    self.model = model if model else self.model_train()
  

  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()
    
    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df, train=True)
    df_d = self.drop_columns(df_p)

    # データ分割
    X = df_d.drop(['target'], axis=1)
    y = df_d['target']

    # データ分割
    # 1から100の間のランダムな整数を取得
    if self.random_:
      random_value = random.randint(1, 100)
    else:
      random_value = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_value if random_value else 42)
    
    # オーバーサンプリング
    smote = SMOTE(random_state=random_value)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
      
    # ランダムフォレストモデルのトレーニング
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_smote, y_train_smote)
    if self.threshold is not None:
      y_pred_proba = model.predict_proba(X_test)[:, 1]  # 予測確率を取得
      y_pred = (y_pred_proba >= self.threshold).astype(int)
    else:
      y_pred = model.predict(X_test)

    # モデルの評価結果を出力
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    self.accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", self.accuracy)

    # ランダムフォレストの場合、特徴量の重要度を表示
    feature_importance = pd.DataFrame({
      'feature': X.columns,
      'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Feature Importance:\n", feature_importance.head(20))

    return model
  

  def predict_target(self, pred_df):
    """予測結果を生成"""
    # pred_dfをコピーして予測用に加工
    df = pred_df.copy()

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p)
    

    # 閾値の有無
    if self.threshold is not None:
      # テストデータで予測 (確率を取得)
      y_pred_proba = self.model.predict_proba(df_x)[:, 1]
      # 閾値を基にクラスを決定
      predicted_target = (y_pred_proba >= self.threshold).astype(int)

    else:
      predicted_target = self.model.predict(df_x)


    # 元の pred_df に予測結果を追加
    df_p['predicted_proba'] = self.model.predict_proba(df_x)[:, 1]
    df_p['predicted_target'] = predicted_target

    return df_p
  

# PyTorchのニューラルネットワークを使った予測クラス
class NNModel(PredBase):
  def __init__(
    self,
    train_df: pd.DataFrame | None, 
    returns_df: pd.DataFrame | None, 
    bet_type: str = 'umaren', 
    threshold: float = None,
    random_: bool = False,
    stochastic_variation: bool = True,
    embedding_dim: int = 10,
    model=None
  ):
      
    super().__init__(returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'nn'
    self.random_ = random_
    self.scaler = StandardScaler()
    if model:
      self.model = model
    else:
      self.df = train_df
      self.embedding_dim = embedding_dim
      self.model = self.model_train()


  def model_train(self):
    df = self.df.copy()
    
    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df = self.preprocess_df(df, train=True)
    df_p = self.drop_columns(df)

    X = df_p.drop(['target'], axis=1)
    y = df_p['target']

    # データ正規化 (標準化)
    self.scaler.fit(X)
    X_scaled = self.scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # データ分割 (ID列と数値列を分割する)
    if self.random_:
      random_value = random.randint(1, 100)
    else:
      random_value = 42
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_value)

    # 数値データのみにSMOTEを適用
    smote = SMOTE(random_state=random_value)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    class Net(nn.Module):
      def __init__(self, input_size):
        super(Net, self).__init__()
        # 数値データ用の全結合層 (ID列 + 数値列)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

      def forward(self, x):
        # 全結合層に通す
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

    # モデルのインスタンス作成 (入力サイズはID列と数値列の合計)
    input_size = X_train_smote.shape[1]
    model = Net(input_size=input_size)

    # トレーニング
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # PyTorchテンソルに変換
    X_train_tensor = torch.tensor(X_train_smote.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_smote.values, dtype=torch.float32)

    # 訓練ループ
    for epoch in range(100):
      model.train()
      optimizer.zero_grad()
      
      # フォワードパス
      output = model(X_train_tensor)
      
      # 損失計算
      loss = criterion(output, y_train_tensor.unsqueeze(1))
      loss.backward()
      optimizer.step()

    # 評価 (テストデータ)
    model.eval()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_pred_proba = model(X_test_tensor).detach().numpy()
    y_pred = (y_pred_proba >= (self.threshold if self.threshold is not None else 0.5)).astype(int)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    self.accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", self.accuracy)

    return model
    

  def predict_target(self, pred_df):
    """予測結果を生成"""
    df = pred_df.copy()

    # 新しいデータに同様の前処理を行う
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p)

    # データ正規化 (標準化)
    self.scaler.fit(df_x)
    df_x_scaled = self.scaler.transform(df_x)
    df_x_scaled = pd.DataFrame(df_x_scaled, columns=df_x.columns)
    
    x_tensor = torch.tensor(df_x_scaled.values, dtype=torch.float32)

    # モデルを評価モードに切り替え
    self.model.eval()

    # 予測結果を得る (確率)
    with torch.no_grad():
      y_pred_proba = self.model(x_tensor).detach().numpy()

    # 確率を0, 1に変換 (しきい値0.5を基準に)
    if self.threshold is not None:
      predicted_target = (y_pred_proba >= self.threshold).astype(int)
    else:
      predicted_target = (y_pred_proba >= 0.5).astype(int)
    
    df_p['predicted_proba'] = y_pred_proba
    df_p['predicted_target'] = predicted_target

    return df_p
  

class LGBModel(PredBase):
  def __init__(
    self, 
    train_df: pd.DataFrame | None, 
    returns_df: pd.DataFrame | None, 
    bet_type='umaren', 
    threshold=None, 
    random_: bool = False,
    stochastic_variation=True, 
    model=None
  ):

    # 学習データと払戻データを初期化
    super().__init__(returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'lgb'
    self.df = train_df
    self.random_ = random_
    self.model = model if model else self.model_train()

  
  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df = self.preprocess_df(df, train=True)
    df_p = self.drop_columns(df)

    # データ分割
    X = df_p.drop(['target'], axis=1)
    y = df_p['target']

    # データ分割
    if self.random_ :
      # 1から100の間のランダムな整数を取得
      random_value = random.randint(1, 100)
    else:
      random_value = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_value)
    
    # オーバーサンプリング
    smote = SMOTE(random_state=random_value)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
      
    # LightGBMモデルのトレーニング
    train_data = lgb.Dataset(X_train_smote, label=y_train_smote)
    params = {'objective': 'binary','metric': 'auc', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05}
    model = lgb.train(params, train_data, num_boost_round=100)
    y_pred_proba = model.predict(X_test)  # 予測確率を取得
    y_pred = (y_pred_proba >= (self.threshold if self.threshold is not None else 0.5)).astype(int)

    # モデルの評価結果を出力
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    self.accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", self.accuracy)

    # LightGBMの場合、特徴量の重要度を表示
    feature_importance = pd.DataFrame({
      'feature': X.columns,
      'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    print("Feature Importance:\n", feature_importance.head(20))

    return model
  

  def predict_target(self, pred_df):
    """予測結果を生成"""
    # pred_dfをコピーして予測用に加工
    df = pred_df.copy()

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p)
    
    y_pred_proba = self.model.predict(df_x)  # 予測確率を取得
    predicted_target = (y_pred_proba >= (self.threshold if self.threshold is not None else 0.5)).astype(int)

    # 元の pred_df に予測結果を追加
    df_p['predicted_proba'] = y_pred_proba
    df_p['predicted_target'] = predicted_target

    return df_p
		

class XGBModel(PredBase):
  def __init__(
    self, 
    train_df: pd.DataFrame | None, 
    returns_df: pd.DataFrame | None, 
    bet_type='umaren', 
    threshold=None, 
    random_: bool = False,
    stochastic_variation=True, 
    model=None
  ):

    # 学習データと払戻データを初期化
    super().__init__(returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'xgb'
    self.df = train_df
    self.random_ = random_
    self.model = model if model else self.model_train()
  

  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()
    
    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df = self.preprocess_df(df, train=True)
    df_p = self.drop_columns(df)

    # データ分割
    X = df_p.drop(['target'], axis=1)
    y = df_p['target']

    # データ分割
    if self.random_:
      random_value = random.randint(1, 100)
    else:
      random_value = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_value)
    
    # オーバーサンプリング
    smote = SMOTE(random_state=random_value)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
      
    # XGBoostモデルのトレーニング
    model = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.1, n_estimators=100, n_jobs=-1)
    model.fit(X_train_smote, y_train_smote)

    if self.threshold is not None:
      y_pred_proba = model.predict_proba(X_test)[:, 1]  # 予測確率を取得
      y_pred = (y_pred_proba >= self.threshold).astype(int)
    else:
      y_pred = model.predict(X_test)

    # モデルの評価結果を出力
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    self.accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", self.accuracy)

    # XGBoostの場合、特徴量の重要度を表示
    feature_importance = pd.DataFrame({
      'feature': X.columns,
      'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Feature Importance:\n", feature_importance.head(20))

    return model
  

  def predict_target(self, pred_df):
    """予測結果を生成"""
    # pred_dfをコピーして予測用に加工
    df = pred_df.copy()

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p)
    

    # 閾値の有無
    if self.threshold is not None:
      # テストデータで予測 (確率を取得)
      y_pred_proba = self.model.predict_proba(df_x)[:, 1]
      # 閾値を基準にクラスを決定
      predicted_target = (y_pred_proba >= self.threshold).astype(int)

    else:
      predicted_target = self.model.predict(df_x)


    # 元の pred_df に予測結果を追加
    df_p['predicted_proba'] = self.model.predict_proba(df_x)[:, 1]
    df_p['predicted_target'] = predicted_target

    return df_p
  

class EnsembleModel(PredBase):
  def __init__(
    self, 
    train_df: pd.DataFrame | None, 
    returns_df: pd.DataFrame | None, 
    bet_type, 
    threshold=None, 
    stochastic_variation=True, 
  ):
    super().__init__(returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'ensemble'
    self.df = train_df
    self.models_dict, self.models_dict_0, self.models_dict_1, self.top_2_keys, self.top_1_keys = self.model_train()

  def model_list(self, df_t, random_, model, return_model):
    if return_model == 'rf':
      rf = RFModel(
        train_df=df_t, 
        returns_df=self.returns_df,
        bet_type=self.bet_type,
        threshold=self.threshold,
        random_=random_,
        stochastic_variation=self.stochastic_variation,
        model=model if model else None
      )
      return rf
    elif return_model == 'nn':
      nn = NNModel(
        train_df=df_t, 
        returns_df=self.returns_df, 
        bet_type=self.bet_type, 
        threshold=self.threshold, 
        random_=random_,
        stochastic_variation=self.stochastic_variation,
        model=model if model else None
      )
      return nn
    elif return_model == 'lgb':
      lgb = LGBModel(
        train_df=df_t, 
        returns_df=self.returns_df, 
        bet_type=self.bet_type, 
        threshold=self.threshold, 
        random_=random_,
        stochastic_variation=self.stochastic_variation,
        model=model if model else None
      )
      return lgb
    elif return_model == 'xgb':
      xgb = XGBModel(
        train_df=df_t, 
        returns_df=self.returns_df, 
        bet_type=self.bet_type, 
        threshold=self.threshold, 
        random_=random_,
        stochastic_variation=self.stochastic_variation,
        model=model if model else None
      )
      return xgb
    

  def ensembled_model_df(
      self,
      pred_df=None, 
      models_dict=None,
      models_dict_0=None,
      models_dict_1=None, 
      top_2_keys=None, 
      top_1_keys=None, 
      output_model=False, 
      output_predicted_df=False
    ):
    
    if self.df is not None:
      df = self.df.copy()
      df_p = self.preprocess_df(df)

    if pred_df is not None:
      df = pred_df.copy()
      df_p = self.preprocess_df(df)

    model_reps = {}

    if models_dict is None:
      models_dict = {}
    else:
      models_dict = self.models_dict

    pred_base = pd.DataFrame()
    for i in range(2):
      # モデルの取得、トレーニングまたは予測
      rf = self.model_list(df, random_=True, model=(models_dict.get(f'rf_{i}')), return_model='rf')
      nn = self.model_list(df, random_=True, model=(models_dict.get(f'nn_{i}')), return_model='nn')
      lgb = self.model_list(df, random_=True, model=(models_dict.get(f'lgb_{i}')), return_model='lgb')
      xgb = self.model_list(df, random_=True, model=(models_dict.get(f'xgb_{i}')), return_model='xgb')
      
      if pred_df is None: 
        models_dict[f'rf_{i}'] = rf.model
        models_dict[f'nn_{i}'] = nn.model
        models_dict[f'lgb_{i}'] = lgb.model
        models_dict[f'xgb_{i}'] = xgb.model  

        model_reps[f'rf_{i}'] = rf.accuracy
        model_reps[f'nn_{i}'] = nn.accuracy
        model_reps[f'lgb_{i}'] = lgb.accuracy
        model_reps[f'xgb_{i}'] = xgb.accuracy

      # 予測結果の作成
      rf_pred = rf.predict_target(df)[['predicted_proba', 'predicted_target']].reset_index(drop=True)\
        .rename(columns={'predicted_proba': f'predicted_proba_rf{i}', 'predicted_target': f'predicted_target_rf{i}'})
      nn_pred = nn.predict_target(df)[['predicted_proba', 'predicted_target']].reset_index(drop=True)\
        .rename(columns={'predicted_proba': f'predicted_proba_nn{i}', 'predicted_target': f'predicted_target_nn{i}'})
      lgb_pred = lgb.predict_target(df)[['predicted_proba', 'predicted_target']].reset_index(drop=True)\
        .rename(columns={'predicted_proba': f'predicted_proba_lgb{i}', 'predicted_target': f'predicted_target_lgb{i}'})
      xgb_pred = xgb.predict_target(df)[['predicted_proba', 'predicted_target']].reset_index(drop=True)\
        .rename(columns={'predicted_proba': f'predicted_proba_xgb{i}', 'predicted_target': f'predicted_target_xgb{i}'})

      pred_base = pd.concat([pred_base, rf_pred, nn_pred, lgb_pred, xgb_pred], axis=1)
    
    df = pd.concat([df, pred_base], axis=1)

    if top_2_keys is None:
      top_2_keys = sorted(model_reps, key=model_reps.get, reverse=True)[:2]
    else:
      top_2_keys = self.top_2_keys

    # layer 0
    model_reps_0 = {}

    if models_dict_0 is None:
      models_dict_0 = {}
    else:
      models_dict_0 = self.models_dict_0

    pred_base = pd.DataFrame()
    for i, top_2_key in enumerate(top_2_keys[:2]):
      model_type = top_2_key.split('_')[0]
      # 予想結果を含めたデータ(df)で学習してモデルを取得（訓練時） or モデルの取得（予測時）
      m = self.model_list(df, random_=True, model=models_dict_0.get(top_2_key), return_model=model_type)
      if pred_df is None: 
        models_dict_0[f'{model_type}_{i}'] = m.model
        model_reps_0[f'{model_type}_{i}'] = m.accuracy
      pred = m.predict_target(df)[['predicted_proba', 'predicted_target']].reset_index(drop=True)\
        .rename(columns={'predicted_proba': f'predicted_proba_{model_type}_0{i}', 'predicted_target': f'predicted_target_{model_type}_0{i}'})
      pred_base = pd.concat([pred_base, pred], axis=1)

    df = pd.concat([df, pred_base], axis=1)

    if top_1_keys is None:
      top_1_keys = sorted(model_reps_0, key=model_reps_0.get, reverse=True)[:1]
    else:
      top_1_keys = self.top_1_keys

    # layer 1
    if models_dict_1 is None:
      models_dict_1 = {}
    else:
      models_dict_1 = self.models_dict_1

    for i, top_1_key in enumerate(top_1_keys[:1]):
      model_type = top_1_key.split('_')[0]
      m = self.model_list(df, random_=True, model=models_dict_1.get(top_1_key), return_model=model_type)
      # 予想結果を含めたデータ(df)で学習（訓練時） or モデルの取得（予測時）
      if pred_df is None: 
        models_dict_1[f'{model_type}_{i}'] = m.model
      pred = m.predict_target(df)[['predicted_proba', 'predicted_target']].reset_index(drop=True)

    df_p = pd.concat([df_p, pred], axis=1)

    if output_model and not output_predicted_df:
      return models_dict, models_dict_0, models_dict_1, top_2_keys, top_1_keys
    
    elif output_predicted_df:
      return df_p

    

  def model_train(self):
    """モデルの学習"""
    print("\n\nモデルの学習\n\n")

    return self.ensembled_model_df(
      pred_df=None, models_dict=None, models_dict_0=None, models_dict_1=None,
      top_2_keys= None, top_1_keys= None, output_model= True, output_predicted_df= False
    )

  
  def predict_target(self, pred_df):
    """予測結果を生成"""
    print("\n\nモデルの予測\n\n")
    # pred_dfをコピーして予測用に加工
    df = pred_df.copy()

    return self.ensembled_model_df(
      pred_df=df, models_dict=self.models_dict, models_dict_0=self.models_dict_0, models_dict_1=self.models_dict_1,
      top_2_keys=self.top_2_keys, top_1_keys=self.top_1_keys, output_model=False, output_predicted_df=True
    )





    
      




