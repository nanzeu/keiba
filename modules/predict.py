import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

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
    train_df: pd.DataFrame, 
    returns_df: pd.DataFrame, 
    bet_type: str = 'umaren', 
    threshold: float = None, 
    stochastic_variation: bool = True
  ):
    
    self.df = train_df  # 学習データの初期化
    self.returns_df = returns_df  # 払戻データの初期化
    self.bet_type = bet_type  # 賭け方の初期化
    self.threshold = threshold  # 閾値の初期化
    self.stochastic_variation = stochastic_variation  # 確率による賭け金額の調整の有無

  def process_missing_values(self, df):
    """欠損値の補完"""
    for col in df.columns:
      if df[col].isnull().sum() > 0:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
          # 数値データの平均値で補完
          df[col].fillna(df[col].mean(), inplace=True)
        else:
          # カテゴリデータは最頻値で補完
          df[col].fillna(df[col].mode()[0], inplace=True)
    return df
  

  def preprocess_df(self, df):
    # 欠損値の補完
    df = self.process_missing_values(df)

    # カウントエンコーディング
    count_enc = ce.CountEncoder(cols=['date'])
    df_ce = count_enc.fit_transform(df[['date']])

    # ラベルエンコーディング
    df_le = pd.DataFrame()
    self.encoded_features = ['date_encoded', 'parent_0', 'parent_1', 'parent_2', 'parent_3', 'parent_4', 'parent_5']
    for col in ['horse_id', 'jockey_id', 'trainer_id', 'owner_id']:
      le = LabelEncoder()
      df_le[col] = le.fit_transform(df[col])

      self.encoded_features.append(f'{col}_encoded')

    # 元のデータフレームにエンコードされた列を追加
    df = df.join(pd.concat([df_ce, df_le], axis=1), rsuffix='_encoded')

    return df
  

  def drop_columns(self, df):
    self.drop_features = ['race_id', 'rank', 'horse_id', 'jockey_id', 'trainer_id', 'owner_id', 'date', 'reference_date']

    # 予測に不要なカラムを削除
    df_p = df.drop(self.drop_features, axis=1)

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
    """回収率を計算し、グラフを生成"""
    df_add_returns = self.returns_against_pred_bet(pred_df)
    df = df_add_returns.dropna(subset=['returns']).reset_index(drop=True)

    # 賭けた回数と払い戻しの総額を計算
    betting_count = len(df)
    total_returns = df['returns'].sum()

    # 累積ベット金額と払い戻しを計算
    df['total_bet'] = df['bet_sum'].cumsum()
    df['total_returns'] = df['returns'].cumsum()
    df['returns_rate'] = df['total_returns'] / df['total_bet']

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
    train_df, 
    returns_df, 
    bet_type='umaren', 
    threshold=None, 
    stochastic_variation=True, 
    model=None
  ):

    # 学習データと払戻データを初期化
    super().__init__(train_df, returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'rf'
    self.model = model if model else self.model_train()
  

  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku', 'sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")
    
    # ラベルエンコーディングと不要なカラムの処理
    df = self.preprocess_df(df)
    df_p = self.drop_columns(df)

    # データ分割
    X = df_p.drop(['target'], axis=1)
    y = df_p['target']

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # オーバーサンプリング
    smote = SMOTE(random_state=42)
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

    # ラベルエンコーディングと不要なカラムの処理
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
    train_df: pd.DataFrame, 
    returns_df: pd.DataFrame, 
    bet_type: str = 'umaren', 
    threshold: float = None,
    stochastic_variation: bool = True,
    embedding_dim: int = 10,
    model=None
  ):
      
    super().__init__(train_df, returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'nn'
    if model:
      self.model = model
    else:
      self.scaler = StandardScaler()
      self.embedding_dim = embedding_dim
      self.model = self.model_train()


  def model_train(self):
    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku', 'sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")
    
    # ラベルエンコーディングと不要なカラムの処理
    df = self.preprocess_df(df)
    df_p = self.drop_columns(df)

    X = df_p.drop(['target'], axis=1)
    y = df_p['target']

    # データ正規化 (標準化)
    self.scaler.fit(X)
    X_scaled = self.scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # データ分割 (ID列と数値列を分割する)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 数値データのみにSMOTEを適用
    smote = SMOTE(random_state=42)
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
    train_df, 
    returns_df, 
    bet_type='umaren', 
    threshold=None, 
    stochastic_variation=True, 
    model=None
  ):

    # 学習データと払戻データを初期化
    super().__init__(train_df, returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'lgb'
    self.threshold = threshold
    self.model = model if model else self.model_train()

  
  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku','sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")
    
    # ラベルエンコーディングと不要なカラムの処理
    df = self.preprocess_df(df)
    df_p = self.drop_columns(df)

    # データ分割
    X = df_p.drop(['target'], axis=1)
    y = df_p['target']

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # オーバーサンプリング
    smote = SMOTE(random_state=42)
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

    # ラベルエンコーディングと不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p)
    
    # 閾値の有無
    y_pred_proba = self.model.predict(df_x)  # 予測確率を取得
    predicted_target = (y_pred_proba >= (self.threshold if self.threshold is not None else 0.5)).astype(int)

    # 元の pred_df に予測結果を追加
    df_p['predicted_proba'] = y_pred_proba
    df_p['predicted_target'] = predicted_target

    return df_p
		

class XGBModel(PredBase):
  def __init__(
    self, 
    train_df, 
    returns_df, 
    bet_type='umaren', 
    threshold=None, 
    stochastic_variation=True, 
    model=None
  ):

    # 学習データと払戻データを初期化
    super().__init__(train_df, returns_df, bet_type, threshold, stochastic_variation)
    self.model_type = 'xgb'
    self.model = model if model else self.model_train()
  

  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku','sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")
    
    # ラベルエンコーディングと不要なカラムの処理
    df = self.preprocess_df(df)
    df_p = self.drop_columns(df)

    # データ分割
    X = df_p.drop(['target'], axis=1)
    y = df_p['target']

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # オーバーサンプリング
    smote = SMOTE(random_state=42)
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

    # ラベルエンコーディングと不要なカラムの処理
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