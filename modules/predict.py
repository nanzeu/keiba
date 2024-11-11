import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression

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
    max_bet: int = 5000,
    pivot_horse: bool = True,
    train = True,
  ):
    
    self.returns_df = returns_df  # 払戻データの初期化
    self.bet_type = bet_type  # 賭け方の初期化
    self.threshold = threshold  # 閾値の初期化
    self.max_bet = max_bet  # 1回あたりの最大賭け金額の初期化
    self.pivot_horse = pivot_horse  # 軸馬の有無
    self.train = train  # インスタンス時の訓練の有無



  def preprocess_df(self, df, encoding=True):
    df_d = df.dropna().copy()
    
      # 対象のカラム
    if encoding:
      columns_to_encode = ['horse_id', 'jockey_id', 'trainer_id']

      # 各カラムごとにLabelEncoderを適用
      for column in columns_to_encode:
        le = LabelEncoder()  # LabelEncoderのインスタンスを作成
        df_d[column] = le.fit_transform(df_d[column])  # カラムにエンコードを適用

    return df_d
  
  

  def drop_columns(self, df):
    df_p = df.copy()

    self.drop_features = [
      'rank', 'win_odds', 'popularity', 'date', 'reference_date', 'reference_year', 'year'
    ]

    # 予測に不要なカラムを削除
    df_d = df_p.drop(self.drop_features, axis=1, errors='ignore')

    return df_d


  def process_bet_type_combinations(self, bet_type, race_group):
    """馬連、馬単、三連複、三連単の組み合わせと払戻の種類を生成"""
    if bet_type == 'umaren':
      combinations = list(itertools.combinations(race_group.index, 2))
      bet_type_return = '馬連'
    elif bet_type == 'umatan':
      combinations = list(itertools.permutations(race_group.index, 2))
      bet_type_return = '馬単'
    elif bet_type == 'sanrenpuku':
      combinations = list(itertools.combinations(race_group.index, 3))
      bet_type_return = '三連複'
    elif bet_type == 'sanrentan':
      combinations = list(itertools.permutations(race_group.index, 3))
      bet_type_return = '三連単'

    return combinations, bet_type_return
  


  def process_bet_type_combinations_with_pivot_horse(self, bet_type, race_group):
    """
    確率が0.85以上の馬を軸馬として設定し、ボックス買いの組み合わせと払戻の種類を生成します。
    """
    pivot_horse_candidates = race_group[race_group['predicted_proba'] >= 0.85].sort_values('predicted_proba', ascending=False).head(1)

    if self.pivot_horse and not pivot_horse_candidates.empty:
      pivot_horse = pivot_horse_candidates.index[0]
      non_pivot_horses = race_group.index.difference([pivot_horse])
    else:
      pivot_horse = None

    if pivot_horse is not None:
      if bet_type == 'umaren':
        combinations = [(pivot_horse, other) for other in non_pivot_horses]
        bet_type_return = '馬連'
      elif bet_type == 'umatan':
        combinations = [(pivot_horse, other) for other in non_pivot_horses]
        bet_type_return = '馬単'
      elif bet_type == 'sanrenpuku':
        combinations = [(pivot_horse, *combo) for combo in itertools.combinations(non_pivot_horses, 2)]
        bet_type_return = '三連複'
      elif bet_type == 'sanrentan':
        combinations = [(pivot_horse, first, second) for first, second in itertools.permutations(non_pivot_horses, 2)]
        bet_type_return = '三連単'
    else:
      combinations, bet_type_return = self.process_bet_type_combinations(bet_type, race_group)

    return combinations, bet_type_return, pivot_horse



  def calc_returns(self, race_group, df_add_returns, combinations, bet_type_return, pivot_horse, bet_only):
    """組み合わせ数に応じて払い戻し額と賭け金額を計算"""
    bet_amount = 100
    bet_sum = bet_amount * len(combinations)
    df_add_returns.loc[race_group.index, 'bet_sum'] = bet_sum

    if bet_type_return == '馬連' or bet_type_return == '馬単':
      rank_threshold = 2

    elif bet_type_return == '三連複' or bet_type_return == '三連単':
      rank_threshold = 3

    if not bet_only:
      # 正しい結果の場合、そうでない場合の払い戻し額を設定  
      correct = race_group[race_group['rank'] <= rank_threshold].sort_values('rank', ascending=True)

      # デフォルトで bool_set を False に設定
      bool_set = False

      # 馬単、三連単で軸馬がいる場合の払い戻し条件
      if (pivot_horse is not None) and len(correct) > 0:
        if bet_type_return == '馬単':
          # 軸馬が1着である必要がある
          bool_set = correct.index[0] == pivot_horse
        elif bet_type_return == '三連単':
          # 軸馬が1着で、かつ正しい組み合わせが存在すること
          bool_set = correct.index[0] == pivot_horse 

      if bet_type_return == '馬単' or bet_type_return == '三連単':
        if len(correct) == rank_threshold and bool_set:
          returns_value = race_group[f'{bet_type_return}_returns'].iloc[0].replace("'", '').replace("[", '').replace("]", '')
          try:
            df_add_returns.loc[race_group.index, 'returns'] = int(returns_value)
          except ValueError:
            df_add_returns.loc[race_group.index, 'returns'] = np.nan
        else:
          df_add_returns.loc[race_group.index, 'returns'] = 0

        df_add_returns = df_add_returns[df_add_returns['bet_sum'] <= self.max_bet]

      else:
        if len(correct) == rank_threshold:
          returns_value = race_group[f'{bet_type_return}_returns'].iloc[0].replace("'", '').replace("[", '').replace("]", '')
          try:
            df_add_returns.loc[race_group.index, 'returns'] = int(returns_value)
          except ValueError:
            df_add_returns.loc[race_group.index, 'returns'] = np.nan
        else:
          df_add_returns.loc[race_group.index, 'returns'] = 0

        df_add_returns = df_add_returns[df_add_returns['bet_sum'] <= self.max_bet]

    else:
      df_add_returns = df_add_returns[df_add_returns['bet_sum'] <= self.max_bet]

    return df_add_returns
  

  
  def returns_against_pred_bet(self, pred_df, bet_only):
    """予測に基づく払戻額を計算し、賭け金額を追加"""
    if not bet_only:
      df = self.predict_target(pred_df)

      df = df.loc[df['predicted_target'] == 1, ['race_id', 'win_odds', 'rank', 'predicted_target', 'predicted_proba']]

      df_add_returns = df.merge(self.returns_df, on='race_id', how='left')
      df_add_returns['returns'] = 0
      df_add_returns['bet_sum'] = 0  # 賭け金額カラムを追加

    else:
      df = pred_df.copy()
      df_add_returns = df[df['predicted_target'] == 1][['race_id', 'predicted_proba', 'predicted_target']]

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
        combinations, bet_type, pivot_horse = self.process_bet_type_combinations_with_pivot_horse(self.bet_type, race_group)
        df_add_returns = self.calc_returns(race_group, df_add_returns, combinations, bet_type, pivot_horse, bet_only)

    if not bet_only:
      return df_add_returns[['race_id', 'returns', 'bet_sum']].drop_duplicates()
    else:
      return df_add_returns[['race_id', 'bet_sum']].drop_duplicates()
    


  def returns_against_high_prob_bet(self, pred_df, bet_only):
    """各レースで確率が一定以上の馬に賭けるように払戻額を計算し、賭け金額を追加"""
    if not bet_only:
      # 予測ターゲット1のみを対象とし、確率がthreshold以上の馬にフィルタリング
      df = self.predict_target(pred_df)
      df = df.loc[:, ['race_id', 'win_odds', 'rank', 'predicted_proba', 'predicted_target']]

      # 払戻データをマージし、賭け金額の初期化
      df_add_returns = df.merge(self.returns_df, on='race_id', how='left')
      df_add_returns['returns'] = 0
      df_add_returns['bet_sum'] = 0  # 賭け金額カラムを追加

    else:
      # 予測確率とターゲット1のみにフィルタ
      df = pred_df.copy()
      df_add_returns = df[['race_id', 'predicted_proba', 'predicted_target']]
      df_add_returns['bet_sum'] = 0  # 賭け金額カラムを追加

    # 各レースで払い戻し額と賭け金額を計算
    for race_id, race_group in df_add_returns.groupby('race_id'):
      if self.bet_type == 'umaren' or self.bet_type == 'umatan':
        top_indices = race_group[race_group['predicted_target'] == 1].sort_values('predicted_proba', ascending=False).head(2).index
        # 全体を0に設定し、上位を1に設定
        race_group['predicted_target'] = 0
        race_group.loc[top_indices, 'predicted_target'] = 1
        race_group = race_group[race_group['predicted_target'] == 1]

      elif self.bet_type == 'sanrenpuku' or self.bet_type == 'sanrentan':
        top_indices = race_group[race_group['predicted_target'] == 1].sort_values('predicted_proba', ascending=False).head(3).index
        # 全体を0に設定し、上位を1に設定
        race_group['predicted_target'] = 0
        race_group.loc[top_indices, 'predicted_target'] = 1
        race_group = race_group[race_group['predicted_target'] == 1]

      # 賭け金と払い戻しの処理
      combinations, bet_type, pivot_horse = self.process_bet_type_combinations_with_pivot_horse(
        self.bet_type, race_group
      )
      
      # 支払額と収益を計算
      df_add_returns = self.calc_returns(
        race_group, df_add_returns, combinations, bet_type, pivot_horse, bet_only
      )
    
    df_add_returns.drop_duplicates(subset=['race_id'], inplace=True)
    df_add_returns = df_add_returns[df_add_returns['bet_sum'] != 0]

    # 賭け金と払戻額をまとめる
    if not bet_only:
      return df_add_returns[['race_id', 'returns', 'bet_sum']].drop_duplicates()
    else:
      return df_add_returns[['race_id', 'bet_sum']].drop_duplicates()
  


  def calc_results(self, pred_df, per_race, bet_only=False):
    if not per_race:
      df = self.returns_against_pred_bet(pred_df, bet_only=bet_only)
    else:
      df = self.returns_against_high_prob_bet(pred_df, bet_only=bet_only)

    if not bet_only:
      df = df.dropna(subset=['returns']).reset_index(drop=True)

    # 累積ベット金額と払い戻しを計算
    df['total_bet'] = df['bet_sum'].cumsum()
    if not bet_only:
      df['total_returns'] = df['returns'].cumsum()
      df['returns_rate'] = df['total_returns'] / df['total_bet']
      df['earned'] = df['total_returns'] - df['total_bet']

    return df



  def plot_returns_rate(self, pred_df, per_race=False):
    """回収率を計算し、グラフを生成"""
    df = self.calc_results(pred_df, per_race=per_race)

    # 賭けた回数と払い戻しの総額を計算
    betting_count = len(df)
    total_returns = df['returns'].sum()

    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'MS Gothic'

    # グラフの作成
    plt.figure(figsize=(6, 4), layout='constrained')
    plt.plot(df.index + 1, (df['returns_rate'] * 100), marker='.', color='b')
    # start_point = int(betting_count * 0.1)
    # plt.xlim([start_point, betting_count])
    # plt.xticks(np.arange(start_point, betting_count + 1, step=200))
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
    bet_type = 'umaren', 
    threshold = None, 
    max_bet: int = 5000,
    pivot_horse: bool = True,
    select_features: bool = True,
    select_num: int = 30,
    selected_features = None,
    train = True,
    model = None,
    save = False,
    save_name = 'rf_model'
  ):

    # 学習データと払戻データを初期化
    super().__init__(returns_df, bet_type, threshold, max_bet, pivot_horse, train)
    self.model_type = 'rf'
    self.df = train_df
    self.select_features = select_features
    self.selected_features = selected_features
    self.select_num = select_num
    self.save = save
    self.save_name = save_name

    if model or not self.train:
      self.model = model
    elif self.select_features:
      self.selected_features, self.model = self.model_train()
    else:
      self.model = self.model_train()

  


  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku','sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")  

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_d = self.drop_columns(df_p).drop(['race_id'], axis=1)

    # データ分割
    X = df_d.drop(['target'], axis=1)
    y = df_d['target']

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ランダムアンダーサンプリング
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
      
    # ランダムフォレストモデルのトレーニング
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    if self.save:
      # モデルを保存
      with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}.pickle'), "wb") as f:
        pickle.dump(model, f)

    if self.select_features:
      # 特徴量重要度を取得し、上位30個の特徴量を選択
      feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
      }).sort_values('importance', ascending=False)

      # 上位30個の特徴量を使用
      selected_features = feature_importance['feature'].head(self.select_num).tolist()
      X_train_res = X_train_res[selected_features]
      X_test = X_test[selected_features]

      # 再度モデルをトレーニング
      model.fit(X_train_res, y_train_res)

      if self.save:
        # モデルを保存
        with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}.pickle'), "wb") as f:
          pickle.dump(model, f)
        with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}_features.pickle'), "wb") as f:
          pickle.dump(selected_features, f)

    # 予測と評価
    if self.threshold is not None:
      y_pred_proba = model.predict_proba(X_test)[:, 1]
      y_pred = (y_pred_proba >= self.threshold).astype(int)
    else:
      y_pred = model.predict(X_test)

    # 特徴量重要度を取得し、上位30個の特徴量を選択
    feature_importance = pd.DataFrame({
      'feature': X_train_res.columns,
      'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    self.accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", self.accuracy)

    # 特徴量の重要度も表示
    print("Selected Feature Importance:\n", feature_importance.head(self.select_num))

    self.model = model
   
    if self.select_features:
      self.selected_features = selected_features
      return selected_features, model

    return model
  


  def predict_target(self, pred_df):
    """予測結果を生成"""
    # pred_dfをコピーして予測用に加工
    df = pred_df.copy()

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p)

    if self.select_features:
      # 特徴量重要度を取得し、上位30個の特徴量を選択
      df_x = df_x[self.selected_features]

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


# PyTorchのニューラルネットワークを使った予測クラス
class NNModel(PredBase):
  def __init__(
    self,
    train_df: pd.DataFrame | None, 
    returns_df: pd.DataFrame | None, 
    bet_type: str = 'umaren', 
    threshold: float = None,
    max_bet: int = 5000,
    pivot_horse: bool = False,
    train = True,
    select_features = True,
    selected_features = None,
    select_num: int = 30,
    model=None,
    save = False,
    save_name = 'nn_model'
  ):
      
    super().__init__(returns_df, bet_type, threshold, max_bet, pivot_horse, train)
    self.model_type = 'nn'
    self.scaler = StandardScaler()
    self.select_features = select_features
    self.selected_features = selected_features
    self.select_num = select_num
    self.save = save
    self.save_name = save_name

    if model or not self.train:
      self.model = model
    elif self.select_features:
      self.df = train_df
      self.selected_features, self.model = self.model_train()
    else:
      self.df = train_df
      self.model = self.model_train()




  def model_train(self):
    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku','sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")
    
    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_d = self.drop_columns(df_p).drop(['race_id'], axis=1)

    print('NN_model_train:', df_d)

    X = df_d.drop(['target'], axis=1)
    y = df_d['target']

    # データ正規化 (標準化)
    self.scaler.fit(X)
    X_scaled = self.scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # PCAの適用 (30次元に圧縮)
    if self.select_features:
      pca = PCA(n_components=self.select_num)
      X_pca = pca.fit_transform(X_scaled.values)  # 数値部分にPCAを適用
      X_scaled = pd.DataFrame(X_pca, columns=[f'pca_{i+1}' for i in range(self.select_num)])  # 列名を設定

    # データ分割 
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ランダムアンダーサンプリング
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    # モデルのインスタンス作成 (入力サイズはID列と数値列の合計)
    input_size = X_train_res.shape[1]

    model = Net(input_size=input_size)

    # トレーニング
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # PyTorchテンソルに変換
    X_train_tensor = torch.tensor(X_train_res.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_res.values, dtype=torch.float32)

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

    self.model = model

    if self.save:
      torch.save(model.state_dict(), os.path.join(local_paths.MODELS_DIR, f'{self.save_name}.pth'))

    if self.select_features:
      if self.save:
        # PCA モデルを保存
        dump(pca, os.path.join(local_paths.MODELS_DIR, f'{self.save_name}_features.joblib'))
        self.selected_features = pca
      return pca, model

    return model
    


  def predict_target(self, pred_df):
    """予測結果を生成"""
    df = pred_df.copy()

    # 新しいデータに同様の前処理を行う
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p).drop(['race_id'], axis=1)

    # データ正規化 (標準化)
    self.scaler.fit(df_x)
    df_x_scaled = self.scaler.transform(df_x)
    df_x_scaled = pd.DataFrame(df_x_scaled, columns=df_x.columns)

    # PCA変換
    if self.select_features:
      df_x_scaled = self.selected_features.transform(df_x_scaled.values)
      df_x_scaled = pd.DataFrame(df_x_scaled, columns=[f'pca_{i+1}' for i in range(self.selected_features.n_components_)])
    
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
    max_bet: int = 5000,
    pivot_horse: bool = True,
    select_features: bool = True,
    select_num: int = 30,
    selected_features=None,
    train = True,
    model=None,
    save = False,
    save_name = 'lgb_model'
  ):

    # 学習データと払戻データを初期化
    super().__init__(returns_df, bet_type, threshold, max_bet, pivot_horse, train)
    self.model_type = 'lgb'
    self.df = train_df
    self.select_features = select_features
    self.select_num = select_num
    self.selected_features = selected_features
    self.save = save
    self.save_name = save_name

    if model or not self.train:
      self.model = model
    elif self.select_features:
      self.selected_features, self.model = self.model_train()
    else:
      self.model = self.model_train()

  
  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku','sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")  


    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_d = self.drop_columns(df_p).drop(['race_id'], axis=1)

    # データ分割
    X = df_d.drop(['target'], axis=1)
    y = df_d['target']

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ランダムアンダーサンプリング
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
      
    # LightGBMモデルのトレーニング
    train_data = lgb.Dataset(X_train_res, label=y_train_res)
    params = {'objective': 'binary','metric': 'auc', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05}
    model = lgb.train(params, train_data, num_boost_round=100)

    if self.save:
      # モデルを保存
      with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}.pickle'), "wb") as f:
        pickle.dump(model, f)

    if self.select_features:
      # 特徴量重要度を取得し、上位30個の特徴量を選択
      feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
      }).sort_values('importance', ascending=False)

      # 上位30個の特徴量を使用
      selected_features = feature_importance['feature'].head(self.select_num).tolist()
      X_train_res = X_train_res[selected_features]
      X_test = X_test[selected_features]

      # 再度モデルをトレーニング
      train_data_selected = lgb.Dataset(X_train_res, label=y_train_res)
      model = lgb.train(params, train_data_selected, num_boost_round=100)

      if self.save:
        # モデルを保存
        with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}.pickle'), "wb") as f:
          pickle.dump(model, f)
        with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}_features.pickle'), "wb") as f:
          pickle.dump(selected_features, f)
      
    y_pred_proba = model.predict(X_test)  # 予測確率を取得
    y_pred = (y_pred_proba >= (self.threshold if self.threshold is not None else 0.5)).astype(int)

    # モデルの評価結果を出力
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    self.accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", self.accuracy)

    # 特徴量重要度を取得し、上位30個の特徴量を選択
    feature_importance = pd.DataFrame({
      'feature': X_train_res.columns,
      'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    # LightGBMの場合、特徴量の重要度を表示
    print("Selected Feature Importance:\n", feature_importance.head(self.select_num))

    self.model = model

    if self.select_features:
      self.selected_features = selected_features
      return selected_features, model

    return model
  
  

  def predict_target(self, pred_df):
    """予測結果を生成"""
    # pred_dfをコピーして予測用に加工
    df = pred_df.copy()

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p).drop(['race_id'], axis=1)

    if self.select_features:
      # 特徴量重要度を取得し、上位30個の特徴量を選択
      df_x = df_x[self.selected_features]
    
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
    max_bet: int = 5000,
    pivot_horse: bool = True,
    select_features: bool = True,
    select_num: int = 30,
    selected_features = None,
    train = True,
    model=None,
    save = False,
    save_name = 'xgb_model'
  ):

    # 学習データと払戻データを初期化
    super().__init__(returns_df, bet_type, threshold, max_bet, pivot_horse, train)
    self.model_type = 'xgb'
    self.df = train_df
    self.select_features = select_features
    self.select_num = select_num
    self.selected_features = selected_features
    self.save = save
    self.save_name = save_name

    if model or not self.train:
      self.model = model
    elif self.select_features:
      self.selected_features, self.model = self.model_train()
    else:
      self.model = self.model_train()
  

  def model_train(self):
    """訓練データを使ってモデルをトレーニング"""
    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku','sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")
    
    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_d = self.drop_columns(df_p).drop(['race_id'], axis=1)

    # データ分割
    X = df_d.drop(['target'], axis=1)
    y = df_d['target']

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ランダムアンダーサンプリング
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
      
    # XGBoostモデルのトレーニング
    model = xgb.XGBClassifier(objective='binary:logistic', max_depth=6, learning_rate=0.1, n_estimators=100, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    if self.save:
      # モデルを保存
      with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}.pickle'), "wb") as f:
        pickle.dump(model, f)

    if self.select_features:
      # 特徴量重要度を取得し、上位30個の特徴量を選択
      feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
      }).sort_values('importance', ascending=False)

      # 上位30個の特徴量を使用
      selected_features = feature_importance['feature'].head(self.select_num).tolist()
      X_train_res = X_train_res[selected_features]
      X_test = X_test[selected_features]

      # 再度モデルをトレーニング
      model.fit(X_train_res, y_train_res)

      if self.save:
        # モデルを保存
        with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}.pickle'), "wb") as f:
          pickle.dump(model, f)
        with open(os.path.join(local_paths.MODELS_DIR, f'{self.save_name}_features.pickle'), "wb") as f:
          pickle.dump(selected_features, f)

    # 予測と評価
    if self.threshold is not None:
      y_pred_proba = model.predict_proba(X_test)[:, 1]
      y_pred = (y_pred_proba >= self.threshold).astype(int)
    else:
      y_pred = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    self.accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", self.accuracy)

    # 特徴量重要度を取得し、上位30個の特徴量を選択
    feature_importance = pd.DataFrame({
      'feature': X_train_res.columns,
      'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # 特徴量の重要度も表示
    print("Selected Feature Importance:\n", feature_importance.head(self.select_num))

    self.model = model

    if self.select_features:
      self.selected_features = selected_features
      return selected_features, model

    return model
  

  def predict_target(self, pred_df):
    """予測結果を生成"""
    # pred_dfをコピーして予測用に加工
    df = pred_df.copy()

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df)
    df_x = self.drop_columns(df_p).drop(['race_id'], axis=1)

    if self.select_features:
      # 特徴量重要度を取得し、上位30個の特徴量を選択
      df_x = df_x[self.selected_features]

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
    max_bet: int = 5000,
    pivot_horse: bool = True,
    select_features: bool = True,
    select_num: int = 30,
    final_model='lgb',
    base_models=None,
    meta_models=None,
    base_models_features=None,
    meta_models_features=None,
    cs: bool = False,
    save: bool = False
  ):
    super().__init__(returns_df, bet_type, threshold, max_bet, pivot_horse)
    self.model_type = 'ensemble'
    self.df = train_df
    self.select_features = select_features
    self.select_num = select_num
    self.final_model = final_model
    self.cs = cs
    self.save = save

    if base_models and meta_models:
      self.base_models = base_models
      self.meta_models = meta_models
      self.base_models_features = base_models_features
      self.meta_models_features = meta_models_features
    else:
      self.base_models, self.meta_models, self.base_models_features, self.meta_models_features = self.model_train()



  def models_instance(self, df, model_type, selected_features=None, model=None, save_name=None):
    """モデルのインスタンスを作成"""
    if model_type == 'rf':
      model = RFModel(
        train_df=df, returns_df=self.returns_df, bet_type=self.bet_type, 
        threshold=self.threshold, max_bet=self.max_bet, pivot_horse=self.pivot_horse, select_features=self.select_features, 
        select_num=self.select_num, selected_features=selected_features, train=True, model=model, 
        save=self.save, save_name=save_name
      )
    elif model_type == 'nn':
      model = NNModel(
        train_df=df, returns_df=self.returns_df, bet_type=self.bet_type, 
        threshold=self.threshold, max_bet=self.max_bet, pivot_horse=self.pivot_horse, select_features=self.select_features,
        select_num=self.select_num, selected_features=selected_features, train=True, model=model, 
        save=self.save, save_name=save_name
      )
    elif model_type == 'lgb':
      model = LGBModel(
        train_df=df, returns_df=self.returns_df, bet_type=self.bet_type, 
        threshold=self.threshold, max_bet=self.max_bet, pivot_horse=self.pivot_horse, select_features=self.select_features, 
        select_num=self.select_num, selected_features=selected_features, train=True, model=model, 
        save=self.save, save_name=save_name
      )
    elif model_type == 'xgb':
      model = XGBModel(
        train_df=df, returns_df=self.returns_df, bet_type=self.bet_type, 
        threshold=self.threshold, max_bet=self.max_bet, pivot_horse=self.pivot_horse, select_features=self.select_features, 
        select_num=self.select_num, selected_features=selected_features, train=True, model=model, 
        save=self.save, save_name=save_name
      )

    return model



  def model_train(self):
    """モデルの学習"""
    print("\n training... \n\n")

    df = self.df.copy()

    if self.bet_type in ['umaren', 'umatan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 2 else 0)  # 馬連、馬単
    elif self.bet_type in ['sanrenpuku','sanrentan']:
      df['target'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)  # 三連複、三連単
    else:
      raise RuntimeError(f"{self.bet_type} is not supported.")  
    
    # ラベルエンコーディング、欠損値の処理
    df_p = self.preprocess_df(df)

    # データ分割
    X = df_p.drop(['target'], axis=1)
    y = df_p['target']

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_models_features = {}
    base_models = {}
    meta_models_features = {}
    meta_models = {}

    # 学習
    X_train = X_train.reset_index(drop=True)
    meta_train = X_train.copy()

    # k-fold
    kf = KFold(n_splits=5)
    for train_idx, val_idx in kf.split(X_train):
      X_train_kf, X_val_kf = X_train.iloc[train_idx], X_train.iloc[val_idx]

      for model_type in ['rf', 'nn', 'lgb', 'xgb']:
        if not self.cs:
          save_name = f'en_{model_type}_basemodel'
        else:
          save_name = f'en_{model_type}_basemodel_cs'
        model = self.models_instance(X_train_kf, model_type, model=None, save_name=save_name)
        val_pred = model.predict_target(X_val_kf)
        if self.select_features:
          base_models_features[model_type] = model.selected_features
        base_models[model_type] = model.model

        meta_train.loc[val_idx, f'predicted_proba_{model_type}'] = val_pred['predicted_proba']
        meta_train.loc[val_idx, f'predicted_target_{model_type}'] = val_pred['predicted_target']

    if not self.cs:
      save_name = f'en_{self.final_model}_metamodel'
    else:
      save_name = f'en_{self.final_model}_metamodel_cs'

    meta_model = self.models_instance(meta_train, self.final_model, model=None, save_name=save_name)
    if self.select_features:
      meta_models_features[self.final_model] = meta_model.selected_features
    meta_models[self.final_model] = meta_model.model

    # テストデータで予測
    X_test = X_test.reset_index(drop=True)
    meta_test = X_test.copy()

    for key, base_model in base_models.items():
      meta_model = self.models_instance(
        X_test, key, selected_features=base_models_features[key] if self.select_features else None, model=base_model
      )
      test_pred = meta_model.predict_target(X_test)
      meta_test[f'predicted_proba_{key}'] = test_pred['predicted_proba']
      meta_test[f'predicted_target_{key}'] = test_pred['predicted_target']

    meta_model = self.models_instance(
      meta_test, self.final_model, selected_features=meta_models_features[self.final_model] if self.select_features else None, model=meta_models[self.final_model]
    )
    pred = meta_model.predict_target(meta_test)['predicted_target']


    # 評価指標を計算
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    # 結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # 追加で混同行列も表示
    conf_matrix = confusion_matrix(y_test, pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    return base_models, meta_models, base_models_features, meta_models_features
        


  def predict_target(self, pred_df):
    """
    予測対象データに対する予測結果を返す関数
    """
    df = pred_df.copy()

    # targetの設定とラベルエンコーディング、不要なカラムの処理
    df_p = self.preprocess_df(df, encoding=False)
    df_x = self.drop_columns(df_p)

    print('predict_dataframe', df_x)

    meta_data = df_x.copy()

    # ベースモデルの予測
    for key, base_model in self.base_models.items():
      model = self.models_instance(
        df_x, key, selected_features=self.base_models_features[key] if self.select_features else None, model=base_model
      )
      test_pred = model.predict_target(df_x)
      meta_data[f'predicted_proba_{key}'] = test_pred['predicted_proba']
      meta_data[f'predicted_target_{key}'] = test_pred['predicted_target']

    print(meta_data)
  
    # メタモデルの予測
    meta_model = self.models_instance(
      meta_data, self.final_model, selected_features=\
        self.meta_models_features[self.final_model] if self.select_features else None, model=self.meta_models[self.final_model])
    meta_pred = meta_model.predict_target(meta_data)[['predicted_proba', 'predicted_target']]
    
    df_p = pd.concat([df_p, meta_pred], axis=1)

    return df_p

    
      




