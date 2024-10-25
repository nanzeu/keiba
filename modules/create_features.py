import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from modules.constants import local_paths, master

class Horse:
  def __init__(
    self,
    peds_dir=local_paths.COMPLETED_DIR,
    input_dir=local_paths.PREPROCESSED_DIR,
    output_dir=local_paths.FEATURES_DIR,
    save_filename='horse_features.csv',
    race_info_filename='race_info.csv',
    results_filename='results.csv',
    horse_results_filename='horse_results.csv',
    peds_filename=None,
  ):
    """
    results_add_infoの日付データを元に、
    それより前の日付のhorse_resultsのデータから特徴量を作成。
    """
    self.peds_df = pd.read_csv(os.path.join(peds_dir, peds_filename), index_col=0, encoding='utf8', sep='\t') if peds_filename else None

    # ファイルのパスを作成
    race_info_path = os.path.join(input_dir, race_info_filename)
    results_path = os.path.join(input_dir, results_filename)
    horse_results_path = os.path.join(input_dir, horse_results_filename)

    # データの読み込み
    race_info = pd.read_csv(race_info_path, index_col=0, encoding='utf8', sep='\t')
    results = pd.read_csv(results_path, index_col=0, encoding='utf8', sep='\t')
    horse_results = pd.read_csv(horse_results_path, index_col=0, encoding='utf8', sep='\t')

    # race_infoとresultsの結合
    self.results_add_info = results.merge(race_info, on='race_id', how='left')
    self.horse_results = horse_results

    # 過去の結果データをフィルタリングし、特徴量を作成
    self.df_f = self.filter()
    self.df_c = self.create_features()

    # 結果をCSVに保存
    self.df_c.to_csv(os.path.join(output_dir, save_filename), sep="\t")

  def filter(self):
    """馬ごとの過去の結果データをフィルタリング"""
    
    # horse_idとdateを用いてマージ
    results_with_date = self.results_add_info[['horse_id', 'date']].drop_duplicates()
    
    # self.horse_resultsと参照日でマージし、過去の結果を一度にフィルタリング
    filtered_results = pd.merge(self.horse_results, results_with_date, on='horse_id', how='left')

    # 参照日より前のデータのみ残す
    filtered_results = filtered_results[filtered_results['date_x'] < filtered_results['date_y']]

    if not filtered_results.empty:
      # 参照日を新しいカラムとして追加
      filtered_results['reference_date'] = filtered_results['date_y']
      filtered_results = filtered_results.drop(columns=['date_y'])
      filtered_results = filtered_results.rename(columns={'date_x': 'date'})
    
    return filtered_results if not filtered_results.empty else pd.DataFrame()
  

  def create_features(self):
    """過去の結果データを基に特徴量を作成"""
    past_results = self.filter()

    # 数値データに基づく特徴量を作成
    features = past_results.groupby(['horse_id', 'reference_date']).agg({
      'rank': ['mean', 'min', 'max', 'std'],     # 順位の平均、最小、最大、標準偏差
      'n_horses': ['mean'],                      # 出走頭数の平均
      'rank_diff': ['mean', 'std'],              # 着差の平均、標準偏差
      '3_furlongs': ['mean', 'std'],             # 3連輪の平均、標準偏差
      'time': ['mean'],                          # 遅延の平均
      'prize': ['mean', 'sum'],                  # 賞金の平均、合計
      'course_len': ['mean', 'min', 'max'],      # コース距離の平均、最小、最大
      'date': ['max']                            # 過去のレースの日付の最大値（最後のレースの日付）
    }).reset_index()

    # カラム名をフラット化
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.rename(columns={'horse_id_': 'horse_id', 'reference_date_': 'reference_date'})

    # カテゴリ変数は最頻値を利用して特徴量を作成
    for col in ['weather', 'race_type', 'ground_state']:
      mode_values = past_results.groupby(['horse_id', 'reference_date'])[col].agg(
        lambda x: x.mode()[0] if not x.mode().empty else None
      ).reset_index(name=f'{col}_mode')
      # 最頻値をマージ
      features = features.merge(mode_values, on=['horse_id', 'reference_date'], how='left')

    # 順位の一貫性を特徴量として追加（順位の最大値と最小値の差）
    features['consistency'] = features['rank_max'] - features['rank_min']

    features['time_per_course_len'] = features['time_mean'] / features['course_len_mean']
    features['3_furlongs_per_course_len'] = features['3_furlongs_mean'] / features['course_len_mean']

    # 最頻値のコース距離を計算
    course_len_mode = past_results.groupby(['horse_id', 'reference_date'])['course_len'].agg(
      lambda x: x.mode()[0] if not x.mode().empty else None
    ).reset_index(name='course_len_mode')

    # 同じコース距離での過去の平均順位を計算
    avg_rank_by_course_len = past_results.groupby(['horse_id', 'course_len']).agg({
      'rank': 'mean'
    }).reset_index().rename(columns={'rank': 'same_course_len_avg_rank'})

    # featuresに最頻値のコース距離での過去の平均順位をマージ
    features = features.merge(course_len_mode, on=['horse_id', 'reference_date'], how='left')
    features = features.merge(avg_rank_by_course_len, 
                              left_on=['horse_id', 'course_len_mode'], 
                              right_on=['horse_id', 'course_len'], 
                              how='left')

    # 最後のレースから次のレースまでの日数を計算
    features['last_race_date'] = pd.to_datetime(features['date_max'])  # 最後のレース日
    features['reference_date'] = pd.to_datetime(features['reference_date'])  # 現在のレース日
    features['days_since_last_race'] = (features['reference_date'] - features['last_race_date']).dt.days

    # レース間隔をカテゴリ化（7日以内, 30日以内, 90日以内）
    features['race_interval_category'] = pd.cut(features['days_since_last_race'], 
                                          bins=[0, 10, 30, 90, float('inf')], 
                                          labels=[0, 1, 2, 3])
    
    # 不要なカラムを削除
    features = features.drop(columns=['course_len', 'last_race_date', 'date_max', 'time_mean'])

    # pedsがあれば結合
    if self.peds_df is not None and not self.peds_df.empty:
      for col in self.peds_df.columns:
        if col != 'horse_id':  # horse_idはエンコードしない
          le = LabelEncoder()
          self.peds_df[col] = le.fit_transform(self.peds_df[col])
        else:
          self.peds_df[col] = self.peds_df[col]
        
      features = features.merge(self.peds_df, on=['horse_id'], how='left')

    return features

