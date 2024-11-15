import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from modules.constants import local_paths

class Horse:
  def __init__(
    self,
    input_dir=local_paths.PREPROCESSED_DIR,
    output_dir=local_paths.FEATURES_DIR,
    candidates_dir=local_paths.CANDIDATES_DIR,
    save_filename='horse_jockey_features.csv',
    race_info_filename='race_info.csv',
    results_filename='results.csv',
    horse_results_filename='horse_results.csv',
    peds_filename=None,
    cs: bool = False,
    new = False
  ):
    """
    results_add_infoの日付データを元に、
    それより前の日付のhorse_resultsのデータから特徴量を作成。
    """

    if cs:
      input_dir = local_paths.PREPROCESSED_CS_DIR
      candidates_dir = local_paths.CANDIDATES_CS_DIR
      output_dir = local_paths.FEATURES_CS_DIR

    if cs and new:
      candidates_dir = local_paths.CANDIDATES_CS_DIR
      output_dir = local_paths.CANDIDATES_CS_DIR
      
  
    # ファイルのパスを作成
    race_info_path = os.path.join(input_dir, race_info_filename)
    results_path = os.path.join(input_dir, results_filename)
    horse_results_path = os.path.join(input_dir, horse_results_filename)
    peds_path = os.path.join(input_dir, peds_filename)

    if new:
      race_info_path = os.path.join(candidates_dir, race_info_filename)
      results_path = os.path.join(candidates_dir, results_filename)
      horse_results_path = os.path.join(candidates_dir, horse_results_filename)
      peds_path = os.path.join(candidates_dir, peds_filename)

    # データの読み込み
    race_info = pd.read_csv(race_info_path, index_col=0, encoding='utf8', sep='\t')
    results = pd.read_csv(results_path, index_col=0, encoding='utf8', sep='\t', dtype={'jockey_id': str, 'trainer_id': str, 'owner_id': str})

    self.horse_results = pd.read_csv(horse_results_path, index_col=0, encoding='utf8', sep='\t')
    self.peds_df = pd.read_csv(peds_path, index_col=0, encoding='utf8', sep='\t') if peds_filename else None
    

    # race_infoとresultsの結合
    self.results_add_info = results.merge(race_info, on='race_id', how='left')

    # 過去の結果データをフィルタリングし、特徴量を作成
    self.df_f = self.filter()
    self.df_c = self.create_features()

    # 結果をCSVに保存
    self.df_c.to_csv(os.path.join(output_dir, save_filename), sep="\t")



  def filter(self):
    """馬と騎手ごとの過去の結果データをフィルタリング"""
    
    # horse_idとdateを用いてresults_add_infoとマージ
    results_with_date = self.results_add_info[['horse_id', 'date']].drop_duplicates()
    filtered_results = pd.merge(self.horse_results, results_with_date, on='horse_id', how='left')
    
    # 参照日より前のデータのみを残す
    filtered_results = filtered_results[filtered_results['date_x'] < filtered_results['date_y']]
    if not filtered_results.empty:
      filtered_results['reference_date'] = filtered_results['date_y']
      filtered_results = filtered_results.drop(columns=['date_y'])
      filtered_results = filtered_results.rename(columns={'date_x': 'date'})
    
    return filtered_results if not filtered_results.empty else pd.DataFrame()
  
  

  def create_features(self):
    """過去の結果データを基に特徴量を作成"""
    past_horse_results = self.filter()

    past_horse_results['date'] = pd.to_datetime(past_horse_results['date'])
    past_horse_results['reference_date'] = pd.to_datetime(past_horse_results['reference_date'])

    # past_horse_results['month'] = past_horse_results['date'].dt.month

    # # 季節を分類する関数を定義
    # def get_season(month):
    #     if month in [3, 4, 5]:
    #         return 'spring'
    #     elif month in [6, 7, 8]:
    #         return 'summer'
    #     elif month in [9, 10, 11]:
    #         return 'autumn'
    #     else:
    #         return 'winter'

    # # 'month' 列に基づいて 'season' 列を追加
    # past_horse_results['season'] = past_horse_results['month'].apply(get_season)

    # 直近2, 5回分のデータの追加
    past_2_results = past_horse_results.sort_values('date')\
      .groupby(['horse_id', 'reference_date']).apply(lambda x: x.tail(2)).reset_index(drop=True)
    past_5_results = past_horse_results.sort_values('date')\
      .groupby(['horse_id', 'reference_date']).apply(lambda x: x.tail(5)).reset_index(drop=True)

    
    # 数値データに基づく特徴量を作成
    aggregation_config = {
        'rank': ['mean', 'min', 'max'],
        'n_horses': 'mean',
        'rank_diff': 'mean',
        '3_furlongs': ['mean', 'median'],
        'time': ['mean', 'median'],
        'prize': ['mean', 'sum'],
        'course_len': ['mean', 'median', 'min', 'max'],
        'date': 'max'
    }
    
    features = past_horse_results.groupby(['horse_id', 'reference_date']).agg(aggregation_config).reset_index()
    features_past_2 = past_2_results.groupby(['horse_id', 'reference_date']).agg(aggregation_config).reset_index()
    features_past_5 = past_5_results.groupby(['horse_id', 'reference_date']).agg(aggregation_config).reset_index()

     # カラム名をフラット化
    def flatten_columns(df, suffix):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
        df.rename(columns={'horse_id_': 'horse_id', 'reference_date_': 'reference_date'}, inplace=True)
        return df


    # # シーズンごとの勝率、連対率、複勝率を計算
    # def calculate_rates(df):
    #   total_races = len(df)
    #   if total_races == 0:
    #       return pd.Series([0, 0, 0], index=['win_rate', 'place_rate', 'show_rate'])
    #   win_rate = (df['rank'] == 1).sum() / total_races
    #   place_rate = (df['rank'] <= 2).sum() / total_races
    #   show_rate = (df['rank'] <= 3).sum() / total_races
    #   return pd.Series([win_rate, place_rate, show_rate], index=['win_rate', 'place_rate', 'show_rate'])

    # # 各シーズンごとに勝率、連対率、複勝率を計算し、特徴量として追加
    # seasonal_rates = past_horse_results.groupby(['horse_id', 'reference_date', 'season']).apply(calculate_rates)

    # # unstack操作
    # seasonal_rates = seasonal_rates.unstack().reset_index()

    # # カラムをフラット化
    # seasonal_rates = flatten_columns(seasonal_rates, '')
    features = flatten_columns(features, '')

    # # 特徴量を統合
    # features = features.merge(seasonal_rates, on=['horse_id', 'reference_date'], how='left')
   
    features_past_2 = flatten_columns(features_past_2, '_past_2')
    features_past_5 = flatten_columns(features_past_5, '_past_5')

    # 2回分と5回分の特徴量を結合
    features = features.merge(features_past_2, on=['horse_id', 'reference_date'], how='left', suffixes=('', '_past_2'))
    features = features.merge(features_past_5, on=['horse_id', 'reference_date'], how='left', suffixes=('', '_past_5'))

    # カテゴリ変数の最頻値を特徴量として追加
    def get_mode(series):
        return series.mode()[0] if not series.mode().empty else None

    for col in ['weather', 'race_type', 'ground_state', 'race_class']:
        mode_col = past_horse_results.groupby(['horse_id', 'reference_date'])[col].apply(get_mode).reset_index(name=f'{col}_mode')
        past_5_mode_col = past_5_results.groupby(['horse_id', 'reference_date'])[col].apply(get_mode).reset_index(name=f'{col}_mode_past_5')
        past_2_mode_col = past_2_results.groupby(['horse_id', 'reference_date'])[col].apply(get_mode).reset_index(name=f'{col}_mode_past_2')
        features = features.merge(mode_col, on=['horse_id', 'reference_date'], how='left')
        features = features.merge(past_5_mode_col, on=['horse_id', 'reference_date'], how='left')
        features = features.merge(past_2_mode_col, on=['horse_id', 'reference_date'], how='left')
    
    # 一貫性の指標を追加
    features['consistency'] = features['rank_max'] - features['rank_min']
    features['3f_norm_by_mean'] = features['3_furlongs_mean'] / (600 / features['course_len_mean'])
    features['3f_norm_by_median'] = features['3_furlongs_median'] / (600 / features['course_len_median'])
    features['time_norm_by_mean'] = features['time_mean'] / features['course_len_mean']
    features['time_norm_by_median'] = features['time_median'] / features['course_len_median']

    # 直近2回分と5回分の一貫性の指標を追加
    features['consistency_past_2'] = features['rank_max_past_2'] - features['rank_min_past_2']
    features['3f_norm_by_mean_past_2'] = features['3_furlongs_mean_past_2'] / (600 / features['course_len_mean_past_2'])
    features['3f_norm_by_median_past_2'] = features['3_furlongs_median_past_2'] / (600 / features['course_len_median_past_2'])
    features['time_norm_by_mean_past_2'] = features['time_mean_past_2'] / features['course_len_mean_past_2']
    features['time_norm_by_median_past_2'] = features['time_median_past_2'] / features['course_len_median_past_2']

    features['consistency_past_5'] = features['rank_max_past_5'] - features['rank_min_past_5']
    features['3f_norm_by_mean_past_5'] = features['3_furlongs_mean_past_5'] / (600 / features['course_len_mean_past_5'])
    features['3f_norm_by_median_past_5'] = features['3_furlongs_median_past_5'] / (600 / features['course_len_median_past_5'])
    features['time_norm_by_mean_past_5'] = features['time_mean_past_5'] / features['course_len_mean_past_5']
    features['time_norm_by_median_past_5'] = features['time_median_past_5'] / features['course_len_median_past_5']


    # コース距離の最頻値と同距離での馬の過去の成績
    course_len_mode = past_horse_results.groupby(['horse_id', 'reference_date'])['course_len'].apply(get_mode).reset_index(name='course_len_mode')
    avg_mode_course_len = past_horse_results.groupby(['horse_id', 'course_len']).agg({
        'rank': ['mean', 'min', 'max'],
        'rank_diff': ['mean'],
        '3_furlongs': ['mean'],
        'time': 'mean',
        'prize': ['mean', 'sum']
    }).reset_index()
    
    avg_mode_course_len.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in avg_mode_course_len.columns]
    avg_mode_course_len.columns = [f'{col}_in_mode_course_len' for col in avg_mode_course_len.columns]
    avg_mode_course_len.rename(columns={'horse_id__in_mode_course_len': 'horse_id', 'course_len__in_mode_course_len': 'course_len'}, inplace=True)

    features = features.merge(course_len_mode, on=['horse_id', 'reference_date'], how='left')
    features = features.merge(avg_mode_course_len, left_on=['horse_id', 'course_len_mode'], right_on=['horse_id', 'course_len'], how='left')

    # コース距離の最頻値と同距離での馬の過去の成績（直近2回分と5回分）
    course_len_mode_past_2 = past_2_results.groupby(['horse_id', 'reference_date'])['course_len'].apply(get_mode).reset_index(name='course_len_mode_past_2')
    avg_mode_course_len_past_2 = past_2_results.groupby(['horse_id', 'course_len']).agg({
        'rank': ['mean', 'min', 'max'],
        'rank_diff': ['mean'],
        '3_furlongs': ['mean'],
        'time': 'mean',
        'prize': ['mean', 'sum']
    }).reset_index()

    course_len_mode_past_5 = past_5_results.groupby(['horse_id', 'reference_date'])['course_len'].apply(get_mode).reset_index(name='course_len_mode_past_5')
    avg_mode_course_len_past_5 = past_5_results.groupby(['horse_id', 'course_len']).agg({
        'rank': ['mean', 'min', 'max'],
        'rank_diff': ['mean'],
        '3_furlongs': ['mean'],
        'time': 'mean',
        'prize': ['mean', 'sum']
    }).reset_index()


    # カラム名の調整
    avg_mode_course_len_past_2.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in avg_mode_course_len_past_2.columns]
    avg_mode_course_len_past_2.columns = [f'{col}_in_mode_course_len_past_2' for col in avg_mode_course_len_past_2.columns]
    avg_mode_course_len_past_5.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in avg_mode_course_len_past_5.columns]
    avg_mode_course_len_past_5.columns = [f'{col}_in_mode_course_len_past_5' for col in avg_mode_course_len_past_5.columns]

    avg_mode_course_len_past_2.rename(columns={'horse_id__in_mode_course_len_past_2': 'horse_id', 'course_len__in_mode_course_len_past_2': 'course_len'}, inplace=True)
    avg_mode_course_len_past_5.rename(columns={'horse_id__in_mode_course_len_past_5': 'horse_id', 'course_len__in_mode_course_len_past_5': 'course_len'}, inplace=True)

    features = features.merge(course_len_mode_past_2, on=['horse_id', 'reference_date'], how='left')
    features = features.merge(avg_mode_course_len_past_2, left_on=['horse_id', 'course_len_mode_past_2'], right_on=['horse_id', 'course_len'], how='left')
    features = features.merge(course_len_mode_past_5, on=['horse_id', 'reference_date'], how='left')
    features = features.merge(avg_mode_course_len_past_5, left_on=['horse_id', 'course_len_mode_past_5'], right_on=['horse_id', 'course_len'], how='left')

    # 日数の特徴量と不要なカラムの削除は元のコードと同じ
    features['last_race_date'] = pd.to_datetime(features['date_max'])
    features['reference_date'] = pd.to_datetime(features['reference_date'])
    features['days_since_last_race'] = (features['reference_date'] - features['last_race_date']).dt.days
    features['race_interval_category'] = pd.cut(features['days_since_last_race'], bins=[0, 2, 30, 90, float('inf')], labels=[0, 1, 2, 3])
    
    
    features.drop(columns=[
      'course_len', 'course_len_x', 'course_len_y', 'last_race_date', 'date_max',
      'time_mean', 'date_max_past_5', 'time_mean_past_5', 'date_max_past_2', 'time_mean_past_2'
    ], inplace=True, errors='ignore')

    # 季節のキーワードを含むカラムを選択
    season_columns = features.columns[features.columns.str.contains('spring|summer|autumn|winter')]

    # 選択したカラムに対して fillna(0) を適用
    features[season_columns] = features[season_columns].fillna(0)

    # peds_dfがある場合の処理
    if self.peds_df is not None and not self.peds_df.empty:
      for col in self.peds_df.columns:
        if col != 'horse_id':  # horse_idはエンコードしない
          self.peds_df[col] = LabelEncoder().fit_transform(self.peds_df[col])
      features = features.merge(self.peds_df, on='horse_id', how='left')
    
    return features




class Jockey:
  def __init__(
    self,
    input_dir=local_paths.PREPROCESSED_DIR,
    output_dir=local_paths.FEATURES_DIR,
    candidates_dir=local_paths.CANDIDATES_DIR,
    save_filename='jockey_features.csv',
    race_info_filename='race_info.csv',
    results_filename='results.csv',
    jockeys_filename='jockeys.csv',
    cs: bool = False,
    new = False
  ):
    """
    results_add_infoの日付データを元に、
    それより前の日付のhorse_resultsのデータから特徴量を作成。
    """
    if cs:
      input_dir = local_paths.PREPROCESSED_CS_DIR
      candidates_dir = local_paths.CANDIDATES_CS_DIR
      output_dir = local_paths.FEATURES_CS_DIR

    if cs and new:
      candidates_dir = local_paths.CANDIDATES_CS_DIR
      output_dir = local_paths.CANDIDATES_CS_DIR
  

    # ファイルのパスを作成
    race_info_path = os.path.join(input_dir, race_info_filename)
    results_path = os.path.join(input_dir, results_filename)
    jockeys_path = os.path.join(input_dir, jockeys_filename)

    if new:
      race_info_path = os.path.join(candidates_dir, race_info_filename)
      results_path = os.path.join(candidates_dir, results_filename)
      jockeys_path = os.path.join(candidates_dir, jockeys_filename)


    # データの読み込み
    race_info = pd.read_csv(race_info_path, index_col=0, encoding='utf8', sep='\t')
    results = pd.read_csv(results_path, index_col=0, encoding='utf8', sep='\t', 
                          dtype={'jockey_id': str, 'trainer_id': str, 'owner_id': str})
    self.jockeys = pd.read_csv(jockeys_path, index_col=0, encoding='utf8', sep='\t', dtype={'jockey_id': str})

    # race_infoとresultsの結合
    self.results_add_info = results.merge(race_info, on='race_id', how='left')

    # 過去の結果データをフィルタリングし、特徴量を作成
    self.df_f = self.filter()
    self.df_c = self.create_features()

    # 結果をCSVに保存
    self.df_c.to_csv(os.path.join(output_dir, save_filename), sep="\t")



  def filter(self):
    """馬と騎手ごとの過去の結果データをフィルタリング"""
    
    # jockey_idとdateを用いてresults_add_infoとマージ
    jockeys_with_date = self.results_add_info[['jockey_id', 'date']].drop_duplicates()
    jockeys_with_date['date'] = pd.to_datetime(jockeys_with_date['date'], errors='coerce')
    jockeys_with_date['year'] = jockeys_with_date['date'].dt.year  # dateから年を抽出
    jockeys_with_date.drop(columns=['date'], inplace=True)
    jockeys_with_date = jockeys_with_date.drop_duplicates()

    filtered_jockeys = pd.merge(self.jockeys, jockeys_with_date, on='jockey_id', how='left')

    # 参照年より前のデータのみを残す
    filtered_jockeys = filtered_jockeys[filtered_jockeys['annual'] < filtered_jockeys['year']]
    if not filtered_jockeys.empty:
      filtered_jockeys['reference_year'] = filtered_jockeys['year']
      filtered_jockeys = filtered_jockeys.drop(columns=['year'])
    
    return filtered_jockeys.drop_duplicates(subset=['annual', 'jockey_id', 'reference_year']) if not filtered_jockeys.empty else pd.DataFrame()
  
  

  def create_features(self):
    """過去の結果データを基に特徴量を作成"""
    past_jockeys = self.filter()

     # ジョッキーのデータに基づく特徴量を作成
    aggregation_config = {
        'jockey_rank': ['mean','min','max'],
        'jockey_n_top_1': ['mean','sum'],
        'jockey_n_top_2': ['mean','sum'],
        'jockey_n_top_3': ['mean','sum'],
        'jockey_n_4th_or_below': ['mean','sum'],
        'jockey_stakes_participation': ['mean','sum'],
        'jockey_stakes_win': ['mean','sum'],
        'jockey_special_participation': ['mean','sum'],
        'jockey_special_win': ['mean','sum'],
        'jockey_flat_participation': ['mean','sum'],
        'jockey_lawn_participation': ['mean','sum'],
        'jockey_lawn_win': ['mean','sum'],
        'jockey_dirt_participation': ['mean','sum'],
        'jockey_dirt_win': ['mean','sum'],
        'jockey_win_proba': ['mean', 'max'],
        'jockey_top_2_proba': ['mean', 'max'],
        'jockey_top_3_proba': ['mean', 'max'],
        'jockey_earned_prize': ['mean', 'max', 'sum'],
    }
    
    # 参照年のデータについての特徴量を作成
    features = past_jockeys.groupby(['jockey_id','reference_year']).agg(aggregation_config).reset_index()
    features.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in features.columns]
    features.rename(columns={'jockey_id_': 'jockey_id','reference_year_':'reference_year'}, inplace=True)

    # 参照年の一つ前の年のデータについての特徴量を作成
    last_year_data = pd.DataFrame()
    past_jockeys_last_year = past_jockeys[past_jockeys['annual'] == past_jockeys['reference_year'].max() - 1]
    for col in past_jockeys_last_year.columns:
      if col in ['jockey_id']:
        last_year_data[col] = past_jockeys_last_year[col]
      elif col in ['annual']:
        continue
      else:
        last_year_data[f'{col}_last_year'] = past_jockeys_last_year[col]

    # 参照年よりさらに一つ前の年のデータについての特徴量を作成
    past_jockeys_last_two_year = past_jockeys[past_jockeys['annual'] >= past_jockeys['reference_year'].max() - 2]
    past_jockeys_last_two_year = past_jockeys_last_two_year.groupby(['jockey_id','reference_year']).agg(aggregation_config).reset_index()
    past_jockeys_last_two_year.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in past_jockeys_last_two_year.columns]
    past_jockeys_last_two_year.rename(columns={'jockey_id_': 'jockey_id','reference_year_':'reference_year'}, inplace=True)

    features = features.merge(last_year_data, on='jockey_id', how='left')
    features = features.merge(past_jockeys_last_two_year, on='jockey_id', how='left', suffixes=('', '_last_two_years'))

    features.fillna(0, inplace=True)

    return features