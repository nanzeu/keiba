import pandas as pd
import os
import json
import re
import numpy as np
from tqdm import tqdm

from modules.constants import local_paths, master

with open(os.path.join(local_paths.MAPPING_DIR, "sex.json"), 'r',encoding='utf-8_sig') as f:
  sex_mapping = json.load(f)

with open(os.path.join(local_paths.MAPPING_DIR, "weather.json"), 'r',encoding='utf-8_sig') as f:
  weather_mapping = json.load(f)

with open(os.path.join(local_paths.MAPPING_DIR, "race_class.json"), 'r',encoding='utf-8_sig') as f:
  race_class_mapping = json.load(f)

with open(os.path.join(local_paths.MAPPING_DIR, "race_type.json"), 'r',encoding='utf-8_sig') as f:
  race_type_mapping = json.load(f)

with open(os.path.join(local_paths.MAPPING_DIR, "ground_state.json"), 'r',encoding='utf-8_sig') as f:
  ground_state_mapping = json.load(f)

with open(os.path.join(local_paths.MAPPING_DIR, "around.json"), 'r',encoding='utf-8_sig') as f:
  around_mapping = json.load(f)

with open(os.path.join(local_paths.MAPPING_DIR, "place.json"), 'r',encoding='utf-8_sig') as f:
  place_mapping = json.load(f)



def process_results(
    input_dir: str = local_paths.RAW_DIR,
    output_dir: str = local_paths.PREPROCESSED_DIR,
    save_file_name: str = "results.csv",
    sex_mapping: dict = sex_mapping,
    cs: bool = False
) -> pd.DataFrame:
  """
  input_dirからrawdataを取得し、output_dirに加工したデータを保存する
  """
  if cs:
    input_dir = local_paths.RAW_CS_DIR
    output_dir = local_paths.PREPROCESSED_CS_DIR

  # データの読み込み
  df = pd.read_csv(os.path.join(input_dir, save_file_name), sep="\t", index_col=0)

  # データを加工
  df['jockey_id'] = df['jockey_id'].astype(str).str.zfill(5)
  df['trainer_id'] = df['trainer_id'].astype(str).str.zfill(5)
  df['rank'] = pd.to_numeric(df['着順'], errors="coerce") 
  df.dropna(subset=['rank'], inplace=True)
  df['frame'] = df['枠番'].astype(int)
  df['number'] = df['馬番'].astype(int)
  df['sex'] = df['性齢'].str[0].map(sex_mapping)
  df['age'] = df['性齢'].str[1:].astype(int)
  df['impost'] = df['斤量'].astype(float)
  df['win_odds'] = df['単勝'].astype(float)
  df['popularity'] = df['人気'].astype(int)

  # 着順ではなく、race_id順にしてリークを防ぐ
  df = df.sort_values(['race_id', 'number'])

  # 不要なカラムを削除
  df = df.drop(
    columns=[
      '着順',
      '枠番',
      '馬番',
      '馬名',
      '性齢',
      '斤量',
      '騎手',
      'タイム',
      '着差',
      '単勝',
      '人気',
      '馬体重',
      '調教師',
    ]
  )

  df.to_csv(os.path.join(output_dir, save_file_name), sep="\t")

  return df


def process_horse_results(
    input_dir: str = local_paths.RAW_DIR,
    output_dir: str = local_paths.PREPROCESSED_DIR,
    save_file_name: str = "horse_results.csv",
    weather_mapping: dict = weather_mapping,
    race_class_mapping: dict = race_class_mapping,
    race_type_mapping: dict = race_type_mapping,
    ground_state_mapping: dict = ground_state_mapping,
    cs: bool = False
) -> pd.DataFrame:
  """
  input_dirからrawdataを取得し、output_dirに加工したデータを保存する
  """
  if cs:
    input_dir = local_paths.RAW_CS_DIR
    output_dir = local_paths.PREPROCESSED_CS_DIR
  
  # データの読み込み
  df = pd.read_csv(os.path.join(input_dir, save_file_name), sep="\t")

  # データを加工
  df['date'] = pd.to_datetime(df['日付'])
  df['weather'] = df['天気'].map(weather_mapping)
  regex_race_class = "|".join(race_class_mapping.keys())
  df['race_class'] = df['レース名'].str.extract(rf"({regex_race_class})")[0].map(race_class_mapping)
  df.rename(columns={'頭数': 'n_horses'}, inplace=True)
  df['rank'] = pd.to_numeric(df['着順'], errors="coerce") 
  df.dropna(subset=['rank'], inplace=True)
  df['race_type'] = df['距離'].str[0].map(race_type_mapping)
  df['course_len'] = df['距離'].str.extract(r'(\d+)').astype(int)
  df['ground_state'] = df['馬場'].map(ground_state_mapping)
  df['rank_diff'] = df['着差'].map(lambda x: 0 if x < 0 else x)
  df['3_furlongs'] = df['上り'].astype(float)
  df['prize'] = df['賞金'].fillna(0)
  time_list = []
  for time in df['タイム']: 
    # NaNのチェック
    if pd.isna(time):
      time_list.append(np.nan)
      continue
    else:
      try:
        times = str(time).split(':')
        total_seconds = int(times[0]) * 60 + float(times[1])
        time_list.append(total_seconds)
      except:
        time_list.append(np.nan)
  df['time'] = time_list

  # 必要なカラム
  df = df[
    [
      'horse_id',
      'date',
      'weather',
      'race_class',
      'n_horses',
      'rank',
      'race_type',
      'course_len',
      'ground_state',
      'rank_diff',
      '3_furlongs',
      'time',
      'prize'
    ]
  ]

  df.to_csv(os.path.join(output_dir, save_file_name), sep="\t")

  return df


def process_race_info(
    input_dir: str = local_paths.RAW_DIR,
    output_dir: str = local_paths.PREPROCESSED_DIR,
    save_file_name: str = "race_info.csv",
    cs: bool = False,
    weather_mapping: dict = weather_mapping,
    race_class_mapping: dict = race_class_mapping,
    race_type_mapping: dict = race_type_mapping,
    ground_state_mapping: dict = ground_state_mapping,
    around_mapping: dict = around_mapping,
    place_mapping: dict = place_mapping,
) -> pd.DataFrame:
  """
  input_dirからrawdataを取得し、output_dirに加工したデータを保存する
  """

  if cs: 
    input_dir = local_paths.RAW_CS_DIR
    output_dir = local_paths.PREPROCESSED_CS_DIR
  
  # データの読み込み
  df = pd.read_csv(os.path.join(input_dir, save_file_name), sep="\t")

  # 各情報データのリストを抽出
  info1_list = []
  info2_list = []
  for n in range(len(df)):
    info1 = df['info1'][n].replace("'", '').replace("[", '').replace("]", '').split(',')
    info2 = df['info2'][n].replace("'", '').replace("[", '').replace("]", '').split(',')
    info1_list.append(info1)
    info2_list.append(info2)

  # 各リストから必要なデータを抽出
  df_p = pd.DataFrame()
  df_p['race_id'] = df['race_id']
  info1_df = pd.DataFrame({'info1': info1_list})
  info2_df = pd.DataFrame({'info2': info2_list})
  df_p['date'] = info2_df['info2'].apply(lambda x : x[0])
  df_p['date'] = pd.to_datetime(df_p['date'], format='%Y年%m月%d日')
  df_p[['race_type', 'around']] = info1_df['info1'].apply(lambda x: pd.Series(list(x[0][:2])))
  df_p['race_type'] = df_p['race_type'].map(race_type_mapping)
  df_p['around'] = df_p['around'].map(around_mapping)
  course_len_list = []
  for info1 in info1_list:
    course_len = re.findall(r'\d+', info1[0])
    if course_len and int(course_len[0]) > 500:
      course_len_list.append(course_len[0])
    else:
      course_len_list.append(0)
  df_p['course_len'] = course_len_list
  df_p['course_len'] = df_p['course_len'].astype(int).replace(0, np.nan)
  df_p['weather'] = info1_df['info1'].apply(lambda x: x[1].split(':'))
  df_p['weather'] = df_p['weather'].apply(lambda x: x[1] if len(x) > 1 else None)
  df_p['weather'] = df_p['weather'].map(weather_mapping)
  df_p['ground_state'] = info1_df['info1'].apply(lambda x: x[2].split(':'))
  df_p['ground_state'] = df_p['ground_state'].apply(lambda x: None if x[1] in master.WEATHER_LIST else x[1])
  df_p['ground_state'] = df_p['ground_state'].map(ground_state_mapping)
  if not cs:
    regex_race_class = "|".join(race_class_mapping.keys())
    df_p['race_class'] = df['title'].str.extract(rf"({regex_race_class})")[0].map(race_class_mapping)
  df_p['place'] = info2_df['info2'].apply(lambda x: x[1]) 
  df_p['place'] = df_p['place'].str.extract(r'\d回(.*?)\d日目')
  df_p['place'] = df_p['place'].map(place_mapping)

  df_p.to_csv(os.path.join(output_dir, save_file_name), sep="\t")

  return df_p



def process_returns(
  input_dir: str = local_paths.RAW_DIR,
  output_dir: str = local_paths.PREPROCESSED_DIR,
  save_file_name: str = "returns.csv",
  cs: bool = False
) -> pd.DataFrame:
  
  if cs:
    input_dir = local_paths.RAW_CS_DIR
    output_dir = local_paths.PREPROCESSED_CS_DIR
  
  df = pd.read_csv(os.path.join(input_dir, save_file_name), index_col=0, sep="\t")

  # 結果を格納するためのデータフレーム
  concat_df = pd.DataFrame()

  # 各レースに対して処理を行う
  for race_id in df.index:
    df_t = pd.DataFrame(index=[race_id])  # レースごとのデータフレーム
    df_t.index.name = 'race_id'  # race_id をインデックスに設定

    # 各ターゲットカラムに対して処理を行う
    for col in df.columns:
      rank_list = []
      returns_list = []
      cell_value = df.loc[race_id, col]
      if isinstance(cell_value, str):  # 文字列ならば処理を適用
        values = cell_value.replace("'", '').replace("[", '').replace("]", '').split(',')
      else:
        values = []  # 非文字列の場合、空リストとして扱う（もしくはNaNをスキップ）
      
      # 値を100以下と100以上に分けて処理
      for n in values:
        if int(n) < 100:
          rank_list.append(int(n))  # 100以下は着順リストへ
        else:
          returns_list.append(int(n))  # 100以上は払い戻しリストへ

      # 新しいカラムを追加
      df_t[f'{col}_rank'] = [rank_list]  # 着順リスト
      df_t[f'{col}_returns'] = [returns_list]  # 払い戻しリスト

    # 結果を結合
    concat_df = pd.concat([concat_df, df_t])
  concat_df.to_csv(os.path.join(output_dir, save_file_name), sep="\t")
  return concat_df




def process_peds(
    input_dir: str = local_paths.RAW_DIR,
    output_dir: str = local_paths.PREPROCESSED_DIR,
    save_file_name: str = "peds.csv",
    cs: bool = False
) -> pd.DataFrame:
  
  if cs:
    input_dir = local_paths.RAW_CS_DIR
    output_dir = local_paths.PREPROCESSED_CS_DIR
  
  df = pd.read_csv(os.path.join(input_dir, save_file_name), index_col=0 , sep="\t")

  # horse_idをキーにした辞書型を使用して親と祖父母の情報を格納
  horse_parents_dict = {}

  # horse_idごとにデータを集約
  for horse_id, group in tqdm(df.groupby('horse_id')):
    # 親の情報を取得
    parents = group['0'].tolist()  
    parents = list(dict.fromkeys(parents))  # 親の重複を排除
    grandparents = group['1'].tolist()  # 祖父母
    
    # horse_idに親と祖父母を対応させる辞書を作成
    horse_parents_dict[horse_id] = {
      'parent_0': parents[0] if len(parents) > 0 else None,
      'parent_1': parents[1] if len(parents) > 1 else None,
      'parent_2': grandparents[0] if len(grandparents) > 0 else None,
      'parent_3': grandparents[1] if len(grandparents) > 1 else None,
      'parent_4': grandparents[2] if len(grandparents) > 2 else None,
      'parent_5': grandparents[3] if len(grandparents) > 3 else None,
    }

  # 辞書からデータフレームに変換
  concat_df = pd.DataFrame.from_dict(horse_parents_dict, orient='index')

  concat_df.index.name = 'horse_id'
  concat_df.to_csv(os.path.join(output_dir, save_file_name), sep="\t")
  return concat_df



def process_jockeys(
    input_dir: str = local_paths.RAW_DIR,
    output_dir: str = local_paths.PREPROCESSED_DIR,
    save_file_name: str = "jockeys.csv",
    cs: bool = False
) -> pd.DataFrame:
  
  if cs:
    input_dir = local_paths.RAW_CS_DIR
    output_dir = local_paths.PREPROCESSED_CS_DIR
  
  df = pd.read_csv(os.path.join(input_dir, save_file_name), sep="\t", dtype={'jockey_id': str})

  df = df.dropna()

  df = df[df['年度'] != '累計']

  df.loc[:, 'annual'] = df['年度'].astype(int)
  df.loc[:, 'jockey_rank'] = df['順位'].astype(int)
  df.loc[:, 'jockey_n_top_1'] = df['1着'].astype(int)
  df.loc[:, 'jockey_n_top_2'] = df['2着'].astype(int)
  df.loc[:, 'jockey_n_top_3'] = df['3着'].astype(int)
  df.loc[:, 'jockey_n_4th_or_below'] = df['着外'].astype(int)
  df.loc[:, 'jockey_stakes_participation'] = df['重賞'].astype(int)
  df.loc[:, 'jockey_stakes_win'] = df['重賞_勝利'].astype(int)
  df.loc[:, 'jockey_special_participation'] = df['特別'].astype(int)
  df.loc[:, 'jockey_special_win'] = df['特別_勝利'].astype(int)
  df.loc[:, 'jockey_flat_participation'] = df['平場'].astype(int)
  df.loc[:, 'jockey_lawn_participation'] = df['芝'].astype(int)
  df.loc[:, 'jockey_lawn_win'] = df['芝_勝利'].astype(int)
  df.loc[:, 'jockey_dirt_participation'] = df['ダート'].astype(int)
  df.loc[:, 'jockey_dirt_win'] = df['ダート_勝利'].astype(int)
  df.loc[:, 'jockey_win_proba'] = df['勝率'].astype(int)
  df.loc[:, 'jockey_top_2_proba'] = df['連対 率'].astype(float)
  df.loc[:, 'jockey_top_3_proba'] = df['複勝 率'].astype(float)
  df.loc[:, 'jockey_earned_prize'] = df['収得賞金 (万円)'].astype(float)

  df = df[
    [
    'annual',
    'jockey_id',
    'jockey_rank',
    'jockey_n_top_1',
    'jockey_n_top_2',
    'jockey_n_top_3',
    'jockey_n_4th_or_below',
    'jockey_stakes_participation',
    'jockey_stakes_win',
    'jockey_special_participation',
    'jockey_special_win',
    'jockey_flat_participation',
    'jockey_lawn_participation',
    'jockey_lawn_win',
    'jockey_dirt_participation',
    'jockey_dirt_win',
    'jockey_win_proba',
    'jockey_top_2_proba',
    'jockey_top_3_proba',
    'jockey_earned_prize',
    ]
  ]

      
  df.to_csv(os.path.join(output_dir, save_file_name), sep="\t")
  return df