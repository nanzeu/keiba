import pandas as pd
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import os
import time
from urllib.request import urlopen
import urllib
import json
from datetime import datetime, timedelta

from modules.constants import local_paths, url_paths



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



def get_html_candidates(
  race_id_list: list[str], 
  save_dir: str = local_paths.HTML_CANDIDATES_DIR,
) -> list[str]:
  """
  race_idからhtmlを取得してsave_dirに保存する。戻り値はhtml_path_list
  """

  html_path_list = []
  for race_id in tqdm(race_id_list):
    filepath = os.path.join(save_dir, f"{race_id}.bin")
    html_path_list.append(filepath)

    # もしすでに存在すればスキップ
    if os.path.isfile(filepath):
      print(f"skipped: {race_id}")

    else:
      try:
        url = url_paths.CANDIDATE_URL + str(race_id)
        html = urlopen(url).read()
        time.sleep(1)
        with open(filepath, "wb") as f:
          f.write(html)

      except urllib.error.URLError as e:
        print(e)
        continue
      
  return html_path_list



def create_candidates(
    html_paths_candidates: list[str],
    save_dir: str = local_paths.CANDIDATES_DIR,
    save_filename: str = "candidates.csv",
  ) -> pd.DataFrame:
  """
  html_paths_candidatesから出馬表データを取得
  """
  dfs = {}

  for html_path in tqdm(html_paths_candidates):
    with open(html_path, "rb") as f:
      try:
        race_id = re.search(r'\d{12}', html_path).group()
        html = f.read()
        df = pd.read_html(html)[0]
        soup = BeautifulSoup(html, "lxml").find(
          "table", class_="ShutubaTable"
        )
        
        df.columns = df.columns.droplevel(0)

        a_list = soup.find_all("a", href=re.compile(r'/horse/'))
        horse_id_list = []
        for a in a_list:
          horse_id = re.search(r'\d{10}', a["href"]).group()
          horse_id_list.append(horse_id)
        df["horse_id"] = horse_id_list

        # jockey_id列追加
        a_list = soup.find_all("a", href=re.compile(r'/jockey/'))
        jockey_id_list = []
        for a in a_list:
          jockey_id = re.search(r'\d{5}', a["href"]).group()
          jockey_id = str(jockey_id).zfill(5)
          jockey_id_list.append(jockey_id)
        df["jockey_id"] = jockey_id_list

        # trainer_id列追加
        a_list = soup.find_all("a", href=re.compile(r'/trainer/'))
        trainer_id_list = []
        for a in a_list:
          trainer_id = re.search(r'\d{5}', a["href"]).group()
          trainer_id = str(trainer_id).zfill(5)
          trainer_id_list.append(trainer_id)
        df["trainer_id"] = trainer_id_list

        df.index = [race_id] * len(df)
        dfs[race_id] = df
      
      except IndexError as e:
        print(f"table not found at {race_id}")
        continue

  concat_df = pd.concat(dfs.values())
  concat_df.index.name = "race_id"
  concat_df.columns = concat_df.columns.str.replace(' ', '')
  concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")
  return concat_df



def create_candidates_info(
  html_paths_candidates: list[str],
  output_dir: str = local_paths.CANDIDATES_DIR,
  save_filename: str = "candidates_info.csv",
  weather_mapping: dict = weather_mapping,
  race_class_mapping: dict = race_class_mapping,
  race_type_mapping: dict = race_type_mapping,
  ground_state_mapping: dict = ground_state_mapping,
  around_mapping: dict = around_mapping,
  place_mapping: dict = place_mapping,
) -> pd.DataFrame:
  
  date = (datetime.now() - timedelta(days=1)).date()

  dfs = {}
  for html_path in tqdm(html_paths_candidates):
    with open(html_path, "rb") as f:
      try:
        html = f.read()
        soup = BeautifulSoup(html, "lxml")

        df = pd.DataFrame()
        span_list = soup.find_all('span')
        div_list = soup.find_all('div', class_="RaceData01")

        for span in span_list:
          type_len = re.search(r'(ダ|芝|障)(\d+)', span.text)

          regex_place = '|'.join(place_mapping.keys())
          place = re.search(regex_place, span.text)

          regex_race_class = '|'.join(race_class_mapping.keys())
          race_class = re.search(regex_race_class, span.text)

          regex_ground_state = '|'.join(ground_state_mapping.keys())
          ground_state = re.search(regex_ground_state, span.text)

          if type_len:
            type_len_re = re.search(r'([^\d])(\d+)', span.text)
            df['race_type'] = type_len_re.group(1).split()
            df['race_type'] = df['race_type'].map(race_type_mapping)
            df['course_len'] = type_len_re.group(2).split()
          
          if place:
            df['place'] = place.group()
            df['place'] = df['place'].map(place_mapping)

          if race_class:
            df['race_class'] = race_class.group()
            df['race_class'] = df['race_class'].map(race_class_mapping)

          if ground_state:
            df['ground_state'] = ground_state.group()
            df['ground_state'] = df['ground_state'].map(ground_state_mapping)
            
        for div in div_list:
          regex_weather = '|'.join(weather_mapping.keys())
          weather = re.search(regex_weather, div.text)

          regex_around = '|'.join(around_mapping.keys())
          around = re.search(regex_around, div.text)

          if weather:
            weather = re.search(r'(?<=天候:)\S+', div.text)
            df['weather'] = weather.group()
            df['weather'] = df['weather'].map(weather_mapping)
          
          if around:
            df['around'] = around.group()
            df['around'] = df['around'].map(around_mapping)

        df['date'] = date

      except IndexError as e:
        print(f"table not found at {race_id}")
        continue

      race_id = re.search(r'\d{12}', html_path).group()
      df.index = [race_id] * len(df)
      dfs[race_id] = df

  concat_df = pd.concat(dfs.values())
  concat_df.index.name = "race_id"
  concat_df.columns = concat_df.columns.str.replace(' ', '')
  concat_df.to_csv(os.path.join(output_dir, save_filename), sep="\t")
  
  return concat_df
          


def process_candidates(
    input_dir: str = local_paths.CANDIDATES_DIR,
    output_dir: str = local_paths.CANDIDATES_DIR,
    save_file_name: str = "candidates.csv",
    sex_mapping: dict = sex_mapping,
):
  # データの読み込み
  df = pd.read_csv(os.path.join(input_dir, save_file_name), sep="\t")

  # データを加工
  df['jockey_id'] = df['jockey_id'].astype(str).str.zfill(5)
  df['trainer_id'] = df['trainer_id'].astype(str).str.zfill(5)
  df['frame'] = df['枠'].astype(int)
  df['number'] = df['馬番'].astype(int)
  df['sex'] = df['性齢'].str[0].map(sex_mapping)
  df['age'] = df['性齢'].str[1:].astype(int)
  df['impost'] = df['斤量'].astype(float)
  df['weight'] = pd.to_numeric(df['馬体重(増減)'].str.extract(r'(\d+)')[0], errors="coerce")
  df['weight_diff'] = pd.to_numeric(df['馬体重(増減)'].str.extract(r'\((.+)\)')[0], errors="coerce").fillna(0)

  # 着順ではなく、race_id順にしてリークを防ぐ
  df = df.sort_values(['race_id', 'number'])

  # 不要なカラムを削除
  df = df.drop(
    columns=[
      '枠', 
      '馬番', 
      '印', 
      '馬名', 
      '性齢', 
      '斤量', 
      '騎手', 
      '厩舎', 
      '馬体重(増減)',
      'Unnamed:9_level_1', 
      '人気', 
      '登録', 
      'メモ',
    ]
  )

  df.to_csv(os.path.join(output_dir, save_file_name), sep="\t")

  return df