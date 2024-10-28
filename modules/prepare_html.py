import time
from tqdm import tqdm
import os
from urllib.request import urlopen
import urllib
import pandas as pd
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError
import threading

from .constants import local_paths, master, url_paths  # ローカルモジュールのインポート

def get_html_race(race_id_list: list[str], save_dir: str = local_paths.HTML_RACE_DIR) -> list[str]:
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
        url =  os.path.join(url_paths.RACE_URL, str(race_id))
        html = urlopen(url).read()
        time.sleep(1)
        with open(filepath, "wb") as f:
          f.write(html)

      except urllib.error.URLError as e:
        print(e)
        continue
      
  return html_path_list


def download_single_horse(args: tuple) -> str:
  """単一の馬のHTMLをダウンロードする関数"""
  horse_id, save_dir, skip = args
  filepath = os.path.join(save_dir, f"{horse_id}.bin")

  # ファイルが存在する場合のチェック
  if os.path.isfile(filepath) and skip:
    return filepath

  try:
    url = os.path.join(url_paths.HORSE_URL, str(horse_id))
    
    # URLOpenのタイムアウトを設定
    html = urllib.request.urlopen(url, timeout=10).read()
    
    # ファイルの書き込み
    with open(filepath, "wb") as f:
      f.write(html)
        
    # スリープはグローバルなレート制限に従う
    time.sleep(1)
    
    return filepath
  
  except Exception as e:
    print(f"Unexpected error for {horse_id}: {e}")
    return None
  

def get_html_horse(
  horse_id_list: list[str],
  save_dir: str = local_paths.HTML_HORSE_DIR,
  skip: bool = True,
  max_workers: int = 4,
  chunk_size: int = 1000
) -> list[str]:
  """
  horse_idからhtmlを取得して保存する関数（並列処理版）
  
  Parameters:
      horse_id_list: 馬IDのリスト
      save_dir: 保存先ディレクトリ
      skip: 既存ファイルをスキップするかどうか
      max_workers: 同時実行する最大スレッド数
      chunk_size: 一度に処理するIDの数
  """
  
  # 保存先ディレクトリの確認・作成
  os.makedirs(save_dir, exist_ok=True)
  
  # グローバルなレート制限用のセマフォ
  semaphore = threading.Semaphore(max_workers)
  
  def download_with_semaphore(args):
    with semaphore:
      return download_single_horse(args)
  
  html_path_list = []
  
  # チャンク単位で処理
  for i in range(0, len(horse_id_list), chunk_size):
    chunk_ids = horse_id_list[i:i + chunk_size]
    
    # 並列ダウンロードの実行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
      args_list = [(horse_id, save_dir, skip) for horse_id in chunk_ids]
      results = list(tqdm(
        executor.map(download_with_semaphore, args_list),
        total=len(chunk_ids),
        desc=f"Downloading chunk {i//chunk_size + 1}/{len(horse_id_list)//chunk_size + 1}"
      ))
    
    # 成功したダウンロードのパスを追加
    html_path_list.extend([path for path in results if path is not None])
    
    # チャンク間で少し待機（サーバー負荷軽減）
    time.sleep(2)
  
  return html_path_list



def get_html_ped(horse_id_list: list, save_dir: str):
  """
  horse_idからhtmlを取得してsave_dirに保存する。
  """
  html_path_list = []
  for horse_id in tqdm(horse_id_list):
    html_path = os.path.join(save_dir, horse_id + ".bin")
    html_path_list.append(html_path)

    if os.path.isfile(html_path):  # すでに存在するファイルならskipする
        print("horse_id {} skipped".format(horse_id))
        continue

    url = url_paths.PED_URL + horse_id
    html = urlopen(url).read()  # urlのhtml情報のスクレイピング

    time.sleep(1)

    with open(html_path, "wb") as f:
        f.write(html)  # スクレイピングしたhtmlの保存（書き込み）

  return html_path_list



def get_html_jockey(
    jockey_id_list: list, 
    save_dir: str = local_paths.HTML_JOCKEY_DIR, 
    skip: bool = True
  ) -> list[str]:
  """
  jockey_idからhtmlを取得してsave_dirに保存する。
  """
  html_path_list = []
  for jockey_id in tqdm(jockey_id_list):
    filepath = os.path.join(save_dir, f"{jockey_id}.bin")
    html_path_list.append(filepath)

    # もしすでに存在すればスキップ
    if os.path.isfile(filepath) and skip:
      print(f"skipped: {jockey_id}")

    else:
      try:
        url =  os.path.join(url_paths.JOCKEY_URL, str(jockey_id))
        html = urlopen(url).read()
        time.sleep(1)
        with open(filepath, "wb") as f:
          f.write(html)

      except urllib.error.URLError as e:
        print(e)
        continue
      
  return html_path_list


# def get_rawdata_results(html_path_list: list):
#     """
#     raceのhtmlを受け取って、レース結果テーブルに変換する。
#     """
#     race_results = {}
#     for html_path in tqdm(html_path_list):
#         with open(html_path, "rb") as f:
#             html = f.read()  # htmlの読み込み

#             df = pd.read_html(html)[0]  # html内のテーブルデータの取得

#             # horse_idとjockey_idの抜き出し
#             soup = BeautifulSoup(html, "html.parser")

#             # horse_idの取り出し
#             horse_id_list = []
#             horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
#                 "a", attrs={"href": re.compile("^/horse")}
#             )  # ^: 先頭の文字列に当てはまるもの
#             for a in horse_a_list:
#                 horse_id = re.findall(r"\d+", a["href"])
#                 horse_id_list.append(horse_id[0])

#             # jockey_idの取り出し
#             jockey_id_list = []
#             jockey_a_list = soup.find(
#                 "table", attrs={"summary": "レース結果"}
#             ).find_all(
#                 "a", attrs={"href": re.compile("^/jockey")}
#             )  # ^: 先頭の文字列に当てはまるもの
#             for a in jockey_a_list:
#                 jockey_id = re.findall(r"\d+", a["href"])
#                 jockey_id_list.append(jockey_id[0])

#             df["horse_id"] = horse_id_list
#             df["jockey_id"] = jockey_id_list

#             # race_idをインデックスに
#             race_id = re.findall(r"(?<=race\\)\d+", html_path)[0]
#             df.index = [race_id] * len(df)

#             race_results[race_id] = df

#     race_results_df = pd.concat([race_results[key] for key in race_results])

#     return race_results_df


# def get_rawdata_horse_results(html_path_list: list):
#     """
#     horseのhtmlを受け取って、レース結果テーブルに変換する。
#     """
#     horse_results = {}
#     for html_path in tqdm(html_path_list):
#         with open(html_path, "rb") as f:
#             html = f.read()  # htmlの読み込み

#             df = pd.read_html(html)[3]  # html内のテーブルデータの取得
#             if df.columns[0] == ["受賞歴"]:
#                 df = pd.read_html(html)[4]  # 受賞歴がある場合の処理

#             # horse_idをインデックスに
#             horse_id = re.findall(r"(?<=horse\\)\d+", html_path)[0]
#             df.index = [horse_id] * len(df)

#             horse_results[horse_id] = df

#     horse_results_df = pd.concat([horse_results[key] for key in horse_results])

#     return horse_results_df


# def get_rawdata_peds(html_path_list: list):
#     """
#     pedsのhtmlを受け取って、レース結果テーブルに変換する。
#     """
#     peds = {}
#     for html_path in tqdm(html_path_list):
#         with open(html_path, "rb") as f:
#             html = f.read()  # htmlの読み込み

#             df = pd.read_html(html)[0]

#             generations = {}
#             for j in reversed(range(5)):
#                 generations[j] = df[j]
#                 df.drop([j], axis=1, inplace=True)
#                 df = df.drop_duplicates()

#             horse_id = re.findall(r"(?<=ped\\)\d+", html_path)[0]
#             ped = pd.concat([generations[j] for j in range(5)]).rename(horse_id)
#             peds[horse_id] = ped.reset_index(drop=True)

#     # 結合
#     peds_df = pd.concat([peds[key] for key in peds], axis=1).T.add_prefix("peds_")
#     return peds_df


# def get_rawdata_infos(html_path_list: list):
#     """
#     raceのhtmlを受け取って、レース情報テーブルに変換する。
#     """
#     race_infos = {}
#     for html_path in tqdm(html_path_list):
#         with open(html_path, "rb") as f:
#             html = f.read()
#             soup = BeautifulSoup(html, "html.parser")

#             #  + \  : リスト同士の結合
#             texts = (
#                 soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
#                 + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
#             )
#             info = re.findall(r"\w+", texts)

#             # データの分別
#             df = pd.DataFrame()
#             for text in info:
#                 if text in master.RACE_TYPE_LIST:
#                     df["race_type"] = [text]
#                 if "障" in text:
#                     df["race_type"] = ["障害"]
#                 if "m" in text:
#                     df["course_len"] = [int(re.findall(r"\d+", text)[-1])]
#                 if text in master.GROUND_STATE_LIST:
#                     df["ground_state"] = [text]
#                 if text in master.WEATHER_LIST:
#                     df["weather"] = [text]
#                 if "年" in text:
#                     df["date"] = [text]

#             race_id = re.findall(r"(?<=race\\)\d+", html_path)[0]
#             df.index = [race_id] * len(df)
#             race_infos[race_id] = df

#     race_infos_df = pd.concat([race_infos[key] for key in race_infos])

#     return race_infos_df


# def get_rawdata_return(html_path_list: list):
#     """
#     raceのhtmlを受け取って、払い戻しテーブルに変換する。
#     """
#     return_tables = {}
#     for html_path in tqdm(html_path_list):
#         with open(html_path, "rb") as f:
#             html = f.read()  # htmlの読み込み

#             html = html.replace(b"<br />", b"br")
#             dfs = pd.read_html(html)  # html内のテーブルデータの取得

#             df = pd.concat([dfs[1], dfs[2]])

#             # race_idをインデックスに
#             race_id = re.findall(r"(?<=race\\)\d+", html_path)[0]
#             df.index = [race_id] * len(df)

#             return_tables[race_id] = df

#         # 結合
#         return_tables_df = pd.concat([return_tables[key] for key in return_tables])

#     return return_tables_df
