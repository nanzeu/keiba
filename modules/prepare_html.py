import time
from tqdm import tqdm
import os
from urllib.request import urlopen, Request
import urllib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .constants import local_paths, url_paths

def fetch_html_for_race(race_id: str, save_dir: str) -> str:
    """
    特定の race_id に対して HTML を取得し、save_dir に保存する。
    取得したファイルのパスを返す。
    """
    filepath = os.path.join(save_dir, f"{race_id}.bin")

    # もしすでに存在すればスキップ
    if os.path.isfile(filepath):
      return filepath

    try:
      url = url_paths.RACE_URL + str(race_id)
      html = urlopen(url).read()
      time.sleep(2)  # サーバー負荷軽減のためのウェイト
      with open(filepath, "wb") as f:
        f.write(html)
    except urllib.error.URLError as e:
      print(f"Error fetching race {race_id}: {e}")
      return None

    return filepath



def get_html_race(race_id_list: list[str], cs: bool = False) -> list[str]:
    """
    race_id から HTML を取得して save_dir に保存する。戻り値は html_path_list。
    """
    save_dir = local_paths.HTML_RACE_DIR

    if cs:
      save_dir = local_paths.HTML_RACE_CS_DIR

    html_path_list = []

    # 並行処理でレースの HTML を取得
    with ThreadPoolExecutor() as executor:
      # 各レース ID に対して fetch_html_for_race を非同期実行
      futures = {executor.submit(fetch_html_for_race, race_id, save_dir): race_id for race_id in race_id_list}

      # 処理が完了した順に結果を取得
      for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        if result:  # None でない場合のみ追加
          html_path_list.append(result)

    return html_path_list



def get_html_horse(
    horse_id_list: list[str],
    save_dir: str = local_paths.HTML_HORSE_DIR,
    skip: bool = True) -> list[str]:
  """
  horse_idはresultsの'horse_id'カラムから取得。
  horse_idからhtmlを取得してsave_dirに保存する。戻り値はhtml_path_list
  skip=Trueでファイルが存在する場合はスキップする。
  """
  html_path_list = []
  for horse_id in tqdm(horse_id_list):
    filepath = os.path.join(save_dir, f"{horse_id}.bin")
    html_path_list.append(filepath)

    # もしすでに存在すればスキップ
    if os.path.isfile(filepath) and skip:
      print(f"skipped: {horse_id}")

    else:
      try:
        url = url_paths.HORSE_URL + str(horse_id)
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        time.sleep(1)
        with open(filepath, "wb") as f:
          f.write(html)

      except urllib.error.URLError as e:
        print(e)
        continue
      
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
        url =  url_paths.JOCKEY_URL + str(jockey_id)
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        time.sleep(1)
        with open(filepath, "wb") as f:
          f.write(html)

      except urllib.error.URLError as e:
        print(e)
        continue
      
  return html_path_list