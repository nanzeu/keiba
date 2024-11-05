import time
from tqdm import tqdm
import os
from urllib.request import urlopen
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
      url = os.path.join(url_paths.RACE_URL, str(race_id))
      html = urlopen(url).read()
      time.sleep(1)  # サーバー負荷軽減のためのウェイト
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
  max_workers: int = 6,
  chunk_size: int = 1000,
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