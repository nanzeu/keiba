import pandas as pd
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.constants import local_paths


def process_html_path(html_path: str) -> pd.DataFrame:
  """
  指定された HTML パスを処理し、必要な DataFrame を返す。
  """
  try:
    race_id = re.search(r'\d{12}', html_path).group()
    with open(html_path, "rb") as f:
      html = f.read()
      soup = BeautifulSoup(html, "lxml").find("table", class_="race_table_01 nk_tb_common")
      df = pd.read_html(html)[0]

      # horse_id 列追加
      a_list = soup.find_all("a", href=re.compile(r'^/horse/'))
      horse_id_list = [re.findall(r'\d{10}', a["href"])[0] for a in a_list]
      df["horse_id"] = horse_id_list

      # jockey_id 列追加
      a_list = soup.find_all("a", href=re.compile(r'^/jockey/'))
      jockey_id_list = [str(re.findall(r'(?<=/jockey/result/recent/)[\w\d]+', a["href"])[0]).zfill(5) for a in a_list]
      df["jockey_id"] = jockey_id_list

      # trainer_id 列追加
      a_list = soup.find_all("a", href=re.compile(r'^/trainer/'))
      trainer_id_list = [str(re.findall(r'(?<=/trainer/result/recent/)[\w\d]+', a["href"])[0]).zfill(5) for a in a_list]
      df["trainer_id"] = trainer_id_list

    # race_id をインデックスに設定
    df.index = [race_id] * len(df)
    return df

  except (IndexError, AttributeError) as e:
    print(f"Error processing {html_path}: {e}")
    return pd.DataFrame()  # エラー時は空の DataFrame を返す



def create_results(
  html_paths_race: list[str],
  save_dir: str = local_paths.RAW_DIR,
  save_filename: str = "results.csv",
  cs: bool = False
) -> pd.DataFrame:
    """
    複数の HTML ファイルを並行処理で読み込み、指定の保存先に CSV ファイルとして保存する。
    """
    if cs:
      save_dir = local_paths.RAW_CS_DIR

      # 条件に合わないファイルだけを処理
      skip_pattern = re.compile(r'^\d{4}(65|55|54|45|44|46|36|51)\d*')
      html_paths_race = [path for path in html_paths_race if not skip_pattern.search(os.path.basename(path))]

    dfs = {}
    with ThreadPoolExecutor() as executor:
      futures = {}
      for html_path in html_paths_race:
        # 条件を満たさない場合のみ、並行処理で処理を実行
        futures[executor.submit(process_html_path, html_path)] = html_path

      for future in tqdm(as_completed(futures), total=len(futures)):
        df = future.result()
        if not df.empty:
          race_id = df.index[0]
          dfs[race_id] = df

    # 結果の結合と保存
    concat_df = pd.concat(dfs.values())
    concat_df.index.name = "race_id"
    concat_df.columns = concat_df.columns.str.replace(' ', '')
    concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")
    return concat_df



def process_html_file(html_path: str) -> pd.DataFrame:
  """
  指定された HTML ファイルから馬の結果情報を DataFrame として取得。
  """
  try:
    with open(html_path, "rb") as f:
      horse_id = re.search(r'\d{10}', html_path).group()
      html = f.read()
      df = pd.read_html(html)[2]
      df['horse_id'] = horse_id  # horse_id列を追加
      return df
  except (IndexError, ValueError) as e:
    print(f"table not found at {html_path}")

  return None

def create_horse_results(
  html_paths_horse: list[str],
  save_dir: str = local_paths.RAW_DIR,
  save_filename: str = "horse_results.csv",
  cs: bool = False
) -> pd.DataFrame:
  """
  HTML ファイルのリストからデータを並行処理で読み込み、結果を CSV ファイルとして保存。
  """
  if cs:
    save_dir = local_paths.RAW_CS_DIR

  dfs = []

  # 並行処理で HTML ファイルを読み込む
  with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_html_file, html_path): html_path for html_path in html_paths_horse}

    for future in tqdm(as_completed(futures), total=len(futures)):
      df = future.result()
      if df is not None:
          dfs.append(df)

  # 全データフレームを結合
  concat_df = pd.concat(dfs)
  concat_df.columns = concat_df.columns.str.replace(' ', '')
  concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")
  return concat_df



def process_html_path(html_path: str) -> pd.DataFrame:
  """
  指定された HTML パスを処理し、DataFrame を返す。
  """
  try:
    with open(html_path, "rb") as f:
      html = f.read()
      soup = BeautifulSoup(html, "lxml").find('div', class_='data_intro')
      info_dict = {}
      info_dict['title'] = soup.find('h1').text
      p_list = soup.find_all('p')
      info_dict['info1'] = re.findall(r'[\w+:]+', p_list[0].text.replace(' ', ''))
      info_dict['info2'] = re.findall(r'\w+', p_list[1].text)

      df = pd.DataFrame().from_dict(info_dict, orient='index').T

      race_id = re.search(r'\d{12}', html_path).group()
      df.index = [race_id] * len(df)
      return df

  except (IndexError, AttributeError) as e:
    print(f"Error processing {html_path}: {e}")
    return pd.DataFrame()  # エラー時は空の DataFrame を返す



def create_race_info(
  html_paths_race: list[str],
  save_dir: str = local_paths.RAW_DIR,
  save_filename: str = "race_info.csv",
  cs: bool = False
) -> pd.DataFrame:
  
  if cs:
    save_dir = local_paths.RAW_CS_DIR
    # 条件に合わないファイルだけを処理
    skip_pattern = re.compile(r'^\d{4}(65|55|54|45|44|46|36|51)\d*')
    html_paths_race = [path for path in html_paths_race if not skip_pattern.search(os.path.basename(path))]

  # 並行処理で HTML ファイルを処理
  dfs = {}
  with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_html_path, html_path): html_path for html_path in html_paths_race}

    for future in tqdm(as_completed(futures), total=len(futures)):
      df = future.result()
      if not df.empty:
        race_id = df.index[0]
        dfs[race_id] = df

  # 結果の結合と保存
  concat_df = pd.concat(dfs.values())
  concat_df.index.name = "race_id"
  concat_df.columns = concat_df.columns.str.replace(' ', '')
  concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")
  return concat_df





def process_html_path(html_path: str) -> pd.DataFrame:
  try:
    with open(html_path, "rb") as f:
      html = f.read()
      soup = BeautifulSoup(html, "lxml").find_all('table', class_='pay_table_01')

      # 結果をまとめるリストにtrの内容を一度に格納
      tr_list = soup[0].find_all('tr') + soup[1].find_all('tr')

      # returns_dictをリスト内包表記で一括作成
      returns_dict = {
        tr.find('th').text: [
          int(n.replace(',', ''))
          for td in tr.find_all('td')
          for n in re.findall(r'\d{1,3}(?:,\d{3})*', str(td))
        ]
        for tr in tr_list
      }

      # DataFrameに変換
      df = pd.DataFrame({key: [value] for key, value in returns_dict.items()})

      # race_idの抽出とインデックス設定
      race_id = re.search(r'\d{12}', html_path).group()
      df.index = [race_id] * len(df)
      return race_id, df
  except (IndexError, ValueError, AttributeError) as e:
    print(f"table not found at {html_path}: {e}")
    return None
  


def create_returns(
  html_paths_race: list[str], 
  save_dir: str = local_paths.RAW_DIR, 
  save_filename: str = "returns.csv", 
  cs: bool = False
) -> pd.DataFrame:
  
  if cs:
    save_dir = local_paths.RAW_CS_DIR

  # 並行処理を用いたHTMLファイルの読み込みと処理
  dfs = {}
  with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_html_path, html_path): html_path for html_path in html_paths_race}
    for future in tqdm(as_completed(futures), total=len(futures)):
      result = future.result()
      if result is not None:
        race_id, df = result
        dfs[race_id] = df

  # 結合と保存
  concat_df = pd.concat(dfs.values())
  concat_df.index.name = "race_id"
  concat_df.columns = concat_df.columns.str.replace(' ', '')
  concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")
  return concat_df



def process_peds_html_file(html_path: str) -> pd.DataFrame:
  try:
    horse_id = re.search(r'\d{10}', html_path).group()  # horse_idを抽出
    df = pd.read_html(html_path)[1]  # `pd.read_html`でHTMLファイルを読み込み

    # horse_id列を追加
    df['horse_id'] = horse_id
    return df

  except (IndexError, FileNotFoundError, ValueError) as e:
    print(f"Error processing {html_path}: {e}")
    return None



def create_peds(
  html_paths_horse: list[str],
  save_dir: str = local_paths.RAW_DIR,
  save_filename: str = "peds.csv",
  max_workers: int = 8,  # 並列処理のスレッド数
  cs: bool = False
) -> pd.DataFrame:
  
  if cs:
    save_dir = local_paths.RAW_CS_DIR

  # 並列処理を利用して各HTMLファイルを処理
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(process_peds_html_file, html_paths_horse), total=len(html_paths_horse)))

  # Noneを除外してデータフレームを連結
  dfs = [df for df in results if df is not None]
  concat_df = pd.concat(dfs, ignore_index=True)

  # horse_idをインデックスとして設定
  concat_df.set_index('horse_id', inplace=True)

  # 保存処理
  concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")

  return concat_df



def process_jockey_html_file(html_path: str) -> pd.DataFrame:
  try:
    # HTMLファイルを読み込む
    with open(html_path, "rb") as f:
      jockey_id = re.search(r'([^\\]{5})\.bin$', html_path).group(1)  # jockey_idを抽出
      
      html = f.read()
      df = pd.read_html(html)[0]

      # jockey_id列を追加
      df['jockey_id'] = jockey_id
      return df

  except (IndexError, FileNotFoundError, ValueError) as e:
    print(f"Error processing {html_path}: {e}")
    return None
  except Exception as e:
    print(f"Unexpected error processing {html_path}: {e}")
    return None



def create_jockeys(
  html_paths_jockey: list[str],
  save_dir: str = local_paths.RAW_DIR,
  save_filename: str = "jockeys.csv",
  max_workers: int = 8,  # 並列処理のスレッド数
  cs: bool = False
) -> pd.DataFrame:
  
  if cs:
    save_dir = local_paths.RAW_CS_DIR

  # 並列処理を利用して各HTMLファイルを処理
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(tqdm(executor.map(process_jockey_html_file, html_paths_jockey), total=len(html_paths_jockey)))

  # Noneを除外してデータフレームを連結
  dfs = [df for df in results if df is not None]
  concat_df = pd.concat(dfs, ignore_index=True)

  # jockey_idをインデックスとして設定
  concat_df.set_index('jockey_id', inplace=True)

  # マルチインデックスのカラム名を結合
  new_columns = []
  seen_columns = set()  # 重複するカラム名のチェック用
  for col in concat_df.columns:
    # マルチインデックスかどうかを確認
    if isinstance(col, tuple):
      main_col = col[0]  # 第一要素を取得
      # 重複チェック
      if main_col not in seen_columns:
        new_columns.append(main_col)  # 重複なし：第一要素のみ追加
        seen_columns.add(main_col)  # チェックリストに追加
      else:
        new_columns.append('_'.join(col))  # 重複あり：全部結合
    else:
      new_columns.append(col)
  concat_df.columns = new_columns

  # 保存処理
  concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")

  return concat_df



