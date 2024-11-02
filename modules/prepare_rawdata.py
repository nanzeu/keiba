import pandas as pd
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import os
from concurrent.futures import ThreadPoolExecutor

from modules.constants import local_paths


def create_results(
    html_paths_race: list[str],
    save_dir: str = local_paths.RAW_DIR,
    save_filename: str = "results.csv"
  ) -> pd.DataFrame:
  dfs = {}
  for html_path in tqdm(html_paths_race):
    with open(html_path, "rb") as f:
      try:
        race_id = re.search(r'\d{12}', html_path).group()
        html = f.read()
        soup = BeautifulSoup(html, "lxml").find(
          "table", class_="race_table_01 nk_tb_common"
        )
        df = pd.read_html(html)[0]

        # horse_id列追加
        a_list = soup.find_all("a", href=re.compile(r'^/horse/'))
        horse_id_list = []
        for a in a_list:
          horse_id = re.findall(r'\d{10}', a["href"])[0]
          horse_id_list.append(horse_id)
        df["horse_id"] = horse_id_list

        # jockey_id列追加
        a_list = soup.find_all("a", href=re.compile(r'^/jockey/'))
        jockey_id_list = []
        for a in a_list:
          jockey_id = re.findall(r'\d{5}', a["href"])[0]
          jockey_id = str(jockey_id).zfill(5)
          jockey_id_list.append(jockey_id)
        df["jockey_id"] = jockey_id_list

        # trainer_id列追加
        a_list = soup.find_all("a", href=re.compile(r'^/trainer/'))
        trainer_id_list = []
        for a in a_list:
          trainer_id = re.findall(r'\d{5}', a["href"])[0]
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


def process_html_file(html_path: str) -> pd.DataFrame:
  try:
    with open(html_path, "rb") as f:
      horse_id = re.search(r'\d{10}', html_path).group()
      html = f.read()
      df = pd.read_html(html)[3]

      # horse_idの列を直接データフレームに追加
      df['horse_id'] = horse_id
      return df
  except (IndexError, ValueError) as e:
    print(f"table not found at {html_path}")
    return None
  
  

def create_horse_results(
    html_paths_horse: list[str],
    save_dir: str = local_paths.RAW_DIR,
    save_filename: str = "horse_results.csv",
  ) -> pd.DataFrame:
  dfs = {}
  for html_path in tqdm(html_paths_horse):
    with open(html_path, "rb") as f:
      try:
        horse_id = re.search(r'\d{10}', html_path).group()
        html = f.read()
        df = pd.read_html(html)[3]

        df.index = [horse_id] * len(df)
        dfs[horse_id] = df

      except IndexError as e:
        print(f"table not found at {horse_id}")
        continue

  concat_df = pd.concat(dfs.values())
  concat_df.index.name = "horse_id"
  concat_df.columns = concat_df.columns.str.replace(' ', '')
  concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")
  return concat_df



def create_returns(
  html_paths_race: list[str],
  save_dir: str = local_paths.RAW_DIR,
  save_filename: str = "returns.csv",
) -> pd.DataFrame:
  dfs = {}
  for html_path in tqdm(html_paths_race):
    with open(html_path, "rb") as f:
      try:
        html = f.read()
        soup = BeautifulSoup(html, "lxml").find_all('table', class_='pay_table_01')

        tr_list = soup[0].find_all('tr')
        for tr in soup[1].find_all('tr'):
          tr_list.append(tr)

        returns_dict={}
        for tr in tr_list:
          key = tr.find('th').text
          tds = tr.find_all('td')
          returns_dict[key] = []
          for n in re.findall(r'\d{1,3}(?:,\d{3})*', str(tds[0])):
            returns_dict[key].append(int(n.replace(',', '')))
          for n in re.findall(r'\d{1,3}(?:,\d{3})*', str(tds[1])):
            returns_dict[key].append(int(n.replace(',', '')))

        df = pd.DataFrame({key: [value] for key, value in returns_dict.items()})

        race_id = re.search(r'\d{12}', html_path).group()
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



def process_peds_html_file(html_path: str) -> pd.DataFrame:
  try:
    horse_id = re.search(r'\d{10}', html_path).group()  # horse_idを抽出
    df = pd.read_html(html_path)[2]  # `pd.read_html`でHTMLファイルを読み込み

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
  max_workers: int = 8  # 並列処理のスレッド数
) -> pd.DataFrame:
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
      jockey_id = re.search(r'\d{5}', html_path).group()  # jockey_idを抽出
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
  max_workers: int = 8  # 並列処理のスレッド数
) -> pd.DataFrame:
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



