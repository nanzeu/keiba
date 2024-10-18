import pandas as pd
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import os

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
          jockey_id_list.append(jockey_id)
        df["jockey_id"] = jockey_id_list

        # trainer_id列追加
        a_list = soup.find_all("a", href=re.compile(r'^/trainer/'))
        trainer_id_list = []
        for a in a_list:
          trainer_id = re.findall(r'\d{5}', a["href"])[0]
          trainer_id_list.append(trainer_id)
        df["trainer_id"] = trainer_id_list

        # owner_id列追加        
        a_list = soup.find_all("a", href=re.compile(r'^/owner/'))
        owner_id_list = []
        for a in a_list:
          owner_id = re.findall(r'\d{6}', a["href"])[0]
          owner_id_list.append(owner_id)
        df["owner_id"] = owner_id_list

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


def create_race_info(
    html_paths_race: list[str],
    save_dir: str = local_paths.RAW_DIR,
    save_filename: str = "race_info.csv",
  ) -> pd.DataFrame:
  dfs = {}
  for html_path in tqdm(html_paths_race):
    with open(html_path, "rb") as f:
      try:
        html = f.read()
        soup = BeautifulSoup(html, "lxml").find('div', class_='data_intro')
        info_dict = {}
        info_dict['title'] = soup.find('h1').text
        p_list = soup.find_all('p')
        info_dict['info1'] = re.findall(r'[\w+:]+',p_list[0].text.replace(' ', ''))
        info_dict['info2'] = re.findall(r'\w+', p_list[1].text)

        df = pd.DataFrame().from_dict(info_dict, orient='index').T

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


def create_peds(
    html_paths_horse: list[str],
    save_dir: str = local_paths.RAW_DIR,
    save_filename: str = "peds.csv",
) -> pd.DataFrame:
    
    dfs = []  # 各データフレームを保存するリスト
    for html_path in tqdm(html_paths_horse):
      try:
        horse_id = re.search(r'\d{10}', html_path).group()  # horse_idを一度だけ抽出
        df = pd.read_html(html_path)[2]  # ファイルを直接`pd.read_html`で読み込む
        # horse_id列を追加
        df['horse_id'] = horse_id
        dfs.append(df)  # リストにデータフレームを追加

      except (IndexError, FileNotFoundError) as e:
        print(f"Error processing {html_path}: {e}")
        continue

    # 最後にすべてのデータフレームを一度に結合
    concat_df = pd.concat(dfs, ignore_index=True)
    # horse_idをインデックスとして設定
    concat_df.set_index('horse_id', inplace=True)
    # インデックスがhorse_idのままCSVに保存
    concat_df.to_csv(os.path.join(save_dir, save_filename), sep="\t")
    return concat_df
