from modules import preapre_new_data, scraping, prepare_html, prepare_rawdata, preprocessing
from modules.constants import local_paths 

from datetime import datetime, timedelta
import pandas as pd
import pickle
import os


def save_data():
  with open(os.path.join(local_paths.DATES_DIR, f'race_date_list_{datetime.now().year}.pickle'), 'rb') as f:
    race_date_list = pickle.load(f)

  with open(os.path.join(local_paths.LISTS_DIR, f'race_id_list_{datetime.now().year}.pickle'), 'rb') as f:
    race_id_list = pickle.load(f)
  
  for race_date in race_date_list:
    race_date_obj = datetime.strptime(race_date, '%Y%m%d')
    if (race_date_obj - timedelta(days=1)).date() == datetime.now().date():
      # レース前日の場合、その日のrace_id_listからデータを取得し、
      # candidates、candidates_infoとして保存し前処理する。
      race_id_list = scraping.get_race_id_list([race_date])
      html_paths_candidates = preapre_new_data.get_html_candidates(race_id_list)
      candidates = preapre_new_data.create_candidates(html_paths_candidates)
      preapre_new_data.create_candidates_info(html_paths_candidates)
      preapre_new_data.process_candidates()

      # 馬、騎手データを取得し、保存する（更新されたデータも取得したいので、skipしない）
      html_paths_horse = prepare_html.get_html_horse(candidates['horse_id'].unique().tolist(), skip=False)
      html_paths_jockeys = prepare_html.get_html_jockey(candidates['jockey_id'].unique().tolist(), skip=False)
      prepare_rawdata.create_horse_results(html_paths_horse, save_dir=local_paths.CANDIDATES_DIR)
      prepare_rawdata.create_jockeys(html_paths_jockeys, save_dir=local_paths.CANDIDATES_DIR)
      prepare_rawdata.create_peds(html_paths_horse, save_dir=local_paths.CANDIDATES_DIR)
      preprocessing.process_horse_results(input_dir=local_paths.CANDIDATES_DIR, output_dir=local_paths.CANDIDATES_DIR)
      preprocessing.process_jockeys(input_dir=local_paths.CANDIDATES_DIR, output_dir=local_paths.CANDIDATES_DIR)
      preprocessing.process_peds(input_dir=local_paths.CANDIDATES_DIR, output_dir=local_paths.CANDIDATES_DIR)

    else:
      continue


# スクリプトが直接実行されたときに関数を呼び出す
if __name__ == "__main__":
  save_data()

