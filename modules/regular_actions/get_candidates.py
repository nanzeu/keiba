from modules import preapre_new_data, scraping, prepare_html, prepare_rawdata, preprocessing
from modules.constants import local_paths 

from datetime import datetime, timedelta
import pandas as pd
import pickle
import os


def save_data(cs: bool = False):
  if cs:
    with open(os.path.join(local_paths.DATES_DIR, f'date_id_dict_{datetime.now().year}_cs.pickle'), 'rb') as f:
      date_id_dict = pickle.load(f)
    # date_id_dict.items()を使用する
    loop_target = date_id_dict.items()
  else:
    with open(os.path.join(local_paths.DATES_DIR, f'race_date_list_{datetime.now().year}.pickle'), 'rb') as f:
      race_date_list = pickle.load(f)
    # race_date_listをそのまま使用する
    loop_target = race_date_list

  for race_date in loop_target:
    # csがTrueの場合、race_dateはタプルになるためkeyとvalueに分解
    if cs:
        race_date, id = race_date  # タプルの最初の要素をrace_dateに設定
    
    race_date_obj = datetime.strptime(race_date, '%Y%m%d')
    if (race_date_obj - timedelta(days=1)).date() == datetime.now().date():
      # レース前日の場合、その日のrace_id_listからデータを取得し、
      # candidates、candidates_infoとして保存し前処理する。
      if not cs:
        race_id_list = scraping.get_race_id_list(race_date_list=[race_date], date_id_dict=None, cs=cs)
      else:
        race_id_list = scraping.get_race_id_list(race_date_list=None, date_id_dict={race_date: id} ,cs=cs)
      html_paths_candidates = preapre_new_data.get_html_candidates(race_id_list, cs=cs)
      candidates = preapre_new_data.create_candidates(html_paths_candidates, cs=cs)
      preapre_new_data.create_candidates_info(html_paths_candidates, cs=cs)
      preapre_new_data.process_candidates(cs=cs)

      if cs:
        save_dir = local_paths.CANDIDATES_CS_DIR
      else:
        save_dir = local_paths.CANDIDATES_DIR

      # 馬、騎手データを取得し、保存する（更新されたデータも取得したいので、skipしない）
      html_paths_horse = prepare_html.\
        get_html_horse(horse_id_list=candidates['horse_id'].unique().tolist(), save_dir=save_dir, skip=False)
      html_paths_jockeys = prepare_html.\
        get_html_jockey(jockey_id_list=candidates['jockey_id'].unique().tolist(), save_dir=save_dir, skip=False)
      prepare_rawdata.create_horse_results(html_paths_horse, save_dir=save_dir)
      prepare_rawdata.create_jockeys(html_paths_jockeys, save_dir=save_dir)
      prepare_rawdata.create_peds(html_paths_horse, save_dir=save_dir)
      preprocessing.process_horse_results(input_dir=save_dir, output_dir=save_dir)
      preprocessing.process_jockeys(input_dir=save_dir, output_dir=save_dir)
      preprocessing.process_peds(input_dir=save_dir, output_dir=save_dir)

    else:
      continue


# スクリプトが直接実行されたときに関数を呼び出す
if __name__ == "__main__":
  save_data()

