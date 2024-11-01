from modules import preapre_new_data, scraping
from modules.constants import local_paths 

import time
from datetime import datetime, timedelta
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
      preapre_new_data.create_candidates(html_paths_candidates)
      preapre_new_data.create_candidates_info(html_paths_candidates)
      preapre_new_data.process_candidates()
      
      print(f"HTML data saved for race ID: {race_id_list}")

    else:
      continue

