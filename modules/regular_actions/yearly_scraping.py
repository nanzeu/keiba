from modules import scraping
from modules.constants import local_paths

from apscheduler.schedulers.background import BackgroundScheduler
import pickle
import os
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

def scrape_data(cs: bool = False):
  try:
    # 現在の年を取得
    current_year = datetime.now().year

    # データの取得
    race_date_list = scraping.get_race_date_list(f'{current_year}-1', f'{current_year}-12', cs=cs)
    race_id_list =scraping.get_race_id_list(race_date_list, cs=cs)

    # ディレクトリが存在しなければ作成
    os.makedirs(local_paths.DATES_DIR, exist_ok=True)
    os.makedirs(local_paths.LISTS_DIR, exist_ok=True)

    if cs:
      with open(os.path.join(local_paths.DATES_DIR, f'race_date_list_{current_year}_cs.pickle'), mode='wb') as f:
        pickle.dump(race_date_list, f)
      
      with open(os.path.join(local_paths.LISTS_DIR, f'race_id_list_{current_year}_cs.pickle'), mode='wb') as f:
        pickle.dump(race_id_list, f)

    else:
      with open(os.path.join(local_paths.DATES_DIR, f'race_date_list_{current_year}.pickle'), mode='wb') as f:
        pickle.dump(race_date_list, f)
      
      with open(os.path.join(local_paths.LISTS_DIR, f'race_id_list_{current_year}.pickle'), mode='wb') as f:
        pickle.dump(race_id_list, f)

  except Exception as e:
    logging.error(f"Error occurred during data scraping: {e}")


if __name__ == "__main__":
  scrape_data()
  scrape_data(cs=True)