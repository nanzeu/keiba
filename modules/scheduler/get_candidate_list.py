from modules import preapre_new_data, scraping
from modules.constants import local_paths 

from apscheduler.schedulers.background import BackgroundScheduler
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
      # レース前日の場合その日のrace_id_listからデータを取得し、candidatesとして保存
      race_id_list = scraping.get_race_id_list(race_date)
      html_paths_candidates = preapre_new_data.get_html_candidates(race_id_list)
      print(f"HTML data saved for race ID: {race_id_list}")

      with open(os.path.join(local_paths.DATES_DIR, f'race_date_list_{datetime.now().year}.pickle'), 'rb') as f:
        race_date_list = pickle.load(f)

      for race_date in race_date_list:
        if race_date == (datetime.now().date() + timedelta(days=1)).strftime('%Y%m%d'):
          date = datetime.strptime(race_date, '%Y%m%d').date()
        else:
          continue



scheduler = BackgroundScheduler()

# 毎日15時に実行（前日でなければスルー）
scheduler.add_job(save_data, 'cron', hour=15)
scheduler.start()

if __name__ == "__main__":
  save_data()