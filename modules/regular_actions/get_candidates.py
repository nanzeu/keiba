from modules import prepare_new_data, scraping, prepare_html, prepare_rawdata, preprocessing
from modules.constants import local_paths 

from datetime import datetime, timedelta
import pandas as pd
import re



def save_data(cs: bool = False):
  # 今月＆来月の開催日を取得
  # 現在の月の次の月を計算
  current_date = datetime.now()
  next_month_date = (current_date.replace(day=1) + timedelta(days=31)).replace(day=1)
  if cs:
    _, date_id_dict = scraping.get_race_date_list(
      f'{current_date.year}-{current_date.month}', f'{next_month_date.year}-{next_month_date.month}', cs=cs
    )
    loop_target = date_id_dict.items()

  else:
    race_date_list, _ = scraping.get_race_date_list(
      f'{current_date.year}-{current_date.month}', f'{next_month_date.year}-{next_month_date.month}', cs=cs
    )
    loop_target = race_date_list

  for race_date in loop_target:
    # csがTrueの場合、race_dateはタプルになるためkeyとvalueに分解
    if cs:
      race_date, id = race_date  # タプルの最初の要素をrace_dateに設定
    
    race_date_obj = datetime.strptime(race_date, '%Y%m%d')
    if (race_date_obj - timedelta(days=1)).date() == datetime.now().date():
      # レース前日の場合、その日のrace_id_listからデータを取得し、
      # candidates、candidates_infoとして保存し前処理する。
      if cs:
        race_id_list = scraping.get_race_id_list(race_date_list=None, date_id_dict={race_date: id} ,cs=cs)

        # スキップ対象の開催場所のみの場合処理を終了
        skip_pattern = re.compile(r'^\d{4}(65|55|54|45|44|46|36|51)\d*')
        filtered_list = [item for item in race_id_list if not skip_pattern.match(item)]

        if len(filtered_list) == 0:
          print(f'Not found candidates-cs {race_date}')
          break

      else:
        race_id_list = scraping.get_race_id_list(race_date_list=[race_date], date_id_dict=None, cs=cs)

      html_paths_candidates = prepare_new_data.get_html_candidates(race_id_list, cs=cs)
      candidates = prepare_new_data.create_candidates(html_paths_candidates, cs=cs)
      prepare_new_data.create_candidates_info(html_paths_candidates, cs=cs)
      prepare_new_data.process_candidates(cs=cs)

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
    

def get_candidates():
  save_data()
  save_data(cs=True)


# スクリプトが直接実行されたときに関数を呼び出す
if __name__ == "__main__":
  get_candidates()
