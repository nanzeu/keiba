import os
import shutil
import time

# 対象ディレクトリと削除の基準日数
DIRECTORY_PATH = "data/candidates"
EXPIRATION_DAYS = 6

def reset_directory_if_expired(directory_path, expiration_days):
  # ディレクトリが存在するか確認
  if os.path.exists(directory_path):
      # ディレクトリの最終更新日時を取得
      last_modified = os.path.getmtime(directory_path)
      current_time = time.time()
      
      # 経過日数を計算
      days_since_modified = (current_time - last_modified) / (60 * 60 * 24)
      
      # 経過日数が指定日数を超えている場合、ディレクトリを削除して再作成
      if days_since_modified > expiration_days:
        print(f"{expiration_days}日以上経過しているため、{directory_path}を削除します。")
        shutil.rmtree(directory_path)
        os.makedirs(directory_path)
        print(f"{directory_path}を再作成しました。")
      else:
        print(f"{expiration_days}日未満のため、{directory_path}は削除しません。")
  else:
      # ディレクトリが存在しない場合、新規作成
      os.makedirs(directory_path)
      print(f"{directory_path}を新規作成しました。")

  # .gitkeepファイルを追加
  gitkeep_path = os.path.join(directory_path, ".gitkeep")
  with open(gitkeep_path, "w") as f:
      pass


if __name__ == "__main__":
  reset_directory_if_expired(DIRECTORY_PATH, EXPIRATION_DAYS)