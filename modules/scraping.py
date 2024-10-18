import re
from tqdm import tqdm
import datetime
from bs4 import BeautifulSoup
from urllib.request import urlopen
import time

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def get_race_date_list(start: str, end: str) -> list[str]:
  dstart = datetime.datetime.strptime(start, '%Y-%m')
  dend = datetime.datetime.strptime(end, '%Y-%m')
  start_year = dstart.year
  end_year = dend.year
  start_month = dstart.month
  end_month = dend.month

  race_date_list = []
  for year in range(start_year, end_year+1):
    for month in tqdm(range(start_month, end_month+1)):
      url = f'https://race.netkeiba.com/top/calendar.html?year={year}&month={month}'
      html = urlopen(url)
      soup = BeautifulSoup(html, "html.parser")

      time.sleep(1)

      # リンクのある部分（開催日）をスクレイピング
      a = soup.find_all("a", attrs={"href": re.compile(r'../top/race_list\.html.kaisai_date=[0-9]+')})

      # 開催日のリスト化
      for tag in a:
        race_date = re.search(r'[0-9]+', tag["href"]).group()
        race_date_list.append(race_date)
  
  return race_date_list


def get_race_id_list(race_date_list: list[str]) -> list[str]:
  options = Options()
  options.add_argument("--headless=new")
  driver_path = ChromeDriverManager().install()

  race_id_list = []
  with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
    for race_date in tqdm(race_date_list):
      url = f'https://race.netkeiba.com/top/race_list.html?kaisai_date={race_date}'
      driver.get(url)
      time.sleep(1)
      li_list = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
      for li in li_list:
        href = li.find_element(By.TAG_NAME, "a").get_attribute("href")
        race_id = re.findall(r'race_id=(\d{12})', href)[0]
        race_id_list.append(race_id)
  
  return race_id_list