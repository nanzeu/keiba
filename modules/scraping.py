import re
from tqdm import tqdm
import datetime
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import time
import os
from modules.constants import local_paths

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def get_race_date_list(start: str, end: str, cs: bool = False) -> list[str]:
  dstart = datetime.datetime.strptime(start, '%Y-%m')
  dend = datetime.datetime.strptime(end, '%Y-%m')
  start_year = dstart.year
  end_year = dend.year
  start_month = dstart.month
  end_month = dend.month

  race_date_list = []
  date_id_dict = {}
  for year in range(start_year, end_year+1):
    for month in tqdm(range(start_month, end_month+1)):
      if cs:
        url = f'https://nar.netkeiba.com/top/calendar.html?year={year}&month={month}'
      else:
        url = f'https://race.netkeiba.com/top/calendar.html?year={year}&month={month}'

      req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
      html = urlopen(req).read()
      soup = BeautifulSoup(html, "html.parser")

      time.sleep(1)

      # リンクのある部分（開催日）をスクレイピング
      date = soup.find_all("a", attrs={"href": re.compile(r'kaisai_date=[0-9]+')})

      # race_dateの取得
      for tag in date:
        race_date = re.search(r'kaisai_date=([0-9]+)', tag["href"]).group(1)
        race_date_list.append(race_date)

        # kaisai_idの取得
        if cs:
          kaisai_id_list = []
          # race_date に対応する kaisai_id を取得
          id = soup.find_all("a", attrs={"href": re.compile(rf'kaisai_date={race_date}&kaisai_id=[0-9]+')})

          for tag in id:
            kaisai_id = re.search(r'kaisai_id=([0-9]+)', tag["href"]).group(1)
            kaisai_id_list.append(kaisai_id)

          date_id_dict[race_date] = kaisai_id_list


  return race_date_list, date_id_dict



def get_race_id_list(
    race_date_list: list[str] = None,
    date_id_dict: dict[str] = None, 
    cs: bool = False
  ) -> list[str]:

  options = Options()
  options.add_argument("--headless=new")
  driver_path = ChromeDriverManager().install()

  race_id_list = []
  with webdriver.Chrome(service=Service(driver_path), options=options) as driver:
    if cs:
      for race_date in tqdm(date_id_dict.keys()):
        for kaisai_id in date_id_dict[race_date]:
          url = f'https://nar.netkeiba.com/top/race_list.html?kaisai_date={race_date}&kaisai_id={kaisai_id}'
          driver.get(url)
          time.sleep(1)
          li_list = driver.find_elements(By.CLASS_NAME, "RaceList_DataItem")
          for li in li_list:
            href = li.find_element(By.TAG_NAME, "a").get_attribute("href")
            race_id = re.findall(r'race_id=(\d{12})', href)[0]
            race_id_list.append(race_id)

    else:
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