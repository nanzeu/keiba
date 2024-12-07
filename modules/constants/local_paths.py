import os


# プロジェクトルートの絶対パス
BASE_DIR: str = os.path.abspath("./")  # ./  =  c:\Users\PC_User\p\keiba

# dataディレクトリまでのパス
DATA_DIR: str = os.path.join(BASE_DIR, "data")  # './data'

# htmlディレクトリまでのパス
HTML_DIR: str = os.path.join(DATA_DIR, "html")  # './data/html'
HTML_RACE_DIR: str = os.path.join(HTML_DIR, "race")  # './data/html/race'
HTML_RACE_CS_DIR: str = os.path.join(HTML_DIR, "race_cs")  # './data/html/race_cs'
HTML_HORSE_DIR: str = os.path.join(HTML_DIR, "horse")  # './data/html/horse'
HTML_PED_DIR: str = os.path.join(HTML_DIR, "ped")  # './data/html/ped'
HTML_JOCKEY_DIR: str = os.path.join(HTML_DIR, "jockey")  # './data/html/jockey'


# datesディレクトリまでのパス
DATES_DIR: str = os.path.join(DATA_DIR, "dates") # './data/dates

# listsディレクトリまでのパス
LISTS_DIR: str = os.path.join(DATA_DIR, "lists") # './data/lists

# preprocessedディレクトリまでのパス
PREPROCESSED_DIR: str = os.path.join(DATA_DIR, "preprocessed") # './data/preprocessed

# rawディレクトリまでのパス
RAW_DIR: str = os.path.join(DATA_DIR, "raw") # './data/raw

# preprocessed_csディレクトリまでのパス
PREPROCESSED_CS_DIR = os.path.join(DATA_DIR, "preprocessed_cs")

# raw_csディレクトリまでのパス
RAW_CS_DIR = os.path.join(DATA_DIR, "raw_cs")

# mappingディレクトリまでのパス
MAPPING_DIR: str = os.path.join(DATA_DIR, "mapping") # './data/mapping

# featuresディレクトリまでのパス
FEATURES_DIR: str = os.path.join(DATA_DIR, "features_V6.0") # './data/features
FEATURES_CS_DIR: str = os.path.join(FEATURES_DIR, "cs") # './data/features/cs

# completedディレクトリまでのパス
COMPLETED_DIR: str = os.path.join(DATA_DIR, "completed_data") # './data/completed_data
COMPLETED_CS_DIR: str = os.path.join(DATA_DIR, "completed_data_cs") # './data/completed_data_cs

# modelsディレクトリまでのパス
MODELS_DIR: str = os.path.join(DATA_DIR, "models") # './data/models

# candidatesディレクトリまでのパス
CANDIDATES_DIR: str = os.path.join(DATA_DIR, "candidates") # './data/candidates
CANDIDATES_CS_DIR: str = os.path.join(CANDIDATES_DIR, "cs") # './data/candidates/cs'