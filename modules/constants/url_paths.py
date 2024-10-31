# netkeiba.comの過去データベースのドメイン
DB_DOMAIN: str = "https://db.netkeiba.com/"
RACE_DOMAIN: str = "https://race.netkeiba.com/race/"

# 各ドメイン
RACE_URL: str = DB_DOMAIN + "race/"
HORSE_URL: str = DB_DOMAIN + "horse/"
JOCKEY_URL: str = DB_DOMAIN + "jockey/result/"
PED_URL: str = HORSE_URL + "ped/"

CANDIDATE_URL: str = RACE_DOMAIN + "shutuba.html?race_id="
