from modules.constants import local_paths
from modules import create_features, predict
from modules.predict import Net  # Net クラスのインポート

import pandas as pd
import os
import torch
import pickle
from joblib import load
import json
from datetime import datetime, timedelta


def predict_data(cs: bool = False):
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
      race_date, _ = race_date  # タプルの最初の要素をrace_dateに設定
    
    race_date_obj = datetime.strptime(race_date, '%Y%m%d')
    if (race_date_obj - timedelta(days=1)).date() == datetime.now().date():
      if not cs:
        candidates_path = local_paths.CANDIDATES_DIR
      else:
        candidates_path = local_paths.CANDIDATES_CS_DIR

      h = create_features.Horse(
        output_dir=candidates_path,
        save_filename="horse_features.csv",
        race_info_filename='candidates_info.csv',
        results_filename='candidates.csv', 
        horse_results_filename='horse_results.csv',
        peds_filename='peds.csv',
        cs=cs,
        new=True
      )

      j = create_features.Jockey(
        output_dir=candidates_path,
        save_filename=f"jockey_features.csv",
        race_info_filename=f'candidates_info.csv',
        results_filename=f'candidates.csv', 
        jockeys_filename=f'jockeys.csv',
        cs=cs,
        new=True
      )

      horse_features = pd.read_csv(os.path.join(candidates_path, f'horse_features.csv'),index_col=0,  sep="\t")
      jockey_features = pd.read_csv(os.path.join(candidates_path, f'jockey_features.csv'),index_col=0,  sep="\t", dtype={'jockey_id': str})
      results = pd.read_csv(os.path.join(candidates_path, f'candidates.csv')\
                            , index_col=0,  sep="\t", dtype={'jockey_id': str, 'trainer_id': str, 'owner_id': str})
      race_info = pd.read_csv(os.path.join(candidates_path, f'candidates_info.csv'), index_col=0,  sep="\t")

      results_add_info = results.merge(race_info, on='race_id', how='left')
      results_add_horse_features = results_add_info.merge(
        horse_features,
        left_on=['horse_id', 'date'],
        right_on=['horse_id', 'reference_date'],
        how='left'
      )

      results_add_horse_features['date'] = pd.to_datetime(results_add_horse_features['date'], errors='coerce')
      results_add_horse_features['year'] = results_add_horse_features['date'].dt.year

      features = results_add_horse_features.merge(jockey_features, left_on=['jockey_id', 'year'], right_on=['jockey_id', 'reference_year'], how='left')

      print(features)

      # jockeyに関連するデータの欠損値を0で埋める
      features.loc[:, features.columns.str.contains('jockey', case=False)] =\
          features.loc[:, features.columns.str.contains('jockey', case=False)].fillna(0)
      
      if not cs:
        # 新たにモデルを初期化
        torch.serialization.add_safe_globals([Net])
        en_nn_basemodel = Net(input_size=30)

        # 保存されたモデル、重みを読み込み
        en_nn_basemodel.load_state_dict(torch.load(os.path.join(local_paths.MODELS_DIR, 'en_nn_basemodel.pth')))

        with open(os.path.join(local_paths.MODELS_DIR, 'en_rf_basemodel.pickle'), 'rb') as f:
          en_rf_basemodel = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_lgb_basemodel.pickle'), 'rb') as f:
          en_lgb_basemodel = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_xgb_basemodel.pickle'), 'rb') as f:
          en_xgb_basemodel = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_lgb_metamodel.pickle'), 'rb') as f:
          en_lgb_metamodel = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_rf_basemodel_features.pickle'), 'rb') as f:
          en_rf_basemodel_features = pickle.load(f)
        en_nn_basemodel_features= load(os.path.join(local_paths.MODELS_DIR, 'en_nn_basemodel_features.joblib'))
        with open(os.path.join(local_paths.MODELS_DIR, 'en_lgb_basemodel_features.pickle'), 'rb') as f:
          en_lgb_basemodel_features = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_xgb_basemodel_features.pickle'), 'rb') as f:
          en_xgb_basemodel_features = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_lgb_metamodel_features.pickle'), 'rb') as f:
          en_lgb_metamodel_features = pickle.load(f)

        base_models = {'rf': en_rf_basemodel, 'nn': en_nn_basemodel, 'lgb': en_lgb_basemodel, 'xgb': en_xgb_basemodel}
        base_models_features = {'rf': en_rf_basemodel_features, 'nn': en_nn_basemodel_features, 
                                'lgb': en_lgb_basemodel_features, 'xgb': en_xgb_basemodel_features}
        meta_models = {'lgb': en_lgb_metamodel}
        meta_models_features = {'lgb': en_lgb_metamodel_features}

        # 予想
        en = predict.EnsembleModel(
          train_df=None, returns_df=None, bet_type='sanrenpuku', threshold=0.6, 
          max_bet=800, pivot_horse=True, select_num=30, final_model='lgb', save=False,
          base_models=base_models, meta_models=meta_models, base_models_features=base_models_features, meta_models_features=meta_models_features, 
        )

      else:
       # 新たにモデルを初期化
        torch.serialization.add_safe_globals([Net])
        en_nn_basemodel = Net(input_size=30)

        # 保存されたモデル、重みを読み込み
        en_nn_basemodel.load_state_dict(torch.load(os.path.join(local_paths.MODELS_DIR, 'en_nn_basemodel_cs.pth')))

        with open(os.path.join(local_paths.MODELS_DIR, 'en_rf_basemodel_cs.pickle'), 'rb') as f:
          en_rf_basemodel = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_lgb_basemodel_cs.pickle'), 'rb') as f:
          en_lgb_basemodel = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_xgb_basemodel_cs.pickle'), 'rb') as f:
          en_xgb_basemodel = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_xgb_metamodel_cs.pickle'), 'rb') as f:
          en_xgb_metamodel = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_rf_basemodel_cs_features.pickle'), 'rb') as f:
          en_rf_basemodel_features = pickle.load(f)
        en_nn_basemodel_features= load(os.path.join(local_paths.MODELS_DIR, 'en_nn_basemodel_cs_features.joblib'))
        with open(os.path.join(local_paths.MODELS_DIR, 'en_lgb_basemodel_cs_features.pickle'), 'rb') as f:
          en_lgb_basemodel_features = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_xgb_basemodel_cs_features.pickle'), 'rb') as f:
          en_xgb_basemodel_features = pickle.load(f)
        with open(os.path.join(local_paths.MODELS_DIR, 'en_xgb_metamodel_cs_features.pickle'), 'rb') as f:
          en_xgb_metamodel_features = pickle.load(f)

        base_models = {'rf': en_rf_basemodel, 'nn': en_nn_basemodel, 'lgb': en_lgb_basemodel, 'xgb': en_xgb_basemodel}
        base_models_features = {'rf': en_rf_basemodel_features, 'nn': en_nn_basemodel_features, 
                                'lgb': en_lgb_basemodel_features, 'xgb': en_xgb_basemodel_features}
        meta_models = {'xgb': en_xgb_metamodel}
        meta_models_features = {'xgb': en_xgb_metamodel_features}

        # 予想
        en = predict.EnsembleModel(
          train_df=None, returns_df=None, bet_type='umaren', threshold=0.6, 
          max_bet=600, pivot_horse=True, select_num=50, final_model='xgb', cs=True, save=False,
          base_models=base_models, meta_models=meta_models, base_models_features=base_models_features, meta_models_features=meta_models_features, 
        )

      pred = en.predict_target(features)
      pred_bet = en.calc_results(pred, per_race=False, bet_only=True)

      pred_target = pred[pred['predicted_target'] == 1][['race_id', 'horse_id', 'number', 'place','predicted_proba', 'predicted_target']]

      pred_df = pred_target.merge(pred_bet, on=['race_id'], how='left')

      with open(os.path.join(local_paths.MAPPING_DIR, 'place.json'), 'rb') as f:
        place_mapping = json.load(f)

      reversed_place = {v: k for k, v in place_mapping.items()}
      pred_df['place'] = pred_df['place'].replace(reversed_place)

      # 賭けないところは排除
      pred_df.dropna(inplace=True)


      if cs:
        full_save_name = f'pred_candidates_full_{(datetime.now() + timedelta(days=1)).date().strftime("%Y%m%d")}_cs.csv'
        save_name = f'pred_candidates_{(datetime.now() + timedelta(days=1)).date().strftime("%Y%m%d")}_cs.csv'
      else:
        full_save_name = f'pred_candidates_full_{(datetime.now() + timedelta(days=1)).date().strftime("%Y%m%d")}.csv'
        save_name = f'pred_candidates_{(datetime.now() + timedelta(days=1)).date().strftime("%Y%m%d")}.csv'

        # pred_dfをルートに保存
        pred_df.to_csv(
          full_save_name, sep="\t", encoding='utf-8'
        )
        pred_df[(pred_df['bet_sum'] > 0)].to_csv(
          save_name, sep="\t", encoding='utf-8'
        )

def predict_candidates():
  predict_data()
  predict_data(cs=True)
  

if __name__ == '__main__':
  predict_candidates()