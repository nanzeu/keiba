from modules.constants import local_paths
from modules import create_features, predict

import pandas as pd
import os
import torch
import pickle
from joblib import load


def predict_candidates():
  h = create_features.Horse(
    output_dir=local_paths.CANDIDATES_DIR,
    save_filename="horse_features.csv",
    race_info_filename='candidates_info.csv',
    results_filename='candidates.csv', 
    horse_results_filename='horse_results_20_to_23.csv',
    peds_filename='peds_20_to_23.csv',
    new=True
  )

  j = create_features.Jockey(
    output_dir=local_paths.CANDIDATES_DIR,
    save_filename=f"jockey_features.csv",
    race_info_filename=f'candidates_info.csv',
    results_filename=f'candidates.csv', 
    jockeys_filename=f'jockeys_20_to_23.csv',
    new=True
  )

  horse_features = pd.read_csv(os.path.join(local_paths.CANDIDATES_DIR, f'horse_features.csv'),index_col=0,  sep="\t")
  jockey_features = pd.read_csv(os.path.join(local_paths.CANDIDATES_DIR, f'jockey_features.csv'),index_col=0,  sep="\t", dtype={'jockey_id': str})
  results = pd.read_csv(os.path.join(local_paths.CANDIDATES_DIR, f'candidates.csv')\
                        , index_col=0,  sep="\t", dtype={'jockey_id': str, 'trainer_id': str, 'owner_id': str})
  race_info = pd.read_csv(os.path.join(local_paths.CANDIDATES_DIR, f'candidates_info.csv'), index_col=0,  sep="\t")

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

  # jockeyに関連するデータの欠損値を0で埋める
  features.loc[:, features.columns.str.contains('jockey', case=False)] =\
      features.loc[:, features.columns.str.contains('jockey', case=False)].fillna(0)

  # 新たにモデルを初期化
  en_nn_basemodel = Net(input_size=30)
  # 保存された重みを読み込み
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

  en = predict.EnsembleModel(
    train_df=None, returns_df=None, bet_type='sanrenpuku', threshold=0.6, 
    stochastic_variation=False, max_bet=1000, pivot_horse=True, save=True,
    base_models=base_models, meta_models=meta_models, base_models_features=base_models_features, meta_models_features=meta_models_features, 
  )

  pred = en.predict_target(features)
  pred_bet = en.calc_bet(pred)

  pred.to_csv(os.path.join(local_paths.CANDIDATES_DIR, f'candidates_predicted.csv'), sep='\t')

if __name__ == '__main__':
  predict_candidates()