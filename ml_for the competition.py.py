import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

for pkg, import_name in [('lightgbm', 'lightgbm'), ('xgboost', 'xgboost'),
                          ('catboost', 'catboost'), ('optuna', 'optuna'),
                          ('biopython', 'Bio')]:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg])

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import optuna
from tqdm.auto import tqdm

SEED = 42
ID_COL = 'seq_id'
TARGET_COL = 'tm'
PRIMARY_METRIC = 'spearman'

N_SPLITS = 5
HOLDOUT_SIZE = 0.15
PAIRPLOT_SAMPLE = 2000
EDA_PLOT_SAMPLE = 5000
HPO_SAMPLE_SIZE = 15000
N_JOBS = -1
N_TRIALS = 30


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()

yellow, cyan_g, cyan_dark = "#F7C53E", "#0CF7AF", "#11AB7C"
purple, purple_dark, purple_light = "#D826F8", "#9309AB", "#b683d6"
blue, red, orange, green = "#0C97FA", "#FA1D19", "#FA9F19", "#0CFA58"
light_blue, soft_blue, dark_blue = "#01FADC", "#81c9e6", "#394be6"

PALETTE_2 = [cyan_g, purple]
PALETTE_3 = [yellow, cyan_g, purple]
PALETTE_7 = [purple_dark, purple_light, purple, blue, light_blue, dark_blue, soft_blue]

sns.set_style("whitegrid")
sns.set_palette(PALETTE_7)
plt.rcParams["figure.facecolor"] = "#f8fafc"
pd.set_option("display.float_format", "{:.4f}".format)

DATA_DIR = './data/novozymes-enzyme-stability-prediction'

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
sample_submission = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
test_labels = pd.read_csv(f'{DATA_DIR}/test_labels.csv')

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print(f'Sample submission shape: {sample_submission.shape}')

assert ID_COL in train.columns and TARGET_COL in train.columns
assert ID_COL in test.columns
assert len(sample_submission) == len(test)
print(f'\nStatistiques de {TARGET_COL} (train): min={train[TARGET_COL].min():.2f}, '
      f'max={train[TARGET_COL].max():.2f}, mean={train[TARGET_COL].mean():.2f}')
print("\n--- Train Head ---")
print(train.head())


def build_eda_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = df[column]
        not_null = series.dropna()
        rows.append({
            'features': column,
            'dtype': str(series.dtype),
            'missing_count': int(series.isna().sum()),
            'missing_pct': float(series.isna().mean() * 100.0),
            'nunique': int(series.nunique()),
            'sample_values': ','.join(not_null.astype(str).unique()[:4]),
        })
    return pd.DataFrame(rows)


eda_cols = [c for c in train.columns if c != TARGET_COL]
eda_summary = build_eda_summary(train[eda_cols])
print(eda_summary)
print(train[[TARGET_COL, 'pH']].describe().T)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
sns.histplot(train[TARGET_COL], bins=50, color=cyan_g, kde=True, ax=axes[0])
axes[0].set_title(f'Distribution - {TARGET_COL}', fontsize=11, fontweight='bold')
sns.boxplot(x=train[TARGET_COL], color=yellow, ax=axes[1])
axes[1].set_title(f'Boxplot - {TARGET_COL}', fontsize=11, fontweight='bold')
plt.show()

missing_train = train.isna().mean().sort_values(ascending=False) * 100
missing_test = test.isna().mean().sort_values(ascending=False) * 100
missing_cols = [c for c in missing_train.index if missing_train[c] > 0]

if missing_cols:
    fig_width = max(8, 0.7 * len(missing_cols))
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)
    x = np.arange(len(missing_cols))
    width = 0.38
    ax.bar(x - width / 2, missing_train[missing_cols], width, label="Train", color=PALETTE_2[0])
    ax.bar(x + width / 2, missing_test.reindex(missing_cols).fillna(0).values, width,
           label="Test", color=PALETTE_2[1])
    ax.set_xticks(x)
    ax.set_xticklabels(missing_cols, rotation=40, ha='right')
    ax.set_ylabel("missing %")
    ax.set_title('% de NaN par colonne', fontsize=13, fontweight='bold')
    ax.legend()
    plt.show()

train_eda = train.copy()
train_eda['seq_length_tmp'] = train_eda['protein_sequence'].str.len()

NUMERICAL_COL = ['pH', 'seq_length_tmp']
eda_sample = train_eda.sample(min(len(train_eda), EDA_PLOT_SAMPLE), random_state=SEED)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for i, col in enumerate(NUMERICAL_COL):
    axes[i].scatter(eda_sample[col], eda_sample[TARGET_COL], alpha=0.25, s=10, color=light_blue)
    axes[i].set_title(f'{col} vs {TARGET_COL}', fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel(TARGET_COL)
plt.show()


def plot_categorical_vs_target(df, column, target, palette):
    for col in column:
        top_cats = df[col].value_counts().head(7).index.tolist()
        df_top = df[df[col].isin(top_cats)]
        if len(df_top) == 0:
            continue
        n_categories = len(top_cats)
        fig_width = max(11, 1.1 * n_categories + 6)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.5), constrained_layout=True)

        order = df_top[col].value_counts().index
        sns.countplot(data=df_top, x=col, order=order, hue=col, palette=palette,
                      legend=False, ax=axes[0])
        axes[0].set_title(f'{col} (top {n_categories})', fontsize=11)
        axes[0].tick_params(axis="x", rotation=25, labelsize=8)

        sns.boxplot(data=df_top, x=col, y=target, order=order, hue=col,
                    palette=palette, legend=False, ax=axes[1])
        axes[1].set_title(f"{target} par {col}", fontsize=11)
        axes[1].tick_params(axis="x", rotation=25, labelsize=8)
        plt.show()


CAT_COL = ['data_source']
plot_categorical_vs_target(train, CAT_COL, TARGET_COL, PALETTE_7)

pair_sample = train_eda.sample(min(len(train_eda), PAIRPLOT_SAMPLE), random_state=SEED)
g = sns.pairplot(
    pair_sample[NUMERICAL_COL + [TARGET_COL]].dropna(),
    corner=True, height=2.0, plot_kws={"alpha": 0.4, "s": 14, "color": purple_light},
)
g.fig.suptitle(f'Pairplot des variables numeriques selon {TARGET_COL}', y=1.02)
plt.show()


STANDARD_AA = 'ACDEFGHIKLMNPQRSTVWY'
_STANDARD_AA_SET = set(STANDARD_AA)


def extract_protein_features(seq: str) -> dict:
    seq = str(seq)
    seq_clean = ''.join(c for c in seq if c in _STANDARD_AA_SET)

    features = {
        'seq_length': len(seq),
        'seq_length_clean': len(seq_clean),
        'n_nonstandard_aa': len(seq) - len(seq_clean),
    }

    default_keys = ['molecular_weight', 'aromaticity', 'instability_index',
                     'isoelectric_point', 'gravy', 'helix_fraction',
                     'turn_fraction', 'sheet_fraction']

    if len(seq_clean) == 0:
        features.update({k: np.nan for k in default_keys})
        for aa in STANDARD_AA:
            features[f'aa_pct_{aa}'] = np.nan
        features['n_charged_positive'] = np.nan
        features['n_charged_negative'] = np.nan
        features['charge_net'] = np.nan
        return features

    try:
        pa = ProteinAnalysis(seq_clean)
        features['molecular_weight'] = pa.molecular_weight()
        features['aromaticity'] = pa.aromaticity()
        features['instability_index'] = pa.instability_index()
        features['isoelectric_point'] = pa.isoelectric_point()
        features['gravy'] = pa.gravy()
        helix, turn, sheet = pa.secondary_structure_fraction()
        features['helix_fraction'] = helix
        features['turn_fraction'] = turn
        features['sheet_fraction'] = sheet
        aa_percent = pa.get_amino_acids_percent()
    except Exception:
        features.update({k: np.nan for k in default_keys})
        aa_percent = {}

    for aa in STANDARD_AA:
        features[f'aa_pct_{aa}'] = aa_percent.get(aa, 0.0)

    features['n_charged_positive'] = sum(seq_clean.count(a) for a in 'KRH')
    features['n_charged_negative'] = sum(seq_clean.count(a) for a in 'DE')
    features['charge_net'] = features['n_charged_positive'] - features['n_charged_negative']

    return features


def add_features(df: pd.DataFrame, ref_stats: dict = None):
    records = [extract_protein_features(s)
               for s in tqdm(df['protein_sequence'], desc='Feature engineering')]
    feat_df = pd.DataFrame(records)
    feat_df[ID_COL] = df[ID_COL].values
    feat_df['pH'] = df['pH'].values

    num_cols_to_fill = [c for c in feat_df.columns if c != ID_COL]
    if ref_stats is None:
        medians = {c: feat_df[c].median() for c in num_cols_to_fill}
    else:
        medians = ref_stats['medians']

    for c in num_cols_to_fill:
        feat_df[c] = feat_df[c].fillna(medians[c])

    if TARGET_COL in df.columns:
        feat_df[TARGET_COL] = df[TARGET_COL].values

    stats_used = {'medians': medians}
    return feat_df, stats_used


train_fe, stats = add_features(train)
test_fe, _ = add_features(test, ref_stats=stats)

feature_cols = [c for c in train_fe.columns if c not in [TARGET_COL, ID_COL]]

y = train_fe[TARGET_COL].values
X = train_fe[feature_cols].reset_index(drop=True)
X_test = test_fe[feature_cols].reset_index(drop=True)

print(f'\nX shape: {X.shape}')
print(f'X_test shape: {X_test.shape}')
assert list(X.columns) == list(X_test.columns)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

if len(X) > HPO_SAMPLE_SIZE:
    X_hpo, _, y_hpo, _ = train_test_split(X, y, train_size=HPO_SAMPLE_SIZE, random_state=SEED)
else:
    X_hpo, y_hpo = X, y


def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
        'verbosity': 0,
    }
    scores = []
    for train_idx, val_idx in kf.split(X_hpo):
        X_tr, X_val = X_hpo.iloc[train_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo[train_idx], y_hpo[val_idx]
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        corr = spearmanr(preds, y_val).correlation
        scores.append(corr if not np.isnan(corr) else -1.0)
    return float(np.mean(scores))


def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': N_JOBS,
    }
    scores = []
    for train_idx, val_idx in kf.split(X_hpo):
        X_tr, X_val = X_hpo.iloc[train_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo[train_idx], y_hpo[val_idx]
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        corr = spearmanr(preds, y_val).correlation
        scores.append(corr if not np.isnan(corr) else -1.0)
    return float(np.mean(scores))


def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'random_state': SEED,
        'silent': True,
        'thread_count': N_JOBS,
    }
    scores = []
    for train_idx, val_idx in kf.split(X_hpo):
        X_tr, X_val = X_hpo.iloc[train_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo[train_idx], y_hpo[val_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        corr = spearmanr(preds, y_val).correlation
        scores.append(corr if not np.isnan(corr) else -1.0)
    return float(np.mean(scores))


print('Optimisation XGB...')
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
print('Best XGB Spearman:', study_xgb.best_value, '| params:', study_xgb.best_params)

print('Optimisation LGB...')
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS)
print('Best LGB Spearman:', study_lgb.best_value, '| params:', study_lgb.best_params)

print('Optimisation CatBoost...')
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=N_TRIALS)
print('Best CatBoost Spearman:', study_cat.best_value, '| params:', study_cat.best_params)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=HOLDOUT_SIZE, random_state=SEED)

models = {
    'XGBoost': XGBRegressor(**study_xgb.best_params, random_state=SEED, n_jobs=N_JOBS,
                             tree_method='hist', verbosity=0),
    'LightGBM': LGBMRegressor(**study_lgb.best_params, random_state=SEED, verbose=-1, n_jobs=N_JOBS),
    'CatBoost': CatBoostRegressor(**study_cat.best_params, random_state=SEED, silent=True,
                                   thread_count=N_JOBS),
}


def evaluate_all_metrics(y_true, y_pred, name=''):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    corr = float(spearmanr(y_pred, y_true).correlation)

    results = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'spearman': corr}
    print(f"\n=== Résumé des métriques — {name} ===")
    for k, v in results.items():
        print(f"{k:>10}: {v:.4f}")
    return results


print(f"\n=== RESULTATS FINAUX (hold-out, target = {TARGET_COL}) ===")
val_preds = {}
metrics_rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    val_preds[name] = y_pred

    scores = evaluate_all_metrics(y_val, y_pred, name)
    scores['model'] = name
    metrics_rows.append(scores)

metrics_df = pd.DataFrame(metrics_rows)[['model', 'mse', 'rmse', 'mae', 'r2', 'spearman']]
metrics_df = metrics_df.sort_values('mse').reset_index(drop=True)
print("\n=== Comparaison des 3 modeles (tries par MSE croissante) ===")
print(metrics_df)

fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
sns.barplot(data=metrics_df, x='model', y='mse', hue='model', palette=PALETTE_3, legend=False, ax=ax)
for i, v in enumerate(metrics_df['mse']):
    ax.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=10)
ax.set_title('MSE par modele (hold-out)', fontsize=12, fontweight='bold')
ax.set_ylabel('MSE')
plt.show()


def find_best_blend_weights(y_true, pred_dict, n_random=3000, seed=SEED):
    names = list(pred_dict.keys())
    preds = np.column_stack([pred_dict[n] for n in names])
    rng = np.random.default_rng(seed)

    best_w, best_score = None, -1.0
    for _ in range(n_random):
        w = rng.dirichlet(np.ones(len(names)))
        score = spearmanr(preds @ w, y_true).correlation
        if score > best_score:
            best_score, best_w = score, w

    step = 0.05
    while step > 1e-3:
        improved = False
        for i in range(len(names)):
            for delta in (step, -step):
                w_try = best_w.copy()
                w_try[i] = max(0.0, w_try[i] + delta)
                if w_try.sum() == 0:
                    continue
                w_try = w_try / w_try.sum()
                score = spearmanr(preds @ w_try, y_true).correlation
                if score > best_score:
                    best_score, best_w = score, w_try
                    improved = True
        if not improved:
            step /= 2

    return dict(zip(names, best_w)), best_score


print("\n=== Recherche des poids de blending (maximisation Spearman) ===")
for name, pred in val_preds.items():
    corr = spearmanr(pred, y_val).correlation
    mse_single = mean_squared_error(y_val, pred)
    print(f"{name:>20}: Spearman seul = {corr:.4f} | MSE seul = {mse_single:.4f}")

blend_weights, blend_spearman = find_best_blend_weights(y_val, val_preds)
print("\nMeilleurs poids trouves:")
for name, w in blend_weights.items():
    print(f"  {name:>20}: {w:.3f}")

blended_val_pred = sum(blend_weights[n] * val_preds[n] for n in val_preds)
blend_mse = mean_squared_error(y_val, blended_val_pred)
print(f"\nSpearman du blend: {blend_spearman:.4f}")
print(f"MSE du blend     : {blend_mse:.4f}")

print("\n=== RESULTATS FINAUX DU BLEND ===")
evaluate_all_metrics(y_val, blended_val_pred, name='Blend')

residuals = y_val - blended_val_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
axes[0].scatter(y_val, blended_val_pred, alpha=0.3, s=10, color=red)
lims = [min(y_val.min(), blended_val_pred.min()), max(y_val.max(), blended_val_pred.max())]
axes[0].plot(lims, lims, 'k--', lw=1)
axes[0].set_xlabel('tm reel')
axes[0].set_ylabel('tm predit (blend)')
axes[0].set_title('Predictions vs realite (holdout)')
sns.histplot(residuals, bins=50, color=red, kde=True, ax=axes[1])
axes[1].axvline(0, color='black', linestyle='--', lw=1)
axes[1].set_title('Distribution des residus')
plt.show()

final_models = {}
test_preds = {}
for name, model in models.items():
    final_model = model.__class__(**model.get_params())
    final_model.fit(X, y)
    final_models[name] = final_model
    test_preds[name] = final_model.predict(X_test)

test_pred_blend = sum(blend_weights[n] * test_preds[n] for n in blend_weights)

y_test_true = test_labels.set_index(ID_COL).loc[test[ID_COL], TARGET_COL].values
print("\n=== EVALUATION SUR LE VRAI TEST SET (test_labels.csv) ===")
for name, pred in test_preds.items():
    evaluate_all_metrics(y_test_true, pred, name=f'{name} (test reel)')
evaluate_all_metrics(y_test_true, test_pred_blend, name='Blend (test reel)')

submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: test_pred_blend})
assert submission.shape == sample_submission.shape
assert list(submission.columns) == list(sample_submission.columns)
submission.to_csv('submission.csv', index=False)
print("\nSubmission sauvegardee -> submission.csv")
print(submission.head())
