import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
)

from xgboost import XGBClassifier          # <-- CORRIGE : c'etait XGBRFClassifier (Random Forest,
                                            #     jamais utilise nulle part alors que 'XGBClassifier'
                                            #     etait appele plus bas -> NameError)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna

SEED = 42
ID_COL = 'id'
TARGET_COL = 'tm'
PRIMARY_METRIC = 'f1_macro'


N_SPLITS = 3
HOLDOUT_SIZE = 0.15
PAIRPLOT_SAMPLE = 4000
EDA_PLOT_SAMPLE = 20000
HPO_SAMPLE_SIZE = 60000
N_JOBS = -1
USE_CLASS_WEIGHTS = True  # donne plus de poids a la classe minoritaire (yes/no) pendant l'entrainement
N_TRIALS = 4


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed()

# Couleurs et Palettes
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

# =========================================================
# Chargement des donnees
# =========================================================
DATA_DIR = r'"C:\Users\pc\Downloads\novozymes-enzyme-stability-prediction"'  # <-- adapte si besoin (Kaggle: /kaggle/input/...)
train = pd.read_csv(f'{DATA_DIR}/train.csv')
test = pd.read_csv(f'{DATA_DIR}/test.csv')
sample_submission = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')




print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
print(f'Sample submission shape: {sample_submission.shape}')

assert ID_COL in train.columns and TARGET_COL in train.columns
assert ID_COL in test.columns
assert len(sample_submission) == len(test)
print(f'Target values (train): {train[TARGET_COL].unique()}')
print(f'Target encoding successful: {set(train[TARGET_COL].unique()).issubset({"yes", "no"})}')
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
print(train[eda_cols].describe().T)
print(f'\nTarget distribution:\n{train[TARGET_COL].value_counts()}')

CLASS_ORDER = sorted(train[TARGET_COL].unique()) 
TARGET_COLOR_MAP = dict(zip(CLASS_ORDER, PALETTE_3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
counts = train[TARGET_COL].value_counts().reindex(CLASS_ORDER)
sns.barplot(x=counts.index, y=counts.values, palette=[TARGET_COLOR_MAP[c] for c in CLASS_ORDER], ax=axes[0])
axes[0].set_title(f'Distribution - {TARGET_COL}', fontsize=11, fontweight='bold')
axes[0].set_ylabel('nombre')
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
pct = counts / counts.sum() * 100
axes[1].pie(pct.values, labels=pct.index, autopct='%1.1f%%', startangle=90,
            colors=[TARGET_COLOR_MAP[c] for c in CLASS_ORDER],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[1].set_title(f"Distribution - {TARGET_COL} (%)", fontsize=13, fontweight="bold")
plt.show()

# --- valeurs manquantes ---
missing_train = train.isna().mean().sort_values(ascending=False) * 100
missing_test = test.isna().mean().sort_values(ascending=False) * 100
missing_cols = [c for c in missing_train.index if missing_train[c] > 0]

if missing_cols:
    fig_width = max(8, 0.7 * len(missing_cols))
    fig, ax = plt.subplots(figsize=(fig_width, 6), constrained_layout=True)
    x = np.arange(len(missing_cols))
    width = 0.38
    ax.bar(x - width / 2, missing_train[missing_cols], width, label="Train", color=PALETTE_2[0])
    ax.bar(x + width / 2, missing_test.reindex(missing_cols).values, width, label="Test", color=PALETTE_2[1])
    ax.set_xticks(x)
    ax.set_xticklabels(missing_cols, rotation=40, ha='right')
    ax.set_ylabel("missing %")
    ax.set_title('% de NaN par colonne', fontsize=13, fontweight='bold')
    ax.legend()
    plt.show()

NUMERICAL_COL = ['seq_id' , 'pH' , 'tm' ]
CAT_COL = ['protein_sequence' ]
eda_sample = train.sample(min(len(train), EDA_PLOT_SAMPLE), random_state=SEED)

n_cols = 2
n_rows = int(np.ceil(len(NUMERICAL_COL) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.2 * n_rows), constrained_layout=True)
axes = axes.flatten()
for i, col in enumerate(NUMERICAL_COL):
    sns.kdeplot(data=eda_sample, x=col, hue=TARGET_COL, hue_order=CLASS_ORDER,
                palette=TARGET_COLOR_MAP, fill=True, alpha=0.35, common_norm=False, ax=axes[i])
    axes[i].set_title(f'Influence de {col} sur {TARGET_COL}', fontsize=11)
for j in range(len(NUMERICAL_COL), len(axes)):
    axes[j].axis('off')
plt.show()

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.2 * n_rows), constrained_layout=True)
axes = axes.flatten()
for i, col in enumerate(NUMERICAL_COL):
    sns.boxplot(data=eda_sample, x=TARGET_COL, y=col, order=CLASS_ORDER,
                palette=TARGET_COLOR_MAP, ax=axes[i], showfliers=False)
    axes[i].set_title(f"{col} par {TARGET_COL}", fontsize=11)
for j in range(len(NUMERICAL_COL), len(axes)):
    axes[j].axis("off")
plt.show()


def plot_categorical_vs_target(df, column, target, class_order, color_map):
    for col in column:
        n_categories = df[col].nunique()
        fig_width = max(11, 1.1 * n_categories + 6)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.5), constrained_layout=True)
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, hue=col, palette=PALETTE_7, legend=False, ax=axes[0])
        axes[0].set_title(col, fontsize=11)
        axes[0].tick_params(axis="x", rotation=25)
        prop = pd.crosstab(df[col], df[target], normalize='index').reindex(columns=class_order).loc[order]
        prop.plot(kind="bar", stacked=True, color=[color_map[c] for c in class_order], ax=axes[1], width=0.75)
        axes[1].set_title(f"{col} vs {target} (proportion)", fontsize=11)
        axes[1].set_ylabel("proportion")
        axes[1].tick_params(axis="x", rotation=25)
        axes[1].legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.show()
def plot_nimerical_vs_target(df , colums , target , class_oreder , color_map)
    for col in columns : 
        n_numerical = df[col].nunique()
        
        

plot_categorical_vs_target(train, CAT_COL, TARGET_COL, CLASS_ORDER, TARGET_COLOR_MAP)

pair_sample = train.sample(min(len(train), PAIRPLOT_SAMPLE), random_state=SEED)
g = sns.pairplot(pair_sample, vars=NUMERICAL_COL[:5], hue=TARGET_COL,
                  hue_order=CLASS_ORDER, palette=TARGET_COLOR_MAP, corner=True,
                  height=1.8, plot_kws={"alpha": 0.45, "s": 16})
sns.move_legend(g, "upper right", bbox_to_anchor=(0.82, 0.85), frameon=False)
g.fig.set_size_inches(12, 12)
g.fig.suptitle('Pairplot des variables numériques selon tm', y=1.02)
plt.show()


# =========================================================
# Feature engineering
# =========================================================
def add_features(df: pd.DataFrame, ref_stats: dict = None):
    df = df.copy()
    EPS = 1.0
    RATIO_CLIP = 50

    num_cols_to_fill = [
        'age',  # <-- AJOUTE : oubliee de la liste -> restait NaN -> plantait LogisticRegression
        'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
        'work_study_hours', 'sleep_hours', 'weekend_screen_time',
        'notifications_per_day', 'app_opens_per_day',
    ]

    if ref_stats is None:
        medians = {c: df[c].median() for c in num_cols_to_fill}
    else:
        medians = ref_stats['medians']

    for c in num_cols_to_fill:
        df[c] = df[c].fillna(medians[c])

    def safe_ratio(numerator, denominator):
        r = numerator / (denominator + EPS)
        return r.clip(-RATIO_CLIP, RATIO_CLIP)

    df['entertainment_hours'] = df['social_media_hours'] + df['gaming_hours']
    df['screen_minus_work'] = df['daily_screen_time_hours'] - df['work_study_hours']
    df['screen_to_sleep_ratio'] = safe_ratio(df['daily_screen_time_hours'], df['sleep_hours'])
    df['social_share_screen'] = safe_ratio(df['social_media_hours'], df['daily_screen_time_hours'])
    df['gaming_share_screen'] = safe_ratio(df['gaming_hours'], df['daily_screen_time_hours'])
    df['work_share_screen'] = safe_ratio(df['work_study_hours'], df['daily_screen_time_hours'])
    df['weekend_screen_delta'] = df['weekend_screen_time'] - df['daily_screen_time_hours']
    df['weekend_screen_ratio'] = safe_ratio(df['weekend_screen_time'], df['daily_screen_time_hours'])
    df['notifications_per_app_open'] = safe_ratio(df['notifications_per_day'], df['app_opens_per_day'])
    df['opens_per_screen_hour'] = safe_ratio(df['app_opens_per_day'], df['daily_screen_time_hours'])
    df['notifications_per_screen_hour'] = safe_ratio(df['notifications_per_day'], df['daily_screen_time_hours'])

    if ref_stats is None:
        norm_max = {
            c: df[c].max() if df[c].max() > 0 else 1.0
            for c in ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                      'notifications_per_day', 'app_opens_per_day']
        }
    else:
        norm_max = ref_stats['norm_max']

    df['digital_intensity_score'] = (
        df['daily_screen_time_hours'] / norm_max['daily_screen_time_hours']
        + df['social_media_hours'] / norm_max['social_media_hours']
        + df['gaming_hours'] / norm_max['gaming_hours']
        + df['notifications_per_day'] / norm_max['notifications_per_day']
        + df['app_opens_per_day'] / norm_max['app_opens_per_day']
    )

    df['age_band'] = pd.cut(
        df['age'], bins=[-np.inf, 17, 20, 24, 29, 34, np.inf],
        labels=['<=17', '18-20', '21-24', '25-29', '30-34', '35+']
    ).astype('object')

    stats_used = {'medians': medians, 'norm_max': norm_max}
    return df, stats_used


train_fe, stats = add_features(train)
test_fe, _ = add_features(test, ref_stats=stats)

feature_cols = [c for c in train_fe.columns if c not in [TARGET_COL, ID_COL]]

le_target = LabelEncoder()
y = le_target.fit_transform(train_fe[TARGET_COL])
print(f"Encodage target : {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

X_train_raw = train_fe[feature_cols].reset_index(drop=True)
X_test_raw = test_fe[feature_cols].reset_index(drop=True)
print(f'\nX_train shape: {X_train_raw.shape}')   # <-- CORRIGE : affichait tout le dataframe avant, pas juste .shape
print(f'X_test shape: {X_test_raw.shape}')

n_train = len(X_train_raw)
all_data = pd.concat([X_train_raw, X_test_raw], axis=0)

# <-- CORRIGE (bug majeur) : columns=feature_cols forçait get_dummies a one-hot-encoder
# TOUTES les colonnes, y compris les numeriques (age, daily_screen_time_hours...),
# ce qui aurait explose le nombre de colonnes et detruit l'info numerique.
# On ne cible que les colonnes reellement categorielles (object/category).
cat_cols_for_dummies = all_data.select_dtypes(include=['object', 'category']).columns.tolist()
all_data = pd.get_dummies(all_data, columns=cat_cols_for_dummies)

X = all_data.iloc[:n_train].reset_index(drop=True)
X_test = all_data.iloc[n_train:].reset_index(drop=True)

# =========================================================
# Validation croisée + recherche d'hyperparametres
# =========================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)  # <-- CORRIGE : import manquant

if len(X) > HPO_SAMPLE_SIZE:
    X_hpo, _, y_hpo, _ = train_test_split(X, y, train_size=HPO_SAMPLE_SIZE, random_state=SEED, stratify=y)
else:
    X_hpo, y_hpo = X, y


def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': SEED,
        'n_jobs': N_JOBS,
        'tree_method': 'hist',
    }
    scores = []
    for train_idx, val_idx in skf.split(X_hpo, y_hpo):
        X_tr, X_val = X_hpo.iloc[train_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo[train_idx], y_hpo[val_idx]
        sw = compute_sample_weight('balanced', y_tr) if USE_CLASS_WEIGHTS else None
        model = XGBClassifier(**params, eval_metric='logloss')  # <-- CORRIGE : etait XGBRFClassifier
        model.fit(X_tr, y_tr, sample_weight=sw)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average='macro'))
    return np.mean(scores)


def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': N_JOBS,
    }
    scores = []
    for train_idx, val_idx in skf.split(X_hpo, y_hpo):
        X_tr, X_val = X_hpo.iloc[train_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo[train_idx], y_hpo[val_idx]
        sw = compute_sample_weight('balanced', y_tr) if USE_CLASS_WEIGHTS else None
        model = LGBMClassifier(**params)
        model.fit(X_tr, y_tr, sample_weight=sw)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average='macro'))
    return np.mean(scores)


def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_state': SEED,
        'silent': True,
        'thread_count': N_JOBS,
    }
    scores = []
    for train_idx, val_idx in skf.split(X_hpo, y_hpo):
        X_tr, X_val = X_hpo.iloc[train_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo[train_idx], y_hpo[val_idx]
        sw = compute_sample_weight('balanced', y_tr) if USE_CLASS_WEIGHTS else None
        model = CatBoostClassifier(**params)  # <-- CORRIGE : model n'etait jamais instancie (NameError)
        model.fit(X_tr, y_tr, sample_weight=sw)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds, average='macro'))
    return np.mean(scores)


def objective_logreg(trial):
    # Regularisation L2 (le C d'sklearn = inverse de la force de regularisation)
    params = {
        'C': trial.suggest_float('C', 1e-3, 10.0, log=True),
        'random_state': SEED,
        'max_iter': 2000,
        'class_weight': 'balanced' if USE_CLASS_WEIGHTS else None,
    }
    scores = []
    for train_idx, val_idx in skf.split(X_hpo, y_hpo):
        X_tr, X_val = X_hpo.iloc[train_idx], X_hpo.iloc[val_idx]
        y_tr, y_val = y_hpo[train_idx], y_hpo[val_idx]
        # StandardScaler est indispensable pour LogisticRegression (contrairement aux
        # modeles a base d'arbres) : les features ont des echelles tres differentes
        # (age ~20-40, ratios ~0-50, dummies 0/1) -> sans scaling la regularisation L2
        # penalise injustement les variables a grande echelle.
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # filet de securite si un NaN residuel passe
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(**params)),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_val)
        scores.append(f1_score(y_val, preds, average='macro'))
    return np.mean(scores)


print('Optimisation XGB...')
study_xgb = optuna.create_study(direction='maximize')  # <-- CORRIGE : etait 'minimise' (typo + mauvais sens)
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
print('Best XGB f1_macro:', study_xgb.best_value, '| params:', study_xgb.best_params)

print('Optimisation LGB...')
study_lgb = optuna.create_study(direction='maximize')  # <-- CORRIGE : etait 'minimise'
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS)
print('Best LGB f1_macro:', study_lgb.best_value, '| params:', study_lgb.best_params)

print('Optimisation CatBoost...')
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=N_TRIALS)
print('Best CatBoost f1_macro:', study_cat.best_value, '| params:', study_cat.best_params)

print('Optimisation LogisticRegression...')
study_logreg = optuna.create_study(direction='maximize')
study_logreg.optimize(objective_logreg, n_trials=N_TRIALS)
print('Best LogReg f1_macro:', study_logreg.best_value, '| params:', study_logreg.best_params)

# =========================================================
# Évaluation finale sur un hold-out stratifié
# =========================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=HOLDOUT_SIZE, random_state=SEED, stratify=y
)
sample_weight_train = compute_sample_weight('balanced', y_train) if USE_CLASS_WEIGHTS else None

models = {
    'XGBoost': XGBClassifier(**study_xgb.best_params, random_state=SEED, eval_metric='logloss',
                             n_jobs=N_JOBS, tree_method='hist'),
    'LightGBM': LGBMClassifier(**study_lgb.best_params, random_state=SEED, verbose=-1, n_jobs=N_JOBS),
    'CatBoost': CatBoostClassifier(**study_cat.best_params, random_state=SEED, silent=True, thread_count=N_JOBS),
    'LogisticRegression': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(**study_logreg.best_params, random_state=SEED, max_iter=2000,
                                    class_weight='balanced' if USE_CLASS_WEIGHTS else None)),
    ]),
}


# =========================================================
# Fonction d'évaluation complète (accuracy, F1, precision, recall, ROC-AUC, PR-AUC)
# =========================================================
def evaluate_all_metrics(y_true, y_pred, y_proba=None, positive_label=1):
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, pos_label=positive_label),
        'precision': precision_score(y_true, y_pred, pos_label=positive_label),
        'recall': recall_score(y_true, y_pred, pos_label=positive_label),
    }
    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_proba)
        results['pr_auc'] = average_precision_score(y_true, y_proba)

    print("=== Résumé des métriques ===")
    for name, value in results.items():
        print(f"{name:>10}: {value:.4f}")
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=list(le_target.classes_)))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_true, y_pred))
    return results


# 'yes' est encode a 1 (ordre alphabetique : no=0, yes=1) -> positive_label=1 correspond bien a 'yes'
positive_class_encoded = int(le_target.transform(['yes'])[0])

print(f"\n=== RESULTATS FINAUX (hold-out, target = {TARGET_COL}) ===")
val_probas = {}   # <-- on garde les probas de chaque modele pour le blending plus bas
for name, model in models.items():
    if isinstance(model, Pipeline):
        # LogisticRegression gere deja le desequilibre via class_weight='balanced'
        # dans le pipeline -> pas besoin (et pas trivial) de repasser sample_weight ici
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight_train)
    y_preds = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, positive_class_encoded]
    val_probas[name] = y_proba

    print(f"\n--- {name} ---")
    scores = evaluate_all_metrics(y_val, y_preds, y_proba, positive_label=positive_class_encoded)

    cm = confusion_matrix(y_val, y_preds)
    fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='mako', cbar=False,
                xticklabels=le_target.classes_, yticklabels=le_target.classes_, ax=ax)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title(f'Matrice de confusion — {name}', fontsize=11)
    plt.show()


# =========================================================
# Blending : recherche des poids qui maximisent le ROC-AUC
# (generalise l'idee de "w * lgb + (1-w) * xgb" a N modeles)
# =========================================================
def find_best_blend_weights(y_true, proba_dict, n_random=3000, seed=SEED):
    """
    proba_dict : {'XGBoost': array_probas, 'LightGBM': array_probas, ...}
    Recherche aleatoire (tirages Dirichlet -> poids qui somment a 1, tous >= 0)
    suivie d'un raffinement local, pour trouver la combinaison qui maximise le
    ROC-AUC du blend. Plus robuste et plus simple qu'une grille exhaustive
    des lors qu'on a plus de 2 modeles (une grille sur 4 dims devient enorme).
    """
    names = list(proba_dict.keys())
    probas = np.column_stack([proba_dict[n] for n in names])  # (n_samples, n_models)
    rng = np.random.default_rng(seed)

    best_w, best_score = None, -1.0
    for _ in range(n_random):
        w = rng.dirichlet(np.ones(len(names)))
        score = roc_auc_score(y_true, probas @ w)
        if score > best_score:
            best_score, best_w = score, w

    # raffinement local : on ajuste chaque poids par petits pas tant que ca ameliore le score
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
                score = roc_auc_score(y_true, probas @ w_try)
                if score > best_score:
                    best_score, best_w = score, w_try
                    improved = True
        if not improved:
            step /= 2

    return dict(zip(names, best_w)), best_score


print("\n=== Recherche des poids de blending (maximisation ROC-AUC) ===")
for name, proba in val_probas.items():
    print(f"{name:>20}: ROC-AUC seul = {roc_auc_score(y_val, proba):.4f}")

blend_weights, blend_roc_auc = find_best_blend_weights(y_val, val_probas)
print("\nMeilleurs poids trouves:")
for name, w in blend_weights.items():
    print(f"  {name:>20}: {w:.3f}")
print(f"ROC-AUC du blend: {blend_roc_auc:.4f}")

blended_val_proba = sum(blend_weights[n] * val_probas[n] for n in val_probas)


# =========================================================
# Ajustement de seuil (F-beta) + evolution des metriques selon le seuil
# =========================================================
from sklearn.metrics import precision_recall_curve

FBETA = 1.0  # F-beta a optimiser


def tune_threshold_fbeta(y_true, proba, beta=FBETA):
    """Version binaire (1 seul seuil, pas un par classe comme dans le projet
    multi-classe precedent) : cherche le seuil qui maximise le F-beta."""
    precision, recall, thresh = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    denom = (beta ** 2 * precision) + recall
    fbeta = np.divide((1 + beta ** 2) * precision * recall, denom,
                       out=np.zeros_like(denom), where=denom > 0)
    best_idx = np.argmax(fbeta)
    return thresh[best_idx], fbeta[best_idx]


def plot_threshold_metrics(y_true, proba, beta=FBETA, n_points=99):

    thresholds = np.linspace(0.01, 0.99, n_points)
    precisions, recalls, f1s, accs = [], [], [], []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))
        accs.append(accuracy_score(y_true, preds))

    best_t, best_fbeta = tune_threshold_fbeta(y_true, proba, beta)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(thresholds, precisions, label='Precision')
    ax.plot(thresholds, recalls, label='Recall')
    ax.plot(thresholds, f1s, label='F1')
    ax.plot(thresholds, accs, label='Accuracy', linestyle='--', alpha=0.6)
    ax.axvline(best_t, color='red', linestyle=':', label=f'Seuil optimal (F{beta}={best_fbeta:.3f})')
    ax.set_xlabel('Seuil de décision (sur P(yes))')
    ax.set_ylabel('Score')
    ax.set_title(f'Évolution des métriques selon le seuil — blend (F{beta})')
    # Beta = 1.0 -> F1 = Precision = Recall
    # Beta = 0.0 -> F1 = 2 * Precision * Recall / (Precision + Recall)
    # Beta = 2.0 -> F1 = (1 + Precision^2) * Recall 
    ax.legend()
    plt.show()

    return best_t


best_threshold = plot_threshold_metrics(y_val, blended_val_proba, beta=FBETA)
print(f"\nSeuil optimal trouve (F{FBETA}): {best_threshold:.4f}")

final_blend_preds = (blended_val_proba >= best_threshold).astype(int)
print("\n=== RESULTATS FINAUX DU BLEND (seuil ajuste) ===")
evaluate_all_metrics(y_val, final_blend_preds, blended_val_proba, positive_label=positive_class_encoded)

