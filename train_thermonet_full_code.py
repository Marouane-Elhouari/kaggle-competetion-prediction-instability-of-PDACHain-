import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from tqdm.notebook import tqdm
pio.renderers.default = 'iframe'
print('Renderer plotly actif:', pio.renderers.default)
import plotly.express as _px_test


RUNNING_ON_KAGGLE = os.path.exists('/kaggle/input')
print('Environnement detecte:', 'Kaggle' if RUNNING_ON_KAGGLE else 'Local (PC)')
if RUNNING_ON_KAGGLE:
    COMPETITION_DIR = '../input/competitions/novozymes-enzyme-stability-prediction'
    VOXEL_DATASET_DIR = '../input/notebooks/vslaykovsky/14656-unique-mutations-voxel-features-pdbs'
else:
    COMPETITION_DIR = './data/novozymes-enzyme-stability-prediction'
    VOXEL_DATASET_DIR = './data/14656-unique-mutations-voxel-features-pdbs'
WILDTYPE_PDB = os.path.join(COMPETITION_DIR, 'wildtype_structure_prediction_af2.pdb')
TEST_CSV = os.path.join(COMPETITION_DIR, 'test.csv')
CSV_PATH = os.path.join(VOXEL_DATASET_DIR, 'dataset.csv')
FEATURES_DIR = os.path.join(VOXEL_DATASET_DIR, 'features')
if RUNNING_ON_KAGGLE:
    for root, dirs, files in os.walk('../input'):
        depth = root.count(os.sep) - '../input'.count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
df = pd.read_csv(CSV_PATH)
df['features_path'] = df.apply(lambda r: os.path.join(FEATURES_DIR, f'{r.PDB_chain}_{r.wildtype}{r.pdb_position}{r.mutant}.npy'), axis=1)
df['features_exists'] = df['features_path'].apply(os.path.exists)
df = df[df['features_exists']].reset_index(drop=True)
df.head()


print(f'1. Chargement du fichier CSV: {CSV_PATH}')
df = pd.read_csv(CSV_PATH)
print(f'   Total mutations dans dataset.csv: {len(df)}')
df['features_path'] = df.apply(lambda r: os.path.join(FEATURES_DIR, f'{r.PDB_chain}_{r.wildtype}{r.pdb_position}{r.mutant}.npy'), axis=1)
df['features_exists'] = df['features_path'].apply(os.path.exists)
n_before = len(df)
df = df[df['features_exists']].reset_index(drop=True)
print(f'   Mutations avec features .npy disponibles: {len(df)} / {n_before}')
df.head()


fig_ddg = px.histogram(df, x='ddG', nbins=60, title="Distribution de ΔΔG (ddG) - changement d'energie de stabilite")
fig_ddg.update_layout(bargap=0.02)
fig_ddg.show()


fig_dt = px.histogram(df, x='dT', nbins=60, title='Distribution de ΔT (dT) - changement de temperature de fusion (Tm)')
fig_dt.update_layout(bargap=0.02)
fig_dt.show()


fig_scatter = px.scatter(df, x='ddG', y='dT', title='ddG vs dT', opacity=0.4, trendline='ols')
fig_scatter.show()


import os as _os
from IPython.display import display as _display, FileLink as _FileLink
if 'show_and_export' not in globals():
    VOXEL_VIZ_OUTPUT_DIR = globals().get('VOXEL_VIZ_OUTPUT_DIR', './voxel_viz_exports')
    _os.makedirs(VOXEL_VIZ_OUTPUT_DIR, exist_ok=True)

    def show_and_export(fig, filename: str, output_dir: str=VOXEL_VIZ_OUTPUT_DIR):
        fig.show(config={'displayModeBar': True, 'scrollZoom': True})
        filepath = _os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn', full_html=True)
        print(f'Figure sauvegardee -> {filepath}')
        _display(_FileLink(filepath))
        return filepath
FEATURE_NAMES = ['hydrophobic', 'aromatic', 'hbond_acceptor', 'hbond_donor', 'positive_ionizable', 'negative_ionizable', 'occupancies']
VOXEL_SIZE = 16
GRID_COORDS = np.array([(x, y, z) for x in range(VOXEL_SIZE) for y in range(VOXEL_SIZE) for z in range(VOXEL_SIZE)])

def plot_feature_diff(sample_idx: int, feature: str='occupancies', threshold: float=0.5, marker_size: int=6):
    row = df.iloc[sample_idx]
    features = np.load(row.features_path)
    ch = FEATURE_NAMES.index(feature)
    wildtype_vals = features[ch].flatten()
    mutant_vals = features[7 + ch].flatten()
    mask_wt = wildtype_vals > threshold
    mask_mut = mutant_vals > threshold
    both_mask = mask_wt & mask_mut
    mutant_only_mask = mask_mut & ~mask_wt
    wildtype_only_mask = mask_wt & ~mask_mut
    categories = [(both_mask, 'blue', 'blue'), (mutant_only_mask, 'red', 'red'), (wildtype_only_mask, 'green', 'green')]
    fig = go.Figure()
    for mask, color, label in categories:
        fig.add_trace(go.Scatter3d(x=GRID_COORDS[mask, 0], y=GRID_COORDS[mask, 1], z=GRID_COORDS[mask, 2], mode='markers', marker=dict(size=marker_size, color=color, opacity=0.85), name=label, legendgroup='color'))
    ddg_txt = f'{row.ddG:.2f}' if pd.notna(row.ddG) else 'NA'
    fig.update_layout(title=f'Train idx:{sample_idx}; ddg={ddg_txt}', height=750, legend_title_text='color', scene=dict(xaxis_title='x (voxel)', yaxis_title='y (voxel)', zaxis_title='z (voxel)'))
    html_filename = f'voxel_viz_{sample_idx}_{feature}_thr{threshold}.html'
    show_and_export(fig, html_filename)
for i in range(min(3, len(df))):
    plot_feature_diff(i, feature='occupancies', threshold=0.5)
    plot_feature_diff(i, feature='occupancies', threshold=0.3)


import os as _os
from IPython.display import display as _display, FileLink as _FileLink
if 'show_and_export' not in globals():
    VOXEL_VIZ_OUTPUT_DIR = globals().get('VOXEL_VIZ_OUTPUT_DIR', './voxel_viz_exports')
    _os.makedirs(VOXEL_VIZ_OUTPUT_DIR, exist_ok=True)

    def show_and_export(fig, filename: str, output_dir: str=VOXEL_VIZ_OUTPUT_DIR):
        fig.show(config={'displayModeBar': True, 'scrollZoom': True})
        filepath = _os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn', full_html=True)
        print(f'Figure sauvegardee -> {filepath}')
        _display(_FileLink(filepath))
        return filepath

def plot_multi_feature_diff(sample_idx: int, features=('occupancies', 'hydrophobic', 'hbond_donor'), threshold: float=0.5, marker_size: int=5):
    row = df.iloc[sample_idx]
    arr = np.load(row.features_path)
    n = len(features)
    fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'scatter3d'}] * n], subplot_titles=list(features))
    colors = {'both': 'blue', 'mutant_only': 'red', 'wildtype_only': 'green'}
    for col, feat_name in enumerate(features, start=1):
        ch = FEATURE_NAMES.index(feat_name)
        wt_vals = arr[ch].flatten()
        mut_vals = arr[7 + ch].flatten()
        mask_wt = wt_vals > threshold
        mask_mut = mut_vals > threshold
        both_mask = mask_wt & mask_mut
        mutant_only_mask = mask_mut & ~mask_wt
        wildtype_only_mask = mask_wt & ~mask_mut
        for mask, key in [(both_mask, 'both'), (mutant_only_mask, 'mutant_only'), (wildtype_only_mask, 'wildtype_only')]:
            fig.add_trace(go.Scatter3d(x=GRID_COORDS[mask, 0], y=GRID_COORDS[mask, 1], z=GRID_COORDS[mask, 2], mode='markers', marker=dict(size=marker_size, color=colors[key], opacity=0.85), name=key, legendgroup=key, showlegend=col == 1), row=1, col=col)
    ddg_txt = f'{row.ddG:.2f}' if pd.notna(row.ddG) else 'NA'
    fig.update_layout(title=f'Train idx:{sample_idx}; ddg={ddg_txt} | threshold={threshold}', height=600, width=380 * n, legend_title_text='color')
    feat_tag = '-'.join(features)
    html_filename = f'voxel_viz_multi_{sample_idx}_{feat_tag}_thr{threshold}.html'
    show_and_export(fig, html_filename)
plot_multi_feature_diff(0, features=('occupancies', 'hydrophobic', 'hbond_donor'), threshold=0.5)


def count_diff_voxels(sample_idx: int, feature: str='occupancies', threshold: float=0.5):
    row = df.iloc[sample_idx]
    arr = np.load(row.features_path)
    ch = FEATURE_NAMES.index(feature)
    wt_vals = arr[ch].flatten()
    mut_vals = arr[7 + ch].flatten()
    mask_wt = wt_vals > threshold
    mask_mut = mut_vals > threshold
    both = int((mask_wt & mask_mut).sum())
    mutant_only = int((mask_mut & ~mask_wt).sum())
    wildtype_only = int((mask_wt & ~mask_mut).sum())
    total_voxels = GRID_COORDS.shape[0]
    return {'sample_idx': sample_idx, 'feature': feature, 'ddG': row.ddG, 'both_blue': both, 'mutant_only_red': mutant_only, 'wildtype_only_green': wildtype_only, 'total_colored': both + mutant_only + wildtype_only, 'total_voxels': total_voxels}
stats_rows = [count_diff_voxels(i, feature='occupancies', threshold=0.5) for i in range(min(5, len(df)))]
stats_df = pd.DataFrame(stats_rows)
print(stats_df)
print()
print(f'Nombre total de mutations dans df (avec fichier .npy existant): {len(df)}')
print(f'Chaque sample a un espace de {GRID_COORDS.shape[0]} voxels possibles (16x16x16), par feature.')


import os as _os
from IPython.display import display as _display, FileLink as _FileLink
if 'show_and_export' not in globals():
    VOXEL_VIZ_OUTPUT_DIR = globals().get('VOXEL_VIZ_OUTPUT_DIR', './voxel_viz_exports')
    _os.makedirs(VOXEL_VIZ_OUTPUT_DIR, exist_ok=True)

    def show_and_export(fig, filename: str, output_dir: str=VOXEL_VIZ_OUTPUT_DIR):
        fig.show(config={'displayModeBar': True, 'scrollZoom': True})
        filepath = _os.path.join(output_dir, filename)
        fig.write_html(filepath, include_plotlyjs='cdn', full_html=True)
        print(f'Figure sauvegardee -> {filepath}')
        _display(_FileLink(filepath))
        return filepath

def find_closest_idx(target_ddg: float):
    diffs = (df['ddG'] - target_ddg).abs()
    return diffs.idxmin()

def plot_ddg_comparison(ddg_values, feature: str='occupancies', threshold: float=0.5, marker_size: int=5):
    sample_indices = [find_closest_idx(v) for v in ddg_values]
    n = len(sample_indices)
    subplot_titles = []
    for target, idx in zip(ddg_values, sample_indices):
        actual_ddg = df.loc[idx, 'ddG']
        subplot_titles.append(f'idx={idx} | ddg={target} | reel={actual_ddg:.4f}')
    fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'scatter3d'}] * n], subplot_titles=subplot_titles)
    colors = {'both': 'blue', 'mutant_only': 'red', 'wildtype_only': 'green'}
    ch = FEATURE_NAMES.index(feature)
    for col, idx in enumerate(sample_indices, start=1):
        row = df.loc[idx]
        arr = np.load(row.features_path)
        wt_vals = arr[ch].flatten()
        mut_vals = arr[7 + ch].flatten()
        mask_wt = wt_vals > threshold
        mask_mut = mut_vals > threshold
        both_mask = mask_wt & mask_mut
        mutant_only_mask = mask_mut & ~mask_wt
        wildtype_only_mask = mask_wt & ~mask_mut
        for mask, key in [(both_mask, 'both'), (mutant_only_mask, 'mutant_only'), (wildtype_only_mask, 'wildtype_only')]:
            fig.add_trace(go.Scatter3d(x=GRID_COORDS[mask, 0], y=GRID_COORDS[mask, 1], z=GRID_COORDS[mask, 2], mode='markers', marker=dict(size=marker_size, color=colors[key], opacity=0.85), name=key, legendgroup=key, showlegend=col == 1), row=1, col=col)
    fig.update_layout(title=f'Comparaison ddG (feature={feature}, threshold={threshold})', height=600, width=420 * n, legend_title_text='color')
    idx_tag = '-'.join((str(i) for i in sample_indices))
    html_filename = f'voxel_viz_ddg_comparison_{idx_tag}_{feature}_thr{threshold}.html'
    show_and_export(fig, html_filename)
    summary = []
    for target, idx in zip(ddg_values, sample_indices):
        summary.append(count_diff_voxels(idx, feature=feature, threshold=threshold))
    return pd.DataFrame(summary)
ddg_targets = [0.705833, -0.12, -0.05]
summary_df = plot_ddg_comparison(ddg_targets, feature='occupancies', threshold=0.5)
print(summary_df)


import numpy as np
N_SAMPLES = 500
sample_idx_list = np.random.RandomState(42).choice(len(df), size=min(N_SAMPLES, len(df)), replace=False)
rows = []
for idx in sample_idx_list:
    stats = count_diff_voxels(idx, feature='occupancies', threshold=0.5)
    total = stats['total_colored']
    blue_fraction = stats['both_blue'] / total if total > 0 else np.nan
    rows.append({'idx': idx, 'ddG': stats['ddG'], 'blue_fraction': blue_fraction, 'total_colored': total})
corr_df = pd.DataFrame(rows).dropna()
correlation = corr_df['ddG'].corr(corr_df['blue_fraction'])
print(f'Correlation (Pearson) entre ddG et fraction de bleu: {correlation:.3f}')
print(f'(proche de 0 = pas de lien direct ; proche de +-1 = lien fort)')
print()
print(corr_df.describe())
fig_corr = px.scatter(corr_df, x='ddG', y='blue_fraction', title=f'ddG vs fraction de voxels inchanges (bleu) | correlation={correlation:.3f}', opacity=0.5, trendline='ols')
fig_corr.show(config={'displayModeBar': True})


def robust_stats(series):
    median = series.median()
    q1, q3 = (series.quantile(0.25), series.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        iqr = series.std() if series.std() > 0 else 1.0
    return (median, iqr)
dt_valid = df['dT'].dropna()
print('=== Comparaison des methodes de normalisation pour dT ===')
print(f'mean={dt_valid.mean():.3f}, std={dt_valid.std():.3f}  (sensibles aux outliers)')
median_dt, iqr_dt = robust_stats(dt_valid)
print(f'median={median_dt:.3f}, IQR={iqr_dt:.3f}  (robustes)')
print(f'Nombre de valeurs |dT| > 20 (outliers potentiels): {(dt_valid.abs() > 20).sum()} / {len(dt_valid)}')


from sklearn.model_selection import GroupKFold
import numpy as np
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)
df['fold'] = -1
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['PDB_chain'])):
    df.loc[val_idx, 'fold'] = fold_idx
print(df['fold'].value_counts().sort_index())


import torch
import random
from torch.utils.data import Dataset

class VoxelDataset(Dataset):

    def __init__(self, dataframe, ddg_mean=0.0, ddg_std=1.0, dt_mean=0.0, dt_std=1.0, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.ddg_mean, self.ddg_std = (ddg_mean, ddg_std)
        self.dt_mean, self.dt_std = (dt_mean, dt_std)
        self.augment = augment
        self.rot_axes_choices = [(1, 2), (1, 3), (2, 3)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(row.features_path).astype(np.float32)
        arr = arr.copy()
        arr[7:] -= arr[:7]
        if self.augment:
            axes = random.choice(self.rot_axes_choices)
            k = random.randint(0, 3)
            if k > 0:
                arr = np.rot90(arr, k=k, axes=axes).copy()
        x = torch.from_numpy(arr)
        y_ddg = torch.tensor((row.ddG - self.ddg_mean) / self.ddg_std, dtype=torch.float32)
        y_dt = torch.tensor((row.dT - self.dt_mean) / self.dt_std, dtype=torch.float32)
        return (x, y_ddg, y_dt)


import torch.nn as nn

class ThermoNet3D(nn.Module):

    def __init__(self, in_channels=14, conv_channels=(32, 64, 128), fc_hidden=64, dropout=0.3):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.conv_block1 = nn.Sequential(nn.Conv3d(in_channels, c1, kernel_size=3, padding=1), nn.BatchNorm3d(c1), nn.ReLU(), nn.MaxPool3d(2))
        self.conv_block2 = nn.Sequential(nn.Conv3d(c1, c2, kernel_size=3, padding=1), nn.BatchNorm3d(c2), nn.ReLU(), nn.MaxPool3d(2))
        self.conv_block3 = nn.Sequential(nn.Conv3d(c2, c3, kernel_size=3, padding=1), nn.BatchNorm3d(c3), nn.ReLU(), nn.AdaptiveAvgPool3d(1))
        self.fc_ddg = nn.Sequential(nn.Linear(c3, fc_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_hidden, 1))
        self.fc_dt = nn.Sequential(nn.Linear(c3, fc_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_hidden, 1))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)
        pred_ddg = self.fc_ddg(x).squeeze(1)
        pred_dt = self.fc_dt(x).squeeze(1)
        return (pred_ddg, pred_dt)


from torch.utils.data import DataLoader
C_DT = 0.01

def train_one_fold(fold, df, epochs=20, batch_size=32, lr=0.0001, device='cuda', weight_decay=0.0, c_dt=None, model_kwargs=None, verbose=True):
    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]
    ddg_mean, ddg_std = robust_stats(train_df['ddG'].dropna())
    dt_mean, dt_std = robust_stats(train_df['dT'].dropna())
    norm_stats = {'ddg_mean': ddg_mean, 'ddg_std': ddg_std, 'dt_mean': dt_mean, 'dt_std': dt_std}
    if verbose:
        print(f'[Fold {fold}] Normalisation (median/IQR) -> ddG: median={ddg_mean:.3f} IQR={ddg_std:.3f} | dT: median={dt_mean:.3f} IQR={dt_std:.3f}')
    train_ds = VoxelDataset(train_df, ddg_mean=ddg_mean, ddg_std=ddg_std, dt_mean=dt_mean, dt_std=dt_std, augment=True)
    val_ds = VoxelDataset(val_df, ddg_mean=ddg_mean, ddg_std=ddg_std, dt_mean=dt_mean, dt_std=dt_std, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = ThermoNet3D(**model_kwargs or {}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    c_dt_value = c_dt if c_dt is not None else C_DT
    losses = []
    ddg_losses = []
    dt_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_ddg_loss, epoch_dt_loss = (0.0, 0.0, 0.0)
        for x, y_ddg, y_dt in train_loader:
            x, y_ddg, y_dt = (x.to(device), y_ddg.to(device), y_dt.to(device))
            optimizer.zero_grad()
            pred_ddg, pred_dt = model(x)
            mask_ddg = ~torch.isnan(y_ddg)
            mask_dt = ~torch.isnan(y_dt)
            loss_ddg = mse(pred_ddg[mask_ddg], y_ddg[mask_ddg]) if mask_ddg.any() else torch.tensor(0.0, device=device, requires_grad=True)
            loss_dt = mse(pred_dt[mask_dt], y_dt[mask_dt]) if mask_dt.any() else torch.tensor(0.0, device=device, requires_grad=True)
            loss = loss_ddg + c_dt_value * loss_dt
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            epoch_ddg_loss += loss_ddg.item() * x.size(0)
            epoch_dt_loss += loss_dt.item() * x.size(0)
        n = len(train_df)
        losses.append(epoch_loss / n)
        ddg_losses.append(epoch_ddg_loss / n)
        dt_losses.append(epoch_dt_loss / n)
        if verbose:
            print(f'[Fold {fold}] Epoch {epoch + 1}/{epochs} - loss={losses[-1]:.4f} (ddg={ddg_losses[-1]:.4f}, dt={dt_losses[-1]:.4f})')
    return (model, val_loader, norm_stats, {'losses': losses, 'ddg_losses': ddg_losses, 'dt_losses': dt_losses})


from scipy.stats import spearmanr

@torch.no_grad()
def evaluate_fold(model, val_loader, norm_stats, device='cuda'):
    model.eval()
    all_pred_ddg, all_true_ddg = ([], [])
    all_pred_dt, all_true_dt = ([], [])
    for x, y_ddg, y_dt in val_loader:
        x = x.to(device)
        pred_ddg, pred_dt = model(x)
        all_pred_ddg.append(pred_ddg.cpu())
        all_true_ddg.append(y_ddg)
        all_pred_dt.append(pred_dt.cpu())
        all_true_dt.append(y_dt)
    pred_ddg = torch.cat(all_pred_ddg).numpy()
    true_ddg = torch.cat(all_true_ddg).numpy()
    pred_dt = torch.cat(all_pred_dt).numpy()
    true_dt = torch.cat(all_true_dt).numpy()
    pred_ddg = pred_ddg * norm_stats['ddg_std'] + norm_stats['ddg_mean']
    true_ddg = true_ddg * norm_stats['ddg_std'] + norm_stats['ddg_mean']
    pred_dt = pred_dt * norm_stats['dt_std'] + norm_stats['dt_mean']
    true_dt = true_dt * norm_stats['dt_std'] + norm_stats['dt_mean']
    mask_ddg = ~np.isnan(true_ddg)
    mask_dt = ~np.isnan(true_dt)
    mse_ddg = float(np.mean((pred_ddg[mask_ddg] - true_ddg[mask_ddg]) ** 2)) if mask_ddg.any() else float('nan')
    mse_dt = float(np.mean((pred_dt[mask_dt] - true_dt[mask_dt]) ** 2)) if mask_dt.any() else float('nan')
    corr_ddg = float(spearmanr(pred_ddg[mask_ddg], true_ddg[mask_ddg]).correlation) if mask_ddg.sum() > 1 else float('nan')
    corr_dt = float(spearmanr(pred_dt[mask_dt], true_dt[mask_dt]).correlation) if mask_dt.sum() > 1 else float('nan')
    print(f'  -> val MSE ddg={mse_ddg:.4f} | val MSE dt={mse_dt:.4f} | Spearman ddg={corr_ddg:.4f} | Spearman dt={corr_dt:.4f}')
    return {'mse_ddg': mse_ddg, 'mse_dt': mse_dt, 'corr_ddg': corr_ddg, 'corr_dt': corr_dt}


try:
    import optuna
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'optuna'])
    import optuna
from optuna.samplers import TPESampler
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device (Optuna):', DEVICE)
N_TRIALS = 20
OPTUNA_FOLDS = [0]
OPTUNA_EPOCHS = 8


def objective(trial):
    lr = trial.suggest_float('lr', 1e-05, 0.01, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-06, 0.01, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    c1 = trial.suggest_categorical('conv_c1', [16, 32, 64])
    c2 = trial.suggest_categorical('conv_c2', [32, 64, 128])
    c3 = trial.suggest_categorical('conv_c3', [64, 128, 256])
    fc_hidden = trial.suggest_categorical('fc_hidden', [32, 64, 128])
    c_dt = trial.suggest_float('c_dt', 0.01, 2.0, log=True)
    model_kwargs = dict(conv_channels=(c1, c2, c3), fc_hidden=fc_hidden, dropout=dropout)
    fold_scores = []
    for step, fold in enumerate(OPTUNA_FOLDS):
        model, val_loader, norm_stats, _ = train_one_fold(fold, df, epochs=OPTUNA_EPOCHS, batch_size=batch_size, lr=lr, device=DEVICE, weight_decay=weight_decay, c_dt=c_dt, model_kwargs=model_kwargs, verbose=False)
        eval_result = evaluate_fold(model, val_loader, norm_stats, device=DEVICE)
        corr_ddg = eval_result['corr_ddg']
        corr_dt = eval_result['corr_dt']
        corr_ddg = -1.0 if corr_ddg is None or np.isnan(corr_ddg) else corr_ddg
        corr_dt = -1.0 if corr_dt is None or np.isnan(corr_dt) else corr_dt
        score = 0.5 * corr_ddg + 0.5 * corr_dt
        fold_scores.append(score)
        trial.report(float(np.mean(fold_scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(fold_scores))
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=0))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print('Meilleur score (0.5*Spearman ddG + 0.5*Spearman dT):', study.best_value)
print('Meilleurs hyperparametres:', study.best_params)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)
if 'study' in globals():
    best = study.best_params
    BEST_MODEL_KWARGS = dict(conv_channels=(best['conv_c1'], best['conv_c2'], best['conv_c3']), fc_hidden=best['fc_hidden'], dropout=best['dropout'])
    BEST_LR = best['lr']
    BEST_WEIGHT_DECAY = best['weight_decay']
    BEST_BATCH_SIZE = best['batch_size']
    BEST_C_DT = best['c_dt']
    print('Hyperparametres Optuna appliques:', best)
else:
    BEST_MODEL_KWARGS = None
    BEST_LR = 0.0001
    BEST_WEIGHT_DECAY = 0.0
    BEST_BATCH_SIZE = 32
    BEST_C_DT = None
    print('Optuna non execute : utilisation des hyperparametres par defaut')
all_results = []
for fold in range(N_FOLDS):
    model, val_loader, norm_stats, train_history = train_one_fold(fold, df, epochs=20, batch_size=BEST_BATCH_SIZE, lr=BEST_LR, device=DEVICE, weight_decay=BEST_WEIGHT_DECAY, c_dt=BEST_C_DT, model_kwargs=BEST_MODEL_KWARGS)
    eval_result = evaluate_fold(model, val_loader, norm_stats, device=DEVICE)
    all_results.append({**train_history, **eval_result})
mean_spearman_ddg = np.mean([r['corr_ddg'] for r in all_results])
print(f'\nSpearman ddG moyen sur {N_FOLDS} folds: {mean_spearman_ddg:.4f}')


import subprocess, sys

def _pip_install_quiet(package: str):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', package], check=True)
try:
    import wandb
except ImportError:
    _pip_install_quiet('wandb')
    import wandb

def get_wandb_api_key(secret_label: str='WANDB_API_KEY') -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret(secret_label)
        if not api_key:
            raise ValueError('Secret vide')
        return api_key
    except Exception as exc:
        raise RuntimeError(f"Impossible de recuperer le secret '{secret_label}'. Verifie qu'il est bien ajoute dans Kaggle > Add-ons > Secrets et que l'option 'Attach' est activee pour ce notebook.") from exc
WANDB_API_KEY = get_wandb_api_key()
wandb.login(key=WANDB_API_KEY)
print('W&B: connexion reussie.')
del WANDB_API_KEY


WANDB_PROJECT = 'thermonet-nesp'
WANDB_ENTITY = None
WANDB_GROUP_OPTUNA = 'optuna-cnn3d-hpo'
WANDB_JOB_TYPE_TRIAL = 'hpo-trial'
WANDB_JOB_TYPE_FINAL = 'final-training'


def train_one_fold_wandb(fold, df, epochs=20, batch_size=32, lr=0.0001, device='cuda', weight_decay=0.0, c_dt=None, model_kwargs=None, wandb_run=None, log_freq=50, verbose=True):
    train_df = df[df['fold'] != fold]
    val_df = df[df['fold'] == fold]
    ddg_mean, ddg_std = robust_stats(train_df['ddG'].dropna())
    dt_mean, dt_std = robust_stats(train_df['dT'].dropna())
    norm_stats = {'ddg_mean': ddg_mean, 'ddg_std': ddg_std, 'dt_mean': dt_mean, 'dt_std': dt_std}
    train_ds = VoxelDataset(train_df, ddg_mean=ddg_mean, ddg_std=ddg_std, dt_mean=dt_mean, dt_std=dt_std, augment=True)
    val_ds = VoxelDataset(val_df, ddg_mean=ddg_mean, ddg_std=ddg_std, dt_mean=dt_mean, dt_std=dt_std, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    model = ThermoNet3D(**model_kwargs or {}).to(device)
    if wandb_run is not None:
        wandb.watch(model, log='all', log_freq=log_freq, log_graph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    c_dt_value = c_dt if c_dt is not None else C_DT
    losses, ddg_losses, dt_losses = ([], [], [])
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_ddg_loss, epoch_dt_loss = (0.0, 0.0, 0.0)
        for x, y_ddg, y_dt in train_loader:
            x, y_ddg, y_dt = (x.to(device), y_ddg.to(device), y_dt.to(device))
            optimizer.zero_grad()
            pred_ddg, pred_dt = model(x)
            mask_ddg = ~torch.isnan(y_ddg)
            mask_dt = ~torch.isnan(y_dt)
            loss_ddg = mse(pred_ddg[mask_ddg], y_ddg[mask_ddg]) if mask_ddg.any() else torch.tensor(0.0, device=device, requires_grad=True)
            loss_dt = mse(pred_dt[mask_dt], y_dt[mask_dt]) if mask_dt.any() else torch.tensor(0.0, device=device, requires_grad=True)
            loss = loss_ddg + c_dt_value * loss_dt
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            epoch_ddg_loss += loss_ddg.item() * x.size(0)
            epoch_dt_loss += loss_dt.item() * x.size(0)
            global_step += 1
        n = len(train_df)
        losses.append(epoch_loss / n)
        ddg_losses.append(epoch_ddg_loss / n)
        dt_losses.append(epoch_dt_loss / n)
        if verbose:
            print(f'[Fold {fold}] Epoch {epoch + 1}/{epochs} - loss={losses[-1]:.4f} (ddg={ddg_losses[-1]:.4f}, dt={dt_losses[-1]:.4f})')
        if wandb_run is not None:
            wandb.log({'fold': fold, 'epoch': epoch, 'train/loss_total': losses[-1], 'train/loss_ddg': ddg_losses[-1], 'train/loss_dt': dt_losses[-1]}, step=global_step)
    history = {'losses': losses, 'ddg_losses': ddg_losses, 'dt_losses': dt_losses}
    return (model, val_loader, norm_stats, history)


from optuna.integration.wandb import WeightsAndBiasesCallback
wandb_kwargs = {'project': WANDB_PROJECT, 'entity': WANDB_ENTITY, 'group': WANDB_GROUP_OPTUNA, 'job_type': WANDB_JOB_TYPE_TRIAL}
wandbc = WeightsAndBiasesCallback(metric_name='spearman_mean', wandb_kwargs=wandb_kwargs, as_multirun=True)

@wandbc.track_in_wandb()
def objective_wandb(trial):
    lr = trial.suggest_float('lr', 1e-05, 0.01, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-06, 0.01, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    c1 = trial.suggest_categorical('conv_c1', [16, 32, 64])
    c2 = trial.suggest_categorical('conv_c2', [32, 64, 128])
    c3 = trial.suggest_categorical('conv_c3', [64, 128, 256])
    fc_hidden = trial.suggest_categorical('fc_hidden', [32, 64, 128])
    c_dt = trial.suggest_float('c_dt', 0.01, 2.0, log=True)
    model_kwargs = dict(conv_channels=(c1, c2, c3), fc_hidden=fc_hidden, dropout=dropout)
    fold_scores = []
    for step, fold in enumerate(OPTUNA_FOLDS):
        model, val_loader, norm_stats, _ = train_one_fold_wandb(fold, df, epochs=OPTUNA_EPOCHS, batch_size=batch_size, lr=lr, device=DEVICE, weight_decay=weight_decay, c_dt=c_dt, model_kwargs=model_kwargs, wandb_run=wandb.run, verbose=False)
        eval_result = evaluate_fold(model, val_loader, norm_stats, device=DEVICE)
        corr_ddg = eval_result['corr_ddg']
        corr_dt = eval_result['corr_dt']
        corr_ddg = -1.0 if corr_ddg is None or np.isnan(corr_ddg) else corr_ddg
        corr_dt = -1.0 if corr_dt is None or np.isnan(corr_dt) else corr_dt
        score = 0.5 * corr_ddg + 0.5 * corr_dt
        fold_scores.append(score)
        wandb.log({'val/mse_ddg': eval_result['mse_ddg'], 'val/mse_dt': eval_result['mse_dt'], 'val/spearman_ddg': corr_ddg, 'val/spearman_dt': corr_dt, 'val/spearman_mean': score, 'fold': fold})
        trial.report(float(np.mean(fold_scores)), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    final_score = float(np.mean(fold_scores))
    wandb.summary['spearman_mean'] = final_score
    return final_score


study_wandb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=0), study_name='thermonet-cnn3d-wandb')
study_wandb.optimize(objective_wandb, n_trials=N_TRIALS, callbacks=[wandbc])
print('Meilleur score (0.5*Spearman ddG + 0.5*Spearman dT):', study_wandb.best_value)
print('Meilleurs hyperparametres:', study_wandb.best_params)
print('Tous les trials sont visibles dans le projet W&B:', WANDB_PROJECT)


best = study_wandb.best_params
FINAL_MODEL_KWARGS = dict(conv_channels=(best['conv_c1'], best['conv_c2'], best['conv_c3']), fc_hidden=best['fc_hidden'], dropout=best['dropout'])
run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type=WANDB_JOB_TYPE_FINAL, config={**best, 'n_folds': N_FOLDS, 'epochs': 20}, name='final-training-best-hp')
all_results = []
for fold in range(N_FOLDS):
    model, val_loader, norm_stats, train_history = train_one_fold_wandb(fold, df, epochs=20, batch_size=best['batch_size'], lr=best['lr'], device=DEVICE, weight_decay=best['weight_decay'], c_dt=best['c_dt'], model_kwargs=FINAL_MODEL_KWARGS, wandb_run=run)
    eval_result = evaluate_fold(model, val_loader, norm_stats, device=DEVICE)
    all_results.append({**train_history, **eval_result})
    wandb.log({'final/fold': fold, 'final/spearman_ddg': eval_result['corr_ddg'], 'final/spearman_dt': eval_result['corr_dt'], 'final/mse_ddg': eval_result['mse_ddg'], 'final/mse_dt': eval_result['mse_dt']})
mean_spearman_ddg = float(np.mean([r['corr_ddg'] for r in all_results]))
mean_spearman_dt = float(np.mean([r['corr_dt'] for r in all_results]))
wandb.summary['mean_spearman_ddg'] = mean_spearman_ddg
wandb.summary['mean_spearman_dt'] = mean_spearman_dt
print(f'Spearman ddG moyen sur {N_FOLDS} folds: {mean_spearman_ddg:.4f}')
print(f'Spearman dT moyen sur {N_FOLDS} folds: {mean_spearman_dt:.4f}')
wandb.finish()
