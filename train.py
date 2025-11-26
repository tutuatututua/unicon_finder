"""LambdaRank training (LightGBM) with panel or snapshot modes.

Goals of this refactor:
  * Keep public API (`train_model_learn_to_rank`) + artifact contract identical.
  * Reduce line count & duplication; centralize repeated logic.
  * Maintain existing metric keys (valid_ndcg@K, manual_ndcg, spearman_* etc.).
  * Preserve JSON + CSV artifact locations.

Two dataset modes:
  1. Full historical panel (per-date groups) from `data/processed/raw_training.parquet`.
  2. Legacy snapshot (chunk groups) from `data/processed/extract_training.parquet`.

Returned metrics dict matches pipeline expectations.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json, math, re

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PRIMARY_TRAINING_SNAPSHOT = Path("data/processed/extract_training.parquet")
RAW_PANEL_PATH = Path("data/processed/raw_training.parquet")
FEATURE_META = Path("models/features.json")
MODEL_PATH = Path("models/lightgbm_model.txt")
METRICS_PATH = Path("models/metrics.json")
FI_CSV = Path("models/feature_importance.csv")
FI_PNG = Path("models/feature_importance.png")

def ensure_dir(p: Path) -> None:
    tgt = p if p.suffix == "" else p.parent
    tgt.mkdir(parents=True, exist_ok=True)

ensure_dir(MODEL_PATH)

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _json_default(obj):  # compact version
    try:
        import numpy as _np, pandas as _pd
        if isinstance(obj, _np.integer): return int(obj)
        if isinstance(obj, _np.floating): return float(obj)
        if isinstance(obj, _np.bool_): return bool(obj)
        if isinstance(obj, _np.ndarray): return obj.tolist()
        if obj is _pd.NA: return None
        if hasattr(obj, 'to_pydatetime'):
            try: return obj.to_pydatetime().isoformat()
            except Exception: pass
    except Exception: pass
    return str(obj)

def _load_dynamic_target(default: str = "target_fwd_252d") -> str:
    meta_path = FEATURE_META
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cand = str(meta.get("target") or '').strip()
            if cand:
                logger.info("Using dynamic target column: %s", cand)
                return cand
    except Exception as e:
        logger.warning("Failed reading target metadata: %s", e)
    return default

TARGET_COL = _load_dynamic_target()
DEFAULT_TARGET_CLIP = 30.0

def _infer_forward_days_from_target(name: str, default: int = 252) -> int:
    m = re.search(r"target_fwd_(\d+)d", str(name))
    return int(m.group(1)) if m else int(default)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RankTrainConfig:
    target_clip: float = DEFAULT_TARGET_CLIP
    n_relevance_bins: int = 6
    params: Optional[Dict[str, Any]] = None
    seed: int = 42
    export_predictions: bool = True
    model: str = "lightgbm"
    manual_test_ndcg: bool = True
    manual_test_cutoffs: Tuple[int, ...] = (5, 10, 20)
    force_negatives_to_zero: bool = False
    use_full_panel: bool = False
    group_by: str = "date"  # kept for backwards compatibility
    min_cross_section: int = 200
    valid_months: int = 12
    test_months: int = 0
    forward_gap_days: Optional[int] = 252
    train_window_years: Optional[int] = 3
    stratified_grouping: bool = True
    proportional_stratification: bool = True
    group_chunk: int = 200
    recency_lambda: float = 0.0
    train_date_step: int = 63
    valid_date_step: int = 21
    test_date_step: int = 21
    valid_fraction: float = 0.2  # legacy snapshot
    test_fraction: float = 0.0   # legacy snapshot

    def __post_init__(self):
        if self.params is None:
            self.params = {
                "objective": "lambdarank","metric": "ndcg","learning_rate": 0.05,
                "num_leaves": 11,"feature_fraction": 0.6,"bagging_fraction": 0.6,
                "bagging_freq": 2,"max_depth": 6,"min_data_in_leaf": 300,
                "lambda_l2": 20.0,"lambda_l1": 0.8,"min_gain_to_split": 0.05,
                "cat_l2": 10.0,"cat_smooth": 20.0,"max_bin": 127,
                "min_sum_hessian_in_leaf": 5.0,"boosting": "gbdt",
                "feature_pre_filter": False,"seed": 42,"verbose": -1,
                "eval_at": [20, 10, 5],
            }
        if not (0 <= self.test_fraction < 1): raise ValueError("test_fraction range error")
        if not (0 < self.valid_fraction < 1): raise ValueError("valid_fraction range error")
        if self.valid_fraction + self.test_fraction >= 0.9: raise ValueError("validation+test too large")
        if self.group_chunk < 5: raise ValueError("group_chunk must be >=5")
        if self.train_date_step < 1: raise ValueError("train_date_step must be >=1")
        if self.valid_date_step < 1: raise ValueError("valid_date_step must be >=1")
        if self.test_date_step < 1: raise ValueError("test_date_step must be >=1")

# ---------------------------------------------------------------------------
# Binning / metrics helpers
# ---------------------------------------------------------------------------
def _safe_quantile_binning(s: pd.Series, n_bins: int) -> tuple[pd.Series, List[float]]:
    try:
        labels, edges = pd.qcut(s, q=n_bins, labels=False, retbins=True, duplicates="drop")
        return labels.astype(int), edges.tolist()
    except Exception as e:
        logger.warning("qcut failed (%s); fallback percentiles", e)
        edges = np.percentile(s, np.linspace(0,100,n_bins+1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index), edges.tolist()
        bins_idx = np.digitize(s, edges[1:-1], right=True)
        return pd.Series(bins_idx, index=s.index, dtype=int), edges.tolist()

def _manual_ndcg(labels: np.ndarray, preds: np.ndarray, group_sizes: List[int], ks: List[int]) -> Dict[str,float]:
    out: Dict[str,List[float]] = {}; start=0
    for g in group_sizes:
        end = start+g
        l = labels[start:end]; p = preds[start:end]
        order = np.argsort(-p); l_sorted = l[order]; ideal = np.sort(l)[::-1]
        for k in ks:
            if g==0: continue
            k_eff = min(k,g); gain = (2**l_sorted[:k_eff]-1); disc = 1/np.log2(np.arange(2,k_eff+2))
            dcg = float(np.sum(gain*disc)); ideal_gain = (2**ideal[:k_eff]-1); idcg = float(np.sum(ideal_gain*disc))
            val = dcg/idcg if idcg>0 else 0.0
            out.setdefault(f"ndcg@{k}",[]).append(val)
        start=end
    return {f"manual_{k}": float(np.mean(v)) for k,v in out.items()}

def _compute_group_stats(sizes: List[int]) -> Dict[str,float]:
    if not sizes: return {"min":0,"max":0,"mean":0.0,"std":0.0}
    arr = np.asarray(sizes)
    return {"min":int(arr.min()),"max":int(arr.max()),"mean":float(arr.mean()),"std":float(arr.std())}

# ---------------------------------------------------------------------------
# Dataset preparation (panel & legacy) -> unified structure
# ---------------------------------------------------------------------------
@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_rel_train: pd.Series
    y_rel_valid: pd.Series
    y_rel_test: pd.Series
    y_cont_valid: pd.Series
    y_cont_test: pd.Series
    group_train: List[int]
    group_valid: List[int]
    group_test: List[int]
    meta: Dict[str,Any]
    train_weights: Optional[np.ndarray] = None

def _assign_relevance(df: pd.DataFrame, cfg: RankTrainConfig) -> pd.DataFrame:
    if cfg.target_clip and cfg.target_clip>0:
        df[TARGET_COL] = df[TARGET_COL].clip(-cfg.target_clip, cfg.target_clip)
    q_labels, edges = _safe_quantile_binning(df[TARGET_COL], cfg.n_relevance_bins)
    df['relevance'] = q_labels.astype(int)
    if cfg.force_negatives_to_zero:
        neg_mask = df[TARGET_COL] < 0
        if neg_mask.any(): df.loc[neg_mask,'relevance'] = 0
    logger.info("Relevance bin edges: %s", edges)
    return df

def _snapshot_bundle(cfg: RankTrainConfig, *, min_group_size: int = 100) -> DatasetBundle:
    if not PRIMARY_TRAINING_SNAPSHOT.exists():
        raise FileNotFoundError(f"Missing {PRIMARY_TRAINING_SNAPSHOT}")
    df = pd.read_parquet(PRIMARY_TRAINING_SNAPSHOT).dropna(subset=[TARGET_COL])
    if df.empty: raise RuntimeError("No labeled rows in snapshot")
    for c in ('sector','industry'):
        if c in df.columns: df[c]=df[c].astype('category')
    df = _assign_relevance(df, cfg)
    exclude = {"ticker","date",TARGET_COL,"relevance"}
    feature_cols = [c for c in df.columns if c not in exclude]
    df_reset = df.reset_index(drop=True)
    X = df_reset[feature_cols]; y_rel = df_reset['relevance'].astype(int); y_cont = df_reset[TARGET_COL]
    rng = np.random.default_rng(cfg.seed); n_total = len(X); GROUP_CHUNK = cfg.group_chunk
    if cfg.stratified_grouping:
        order = np.argsort(y_rel.to_numpy())
        n_groups_target = max(1, math.ceil(n_total/GROUP_CHUNK))
        if cfg.proportional_stratification:
            label_groups: Dict[int,List[int]] = {}
            for idx_i in order:
                lab=int(y_rel.iloc[idx_i]); label_groups.setdefault(lab,[]).append(int(idx_i))
            buckets=[[] for _ in range(n_groups_target)]
            for lab, arr in label_groups.items():
                rng.shuffle(arr)
                for pos,val in enumerate(arr): buckets[pos % n_groups_target].append(val)
            for b in buckets: rng.shuffle(b)
            idx = np.concatenate(buckets) if buckets else np.empty(0,dtype=int)
        else:
            buckets=[[] for _ in range(n_groups_target)]
            for i,v in enumerate(order): buckets[i % n_groups_target].append(int(v))
            for b in buckets: rng.shuffle(b)
            idx = np.concatenate(buckets) if buckets else np.empty(0,dtype=int)
    else:
        idx = np.arange(n_total); rng.shuffle(idx)
    group_ids = np.full(n_total,-1,dtype=int); groups: List[int]=[]; gid=0
    for start in range(0,len(idx),GROUP_CHUNK):
        chunk = idx[start:start+GROUP_CHUNK]
        if len(chunk)<min_group_size and gid>0:
            group_ids[chunk]=gid-1; groups[-1]+=len(chunk); break
        group_ids[chunk]=gid; groups.append(len(chunk)); gid+=1
    if (group_ids<0).any(): raise RuntimeError("Unassigned rows after grouping")
    unique_gids = np.unique(group_ids); rng.shuffle(unique_gids)
    n_groups = len(unique_gids); n_test = int(round(cfg.test_fraction*n_groups)); n_valid = max(1,int(round(cfg.valid_fraction*n_groups)))
    test_ids = list(unique_gids[:n_test]); valid_ids = list(unique_gids[n_test:n_test+n_valid]); train_ids = list(unique_gids[n_test+n_valid:])
    if not train_ids:
        if len(valid_ids)>1: train_ids.append(valid_ids.pop())
        elif len(test_ids)>1: train_ids.append(test_ids.pop())
    group_to_rows = {int(g): np.where(group_ids==g)[0] for g in unique_gids}
    def _collect(ids: List[int]):
        rows = [group_to_rows[g] for g in ids]; idx_cat = np.concatenate(rows) if rows else np.empty(0,dtype=int)
        sizes = [len(r) for r in rows]
        return X.iloc[idx_cat].reset_index(drop=True), y_rel.iloc[idx_cat].reset_index(drop=True), y_cont.iloc[idx_cat].reset_index(drop=True), sizes
    X_train, y_rel_train, y_cont_train, group_train = _collect(train_ids)
    X_valid, y_rel_valid, y_cont_valid, group_valid = _collect(valid_ids)
    X_test, y_rel_test, y_cont_test, group_test = _collect(test_ids)
    meta = {
        "feature_cols": feature_cols,
        "train_rows": len(X_train),"valid_rows": len(X_valid),"test_rows": len(X_test),
        "n_groups_train": len(group_train),"n_groups_valid": len(group_valid),"n_groups_test": len(group_test),
        "avg_group_size_train": float(np.mean(group_train) if group_train else 0.0),
    }
    # categorical levels for prediction alignment
    try:
        cat_levels: Dict[str,List[str]] = {}
        for c in ('sector','industry'):
            if c in df_reset.columns and hasattr(df_reset[c],'cat'):
                cat_levels[c] = list(map(str, list(df_reset[c].cat.categories)))
        if cat_levels: meta['categorical_levels']=cat_levels
    except Exception: pass
    return DatasetBundle(X_train,X_valid,X_test,y_rel_train,y_rel_valid,y_rel_test,y_cont_valid,y_cont_test,group_train,group_valid,group_test,meta,None)

def _panel_bundle(cfg: RankTrainConfig) -> DatasetBundle:
    if not RAW_PANEL_PATH.exists(): raise FileNotFoundError(f"Missing {RAW_PANEL_PATH}")
    df = pd.read_parquet(RAW_PANEL_PATH)
    if df.empty: raise RuntimeError("Panel features empty")
    df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=[TARGET_COL], how='all')
    for c in ('sector','industry'):
        if c in df.columns: df[c]=df[c].astype('category')
    df = _assign_relevance(df, cfg)
    # filter by cross-section
    counts = df.groupby('date').size(); keep_dates = counts[counts>=cfg.min_cross_section].index
    df = df[df['date'].isin(keep_dates)].copy()
    if df.empty: raise RuntimeError("No dates pass min_cross_section")
    end_date = df['date'].max(); has_test = cfg.test_months>0
    test_start = (end_date - pd.DateOffset(months=cfg.test_months)) if has_test else end_date
    valid_start = (test_start - pd.DateOffset(months=cfg.valid_months)) if cfg.valid_months>0 else test_start
    fwd_days = cfg.forward_gap_days or _infer_forward_days_from_target(TARGET_COL,252)
    uniq_dates = np.array(sorted(df['date'].unique()))
    
    try:
        idx_valid = int(np.searchsorted(uniq_dates, np.datetime64(valid_start), 'left'))
        train_end_idx = max(0, idx_valid - fwd_days - 1)
        train_end = pd.Timestamp(uniq_dates[train_end_idx]).tz_localize(None).normalize()
    except Exception:
        train_end = valid_start - pd.Timedelta(days=fwd_days)
    
    train_start = (train_end - pd.DateOffset(years=cfg.train_window_years)) if (cfg.train_window_years and cfg.train_window_years>0) else df['date'].min()
    
    def _split(d: pd.Timestamp) -> str:
        if has_test and d >= test_start: return 'test'
        if d >= valid_start and (not has_test or d < test_start): return 'valid'
        if d <= train_end: return 'train'
        return 'drop'
    df['split'] = df['date'].apply(_split)
    df.loc[(df['split']=='train') & (df['date']<train_start),'split'] = 'drop'
    df = df[df['split']!='drop'].copy()
    df.drop(columns=['industry','sector'], inplace=True)
    # Log unique dates per split for transparency
    try:
        for part in ('train','valid','test'):
            sub_dates = np.array(sorted(df.loc[df['split']==part, 'date'].unique()))
            if sub_dates.size:
                logger.info("%s dates: %d | %s -> %s", part.capitalize(), sub_dates.size, pd.Timestamp(sub_dates[0]).date(), pd.Timestamp(sub_dates[-1]).date())
            else:
                logger.info("%s dates: 0", part.capitalize())
    except Exception as _e:
        logger.debug("Unique date logging skipped: %s", _e)
    logger.info(f"train_unique: {df.loc[(df['split']=='train'),'date'].unique()}")
    logger.info(f"validate_unique: {df.loc[(df['split']=='valid'),'date'].unique()}")
    logger.info(f"test_unique: {df.loc[(df['split']=='test'),'date'].unique()}")
    exclude = {"ticker","date",TARGET_COL,"relevance","split"}
    feature_cols = [c for c in df.columns if c not in exclude]
    def _collect(name: str):
        sub = df[df['split']==name].copy()
        if name in ('train','valid','test') and not sub.empty:
            step = cfg.train_date_step if name=='train' else (cfg.valid_date_step if name=='valid' else cfg.test_date_step)
            uniq_before = np.array(sorted(sub['date'].unique()))
            try:
                if uniq_before.size:
                    logger.info("%s pre-step: dates=%d, step=%d, range=%s->%s", name, uniq_before.size, step, pd.Timestamp(uniq_before[0]).date(), pd.Timestamp(uniq_before[-1]).date())
            except Exception: pass
            if step > 1:
                keep = set(uniq_before[::step])
                sub = sub[sub['date'].isin(keep)].copy()
                uniq_after = np.array(sorted(sub['date'].unique()))
                try:
                    head = ", ".join(str(pd.Timestamp(x).date()) for x in uniq_after[:5])
                    tail = ", ".join(str(pd.Timestamp(x).date()) for x in uniq_after[-5:]) if uniq_after.size>5 else ""
                    logger.info("%s post-step: kept=%d/%d (%.1f%%), range=%s->%s, head=[%s]%s", name, uniq_after.size, uniq_before.size, (100.0*uniq_after.size/max(1,uniq_before.size)), pd.Timestamp(uniq_after[0]).date() if uniq_after.size else None, pd.Timestamp(uniq_after[-1]).date() if uniq_after.size else None, head, (", tail=["+tail+"]" if tail else ""))
                except Exception: pass
        if sub.empty:
            return pd.DataFrame(columns=feature_cols), pd.Series(dtype=int), pd.Series(dtype=float), [], np.array([],dtype=float)
        sub = sub.sort_values([c for c in ('date','ticker') if c in sub.columns])
        groups = sub.groupby('date', sort=False); group_sizes = groups.size().tolist(); X = sub[feature_cols].reset_index(drop=True)
        y_rel = sub['relevance'].astype(int).reset_index(drop=True); y_cont = sub[TARGET_COL].reset_index(drop=True)
        if name=='train' and cfg.recency_lambda>0:
            age_years = (end_date - sub['date']).dt.days.to_numpy()/365.25
            weights = np.exp(-cfg.recency_lambda*age_years)
        else: weights = np.ones(len(sub),dtype=float)
        return X,y_rel,y_cont,group_sizes,weights
    X_train,y_rel_train,y_cont_train,group_train,w_train = _collect('train')
    X_valid,y_rel_valid,y_cont_valid,group_valid,_wv = _collect('valid')
    X_test,y_rel_test,y_cont_test,group_test,_wt = _collect('test')

    meta = {
        'feature_cols': feature_cols,'train_rows': len(X_train),'valid_rows': len(X_valid),'test_rows': len(X_test),
        'n_groups_train': len(group_train),'n_groups_valid': len(group_valid),'n_groups_test': len(group_test),
        'avg_group_size_train': float(np.mean(group_train) if group_train else 0.0),
        'split_params': {'valid_months': cfg.valid_months,'test_months': cfg.test_months,'forward_gap_days': fwd_days,'train_window_years': cfg.train_window_years,'min_cross_section': cfg.min_cross_section},
        'date_range': {'min': str(df['date'].min().date()),'max': str(df['date'].max().date()),'train_start': str(train_start.date()),'train_end': str(train_end.date()),'valid_start': str(valid_start.date()),'test_start': (str(test_start.date()) if has_test else None)},
        'train_weights_sum': float(np.sum(w_train) if w_train.size else 0.0),
        'date_steps': {'train': cfg.train_date_step, 'valid': cfg.valid_date_step, 'test': cfg.test_date_step},
    }
    try:
        grp_sum = df.groupby(['split','date']).size().rename('size').reset_index().sort_values(['split','date'])
        grp_sum.to_csv(MODEL_PATH.parent/"grouping_summary.csv", index=False)
    except Exception: pass
    return DatasetBundle(X_train,X_valid,X_test,y_rel_train,y_rel_valid,y_rel_test,y_cont_valid,y_cont_test,group_train,group_valid,group_test,meta,w_train)

# ---------------------------------------------------------------------------
# LightGBM dataset builder
# ---------------------------------------------------------------------------
def _build_lgb_datasets(X_train: pd.DataFrame, y_rel_train: pd.Series, group_train: List[int], X_valid: pd.DataFrame, y_rel_valid: pd.Series, group_valid: List[int], *, dataset_params: Optional[Dict[str,Any]] = None, train_weights: Optional[np.ndarray] = None) -> tuple[lgb.Dataset,lgb.Dataset]:
    lgb_train = lgb.Dataset(X_train,label=y_rel_train.values,group=group_train,weight=(train_weights if train_weights is not None else None),free_raw_data=False,params=(dataset_params or None))
    lgb_valid = lgb.Dataset(X_valid,label=y_rel_valid.values,group=group_valid,reference=lgb_train,free_raw_data=False,params=(dataset_params or None))
    return lgb_train,lgb_valid

# ---------------------------------------------------------------------------
# Hyperparameter tuning (random search)
# ---------------------------------------------------------------------------
def _random_choice(rng: np.random.Generator, vals: List[Any]) -> Any: return vals[int(rng.integers(0,len(vals)))]

def _sample_from_space(rng: np.random.Generator, space: Dict[str,Any]) -> Dict[str,Any]:
    out: Dict[str,Any] = {}
    for k,spec in (space or {}).items():
        if spec is None or isinstance(spec,(str,bool)): out[k]=spec; continue
        if isinstance(spec,(list,tuple)) and len(spec)>0 and not (len(spec)==2 and all(isinstance(x,(int,float)) for x in spec)):
            out[k]=_random_choice(rng,list(spec)); continue
        if isinstance(spec,tuple) and len(spec)==2 and all(isinstance(x,(int,float)) for x in spec):
            low,high=spec
            out[k]= int(rng.integers(low,high+1)) if isinstance(low,int) and isinstance(high,int) else float(rng.uniform(float(low),float(high)))
            continue
        if isinstance(spec,dict) and 'low' in spec and 'high' in spec:
            low,high = float(spec['low']), float(spec['high']); typ = spec.get('type','float'); log = bool(spec.get('log',False))
            sample = float(np.exp(rng.uniform(np.log(max(low,1e-12)), np.log(max(high,low*1.000001))))) if log else float(rng.uniform(low,high))
            out[k] = int(round(sample)) if typ=='int' else float(sample); continue
        out[k]=spec
    return out

def tune_hyperparameters(param_grid=None, max_evals=50, seed=42, extra_fixed_params=None, cfg: Optional[RankTrainConfig] = None, bundle: Optional[DatasetBundle] = None):
    """Random search over LambdaRank LightGBM params.

    Enhancements:
      * Accepts existing `cfg` (RankTrainConfig) so caller overrides are honored.
      * Accepts pre-built `bundle` to avoid redundant dataset construction.
      * Multi-cutoff NDCG score (smoothed) prioritizes lower ks via (10/k) scaling.
    """

    def combined_ndcg_score(evals, ks, smooth=3):
        valid = evals.get("valid", {})
        vals = []
        for k in ks:
            series = valid.get(f"ndcg@{k}", [])
            if not series:
                return float("-inf")
            tail = series[-smooth:] if len(series) >= smooth else series
            vals.append(float(np.mean(tail)) * 10 / k)
        return float(np.mean(vals))

    if cfg is None:
        cfg = RankTrainConfig()
    if bundle is None:
        bundle = (_panel_bundle(cfg) if (cfg.use_full_panel and RAW_PANEL_PATH.exists()) else _snapshot_bundle(cfg))

    base = {k: v for k, v in (cfg.params or {}).items()}
    base.update({"objective": "lambdarank", "metric": "ndcg"})
    base.setdefault("eval_at", [ 20, 10, 5])
    max_rel = int(max(bundle.y_rel_train.max(), bundle.y_rel_valid.max()))
    base.setdefault("label_gain", [int(2 ** i - 1) for i in range(max_rel + 1)])
    if extra_fixed_params:
        base.update(extra_fixed_params)

    ks_used = sorted(base["eval_at"])
    rng = np.random.default_rng(seed)
    trials: List[Dict[str, Any]] = []
    trials_sus: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_params: Dict[str, Any] = {}

    for i in range(int(max_evals)):
        trial_params = _sample_from_space(rng, param_grid or {})
        params = {"objective": "lambdarank", "metric": "ndcg", **base, **trial_params}
        evals: Dict[str, Dict[str, List[float]]] = {}
        try:
            ds_params = {"max_bin": int(params["max_bin"])} if params.get("max_bin") else None
            lgb_train, lgb_valid = _build_lgb_datasets(
                bundle.X_train, bundle.y_rel_train, bundle.group_train,
                bundle.X_valid, bundle.y_rel_valid, bundle.group_valid,
                dataset_params=ds_params,
                train_weights=bundle.train_weights
            )
            booster = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_valid, lgb_train],
                valid_names=["valid", "train"],
                num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(40, verbose=False),
                    lgb.record_evaluation(evals)
                ]
            )
        except Exception as e:
            logger.warning("Trial %d failed: %s", i + 1, e)
            continue

        score = combined_ndcg_score(evals, ks_used, smooth=3)
        row: Dict[str, Any] = {"trial": i + 1, "score": score, "best_iteration": int(getattr(booster, "best_iteration", -1) or -1)}
        row.update(trial_params)
        trials.append(row)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_params = trial_params
            trials_sus.append(row)
            logger.info("New best @trial %d: combined_ndcg=%.5f", i + 1, score)

    tuning_df = pd.DataFrame(trials)
    tuning_csv = MODEL_PATH.parent / "tuning_results.csv"
    ensure_dir(tuning_csv.parent)
    tuning_df.sort_values("score", ascending=False).to_csv(tuning_csv, index=False)


    try:
        df_plot = pd.DataFrame(trials_sus).sort_values("trial").reset_index(drop=True)
        param_cols = [c for c in df_plot.columns if c not in {"trial","score","best_iteration"}]
        if param_cols:
            # Normalize columns to [0,1] for a shared y-scale; categorical mapped to codes
            normed: Dict[str, np.ndarray] = {}
            cat_maps: Dict[str, Dict[Any, int]] = {}
            for col in param_cols:
                s = df_plot[col]
                if s.dtype == object or str(s.dtype).startswith("category") or s.dtype == bool:
                    uniq = list(dict.fromkeys(s.tolist()))
                    m = {v: i for i, v in enumerate(uniq)}
                    vals = s.map(m).astype(float).to_numpy()
                    cat_maps[col] = m
                    # normalize codes to [0,1] if >1 distinct
                    if len(uniq) > 1:
                        vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals) + 1e-12)
                    else:
                        vals = np.zeros_like(vals)
                    normed[col] = vals
                else:
                    arr = s.astype(float).to_numpy()
                    mn, mx = float(np.min(arr)), float(np.max(arr))
                    vals = (arr - mn) / (mx - mn + 1e-12) if mx > mn else np.zeros_like(arr)
                    normed[col] = vals

            plt.figure(figsize=(11, 6))
            x = df_plot["trial"].to_numpy()
            for col in param_cols:
                plt.plot(x, normed[col], label=str(col), linewidth=1.3, alpha=0.9)

            # Overlay normalized score on secondary y-axis for reference
            ax = plt.gca()
            ax2 = ax.twinx()
            score = df_plot["score"].to_numpy()
            score_n = (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-12)
            ax2.plot(x, score_n, color="#444444", linestyle="--", linewidth=1.5, label="score(norm)",marker='o', )
            ax2.set_ylabel("score (norm)")

            # Mark best trial
            best_idx = int(np.argmax(score)) if len(score) else -1
            if best_idx >= 0:
                ax.axvline(x[best_idx], color="#999999", linestyle=":", linewidth=1)

            ax.set_title("Tuning Parameters Over Trials (normalized); dashed = score")
            ax.set_xlabel("Trial")
            ax.set_ylabel("parameter value (normalized)")
            # Build combined legend
            lines_labels = []
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            lines_labels.extend(zip(h1, l1))
            lines_labels.extend(zip(h2, l2))
            handles, labels = zip(*lines_labels) if lines_labels else ([], [])
            ax.legend(handles, labels, ncol=2, fontsize=8, loc="upper left", framealpha=0.85)
            ax.grid(alpha=0.2, linestyle="--")
            plt.tight_layout()
            combo_png = MODEL_PATH.parent / "tuning_parameters_combined.png"
            plt.savefig(combo_png, dpi=140)
            plt.close()

            # Save categorical mappings to a CSV for readability
            if cat_maps:
                rows = []
                for k, mp in cat_maps.items():
                    for v, code in mp.items():
                        rows.append({"parameter": k, "category": str(v), "code": int(code)})
                map_df = pd.DataFrame(rows)
                map_csv = MODEL_PATH.parent / "tuning_param_category_mapping.csv"
                map_df.to_csv(map_csv, index=False)
        else:
            combo_png = None
    except Exception as e:
        logger.warning("Combined tuning parameter plot failed: %s", e)
        combo_png = None

    return {"best_params": {**base, **best_params}, "best_score": float(best_score), "trials_csv": str(tuning_csv),  "trials_plot_combined": (str(combo_png) if combo_png else None)}


# ---------------------------------------------------------------------------
# Training core
# ---------------------------------------------------------------------------
def _prepare_bundle(cfg: RankTrainConfig) -> DatasetBundle:
    if cfg.use_full_panel and RAW_PANEL_PATH.exists(): return _panel_bundle(cfg)
    return _snapshot_bundle(cfg)

def _train_booster(cfg: RankTrainConfig, bundle: DatasetBundle, *, params: Dict[str,Any], num_boost_round: int, early_stopping_rounds: int) -> Tuple[lgb.Booster, Dict[str, Dict[str, List[float]]]]:
    ds_params = {"max_bin": int(params['max_bin'])} if 'max_bin' in params and params['max_bin'] is not None else None
    lgb_train,lgb_valid = _build_lgb_datasets(bundle.X_train,bundle.y_rel_train,bundle.group_train,bundle.X_valid,bundle.y_rel_valid,bundle.group_valid,dataset_params=ds_params,train_weights=bundle.train_weights)
    evals: Dict[str,Dict[str,List[float]]] = {}
    booster = lgb.train(params,lgb_train,valid_sets=[lgb_valid,lgb_train],valid_names=["valid","train"],num_boost_round=int(num_boost_round),callbacks=[lgb.early_stopping(int(early_stopping_rounds),verbose=True), lgb.log_evaluation(period=50), lgb.record_evaluation(evals)])
    return booster, evals

def _extract_ndcg_metrics(evals_result: Dict[str,Dict[str,List[float]]], booster: lgb.Booster) -> Dict[str,float]:
    out: Dict[str,float] = {}; best_iter = int(getattr(booster,'best_iteration',-1) or -1)
    def _val(series: List[float]) -> float:
        if best_iter>0 and best_iter <= len(series): return float(series[best_iter-1])
        return float(series[-1])
    for scope in ('valid','train'):
        for k,series in evals_result.get(scope, {}).items():
            if k.startswith('ndcg') and series:
                key = f"{scope}_{k}"; out[key] = float(series[-1])
                out[f"{scope}_best_{k}"] = _val(series)
    return out

def _plot_learning_curves(evals_result: Dict[str,Dict[str,List[float]]], booster: lgb.Booster) -> None:
    if not evals_result: return
    metric_keys = [m for m in evals_result.get('train', {}).keys() if m.startswith('ndcg')]
    rows = len(next(iter(evals_result['train'].values()))) if metric_keys else 0
    curve_rows=[]
    for i in range(rows):
        row: Dict[str, Any] = {'iteration': i+1}
        for mk in metric_keys:
            row[f'train_{mk}']=evals_result['train'][mk][i]
            if mk in evals_result.get('valid', {}): row[f'valid_{mk}']=evals_result['valid'][mk][i]
        curve_rows.append(row)
    curve_df = pd.DataFrame(curve_rows)
    lc_csv = MODEL_PATH.parent/'learning_curve_ndcg.csv'; curve_df.to_csv(lc_csv,index=False)
    try:
        plt.figure(figsize=(8, 4+len(metric_keys)))
        for idx,mk in enumerate(metric_keys, start=1):
            plt.subplot(len(metric_keys),1,idx)
            plt.plot(curve_df['iteration'], curve_df[f'train_{mk}'], label='train', color='#1b9e77')
            if f'valid_{mk}' in curve_df: plt.plot(curve_df['iteration'], curve_df[f'valid_{mk}'], label='valid', color='#d95f02')
            plt.axvline(int(getattr(booster,'best_iteration',0) or 0), color='gray', ls='--', lw=0.8)
            plt.ylabel(mk);
            if idx==1: plt.title('Learning Curves (NDCG)')
            if idx==len(metric_keys): plt.xlabel('Iteration')
            plt.legend(fontsize=8)
        plt.tight_layout(); lc_png = MODEL_PATH.parent/'learning_curve_ndcg.png'; plt.savefig(lc_png,dpi=150); plt.close(); logger.info("Saved learning curves: %s", lc_csv)
    except Exception as e: logger.warning("Curve plot failed: %s", e)

def _export_feature_importance(booster: lgb.Booster) -> None:
    fi_gain = booster.feature_importance('gain'); fi_split = booster.feature_importance('split'); fi_df = pd.DataFrame({'feature': booster.feature_name(),'gain': fi_gain,'split': fi_split}).sort_values('gain',ascending=False)
    fi_df.to_csv(FI_CSV,index=False)
    try:
        top = fi_df.head(30); plt.figure(figsize=(8,max(4,len(top)*0.25)))
        plt.barh(top['feature'][::-1], top['gain'][::-1], color='#2166ac'); plt.xlabel('Gain'); plt.title('Feature Importance (LambdaRank)'); plt.tight_layout(); plt.savefig(FI_PNG,dpi=150); plt.close()
    except Exception as e: logger.warning("FI plot failed: %s", e)

def _add_prediction_exports(cfg: RankTrainConfig, metrics: Dict[str,Any]) -> None:
    if not cfg.export_predictions: return
    try:
        from predict import predict as _predict
        latest_df = _predict(top_n=None, snapshot='latest'); last_lbl_df = _predict(top_n=None, snapshot='last_labeled')
        pred_csv = MODEL_PATH.parent/'predictions.csv'; last_year_csv = MODEL_PATH.parent/'last_year_predictions.csv'
        latest_df.to_csv(pred_csv,index=False); last_lbl_df.to_csv(last_year_csv,index=False)
        metrics['predictions_csv']=str(pred_csv); metrics['last_year_predictions_csv']=str(last_year_csv)
    except Exception as e: logger.warning("Prediction export failed: %s", e)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train_model_learn_to_rank(tune: bool = False, tune_param_grid: Optional[Dict[str,Any]] = None, max_combinations: int = 200, *, use_full_panel: Optional[bool] = None, train_window_years: Optional[int] = None, recency_lambda: Optional[float] = None, min_cross_section: Optional[int] = None, valid_months: Optional[int] = None, test_months: Optional[int] = None, forward_gap_days: Optional[int] = None, num_boost_round: int = 800, params: Optional[Dict[str,Any]] = None, early_stopping_rounds: int = 20, primary_k: Optional[int] = None, valid_date_step: Optional[int] = None, test_date_step: Optional[int] = None) -> dict:
    cfg = RankTrainConfig()
    if use_full_panel is not None: cfg.use_full_panel = bool(use_full_panel)
    if train_window_years is not None: cfg.train_window_years = int(train_window_years)
    if recency_lambda is not None: cfg.recency_lambda = float(recency_lambda)
    if min_cross_section is not None: cfg.min_cross_section = int(min_cross_section)
    if valid_months is not None: cfg.valid_months = int(valid_months)
    if test_months is not None: cfg.test_months = int(test_months)
    if forward_gap_days is not None: cfg.forward_gap_days = int(forward_gap_days)
    if valid_date_step is not None: cfg.valid_date_step = int(valid_date_step)
    if test_date_step is not None: cfg.test_date_step = int(test_date_step)

    bundle = _prepare_bundle(cfg)

    if tune:
        logger.info("Hyperparameter tuning (max_evals=%d)...", int(max_combinations))
        t_res = tune_hyperparameters(param_grid=tune_param_grid, max_evals=int(max_combinations), seed=cfg.seed, cfg=cfg, bundle=bundle)
        best_params = t_res.get('best_params', {})
        if best_params:
            cfg.params = {**(cfg.params or {}), **best_params}
            logger.info("Tuning best params merged: %s", best_params)
        else:
            logger.warning("Tuning produced no valid trials; keeping existing params")

    if params: cfg.params = {**(cfg.params or {}), **params}
    lgb_params = {**(cfg.params or {})}; lgb_params['objective'] = 'lambdarank'; lgb_params['metric'] = 'ndcg'
    eval_at = list(lgb_params.get('eval_at', [20, 10, 5]))
    if primary_k is None and eval_at: primary_k = int(eval_at[0])
    if primary_k is not None:
        rest = [int(k) for k in eval_at if int(k)!=int(primary_k)]
        lgb_params['eval_at'] = [int(primary_k)] + rest
    max_rel = int(max(bundle.y_rel_train.max(), bundle.y_rel_valid.max(), (bundle.y_rel_test.max() if len(bundle.y_rel_test) else 0)))
    lgb_params.setdefault('label_gain', [int(2**i-1) for i in range(max_rel+1)])


    booster, evals_result = _train_booster(cfg,bundle,params=lgb_params,num_boost_round=num_boost_round,early_stopping_rounds=early_stopping_rounds)
    _plot_learning_curves(evals_result, booster)

    pred_valid = booster.predict(bundle.X_valid, num_iteration=booster.best_iteration)
    try: spearman_valid = float(getattr(spearmanr(bundle.y_cont_valid.values,pred_valid,nan_policy='omit'),'correlation', np.nan))
    except Exception: spearman_valid = float('nan')
    if len(bundle.X_test):
        pred_test = booster.predict(bundle.X_test, num_iteration=booster.best_iteration)
        try: spearman_test = float(getattr(spearmanr(bundle.y_cont_test.values,pred_test,nan_policy='omit'),'correlation', np.nan))
        except Exception: spearman_test = float('nan')
    else:
        pred_test = None; spearman_test = float('nan')

    ndcg_metrics = _extract_ndcg_metrics(evals_result, booster)
    metrics: Dict[str,Any] = {
        **ndcg_metrics,
        'spearman_valid': spearman_valid,
        'spearman_test': spearman_test,
        'best_iteration': int(getattr(booster,'best_iteration',-1) or -1),
        'train_rows': bundle.meta['train_rows'],'valid_rows': bundle.meta['valid_rows'],'test_rows': bundle.meta['test_rows'],
        'n_groups_train': bundle.meta.get('n_groups_train'),'n_groups_valid': bundle.meta.get('n_groups_valid'),'n_groups_test': bundle.meta.get('n_groups_test'),
        'avg_group_size_train': bundle.meta.get('avg_group_size_train'),'target_clip': cfg.target_clip,'n_relevance_bins': cfg.n_relevance_bins,'objective': 'lambdarank','metric': 'ndcg'
    }
    if cfg.manual_test_ndcg and len(bundle.X_test) and pred_test is not None:
        metrics.update(_manual_ndcg(bundle.y_rel_test.to_numpy(), np.asarray(pred_test), bundle.group_test, list(cfg.manual_test_cutoffs)))

    metrics.update({
        'label_dist_train': dict(pd.Series(bundle.y_rel_train).value_counts().sort_index()),
        'label_dist_valid': dict(pd.Series(bundle.y_rel_valid).value_counts().sort_index()),
        'label_dist_test': dict(pd.Series(bundle.y_rel_test).value_counts().sort_index()),
        'group_stats_train': _compute_group_stats(bundle.group_train),
        'group_stats_valid': _compute_group_stats(bundle.group_valid),
        'group_stats_test': _compute_group_stats(bundle.group_test),
        'config': {k: v for k,v in asdict(cfg).items() if k!='params'}
    })

    ensure_dir(MODEL_PATH); booster.save_model(str(MODEL_PATH))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding='utf-8')
    (MODEL_PATH.parent/'config.json').write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding='utf-8')
    FEATURE_META.write_text(json.dumps({'features': bundle.meta['feature_cols'], 'objective': 'lambdarank', 'target': TARGET_COL}, indent=2), encoding='utf-8')
    _export_feature_importance(booster)
    _add_prediction_exports(cfg, metrics)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding='utf-8')  # update with prediction paths

    return metrics

__all__ = [
    'train_model_learn_to_rank','tune_hyperparameters','RankTrainConfig','TARGET_COL'
]
