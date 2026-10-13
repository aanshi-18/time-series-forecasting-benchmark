#!/usr/bin/env python3
"""Compare forecasting models on champagne sales and bike-sharing demand.

Run:
    python run_walkforward.py --data-dir data --out-dir results
    python run_walkforward.py --data-dir data --out-dir results --resume

Outputs: saved predictions, summary.txt and metrics_long.csv.
Neural models use separate fits for each horizon and multiple random seeds.
Bike windows advance by records; missing hours are not imputed.

Evaluation caveat: folds are fitted through the training boundary, which is
later than the origin for the first h-1 multi-step test targets. Parameters
can therefore use post-origin observations, though test targets are excluded
from fitting. Bike ARIMAX uses realized future covariates.
"""
import argparse, calendar, datetime, json, os, platform, sys, time, warnings, glob
import numpy as np
import pandas as pd

BIKE_COVARS = ["holiday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
DL_MODELS = ["LSTM", "GRU", "CNN-LSTM", "Transformer", "Transformer+PE"]
CLASSICAL = ["SARIMA", "ARIMAX"]
NAIVE = ["SeasonalNaive"]
ALL_MODELS = NAIVE + CLASSICAL + DL_MODELS
D_MODEL, N_HEADS, KEY_DIM, LR = 32, 4, 16, 1e-3
FOURIER_K = 3

CFG = {
    "champagne": dict(test=12, horizons=[1, 3, 6, 12], unit="months", season=12, naive_lag=12,
                      L={m: 12 for m in DL_MODELS}, conv_padding="same",
                      batch=4, max_epochs=300, patience=20, val_frac=0.20),
    "bike": dict(test=168, horizons=[1, 6, 24, 48], unit="hours", season=24, naive_lag=168,
                 L={"LSTM": 24, "GRU": 24, "CNN-LSTM": 48, "Transformer": 48, "Transformer+PE": 48},
                 conv_padding="valid", batch=128, max_epochs=60, patience=5, val_frac=0.10),
}


# Data
def find_file(data_dir, patterns):
    for pat in patterns:
        hit = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hit:
            return hit[0]
    raise FileNotFoundError(f"none of {patterns} found in {os.path.abspath(data_dir)}")


def load_champagne(d):
    df = pd.read_csv(find_file(d, ["*champagne*.csv"]))
    df.columns = ["month", "sales"]
    df = df[df["month"].astype(str).str.match(r"^\d{4}-\d{2}$")].copy()      # drops footer / blank rows
    df["sales"] = df["sales"].astype(float)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    assert df["sales"].notna().all()
    return df.reset_index(drop=True)


def load_bike(d):
    df = pd.read_csv(find_file(d, ["hour.csv", "hour_csv.zip", "*hour*.csv", "*hour*.zip"]))
    df = df.sort_values("instant").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    assert df[["cnt"] + BIKE_COVARS].notna().all().all()
    return df


def make_folds(ds, n, champ_folds, bike_folds):
    T = CFG[ds]["test"]
    if ds == "champagne":
        starts = [n - T * k for k in range(champ_folds, 0, -1)]
    elif bike_folds == 1:
        starts = [n - T]
    else:
        starts = np.linspace(0.55 * n, n - T, bike_folds).round().astype(int).tolist()
    starts = [int(s) for s in starts]
    assert starts[-1] + T <= n and all(a + T <= b for a, b in zip(starts, starts[1:]))
    assert starts[0] >= max(CFG[ds]["L"].values()) + max(CFG[ds]["horizons"]) + 15
    return starts


# Metrics and tests
def metrics(y, p):
    e = y - p
    return dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                mape=float(np.mean(np.abs(e) / np.abs(y)) * 100),
                smape=float(np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p))) * 100))


def dm_test(e1, e2, h, power=1):
    """Diebold-Mariano-type test of equal predictive accuracy (loss |e|^power).
    Newey-West (Bartlett) long-run variance, bandwidth max(h-1, floor(n^(1/3))); t(n-1) p-value (two-sided).
    Returns (statistic, p). statistic < 0  <=>  model 1 has the lower loss."""
    from scipy import stats
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d); dbar = d.mean(); dc = d - dbar
    lag = min(max(h - 1, int(n ** (1 / 3))), n - 2)
    var = float(np.dot(dc, dc) / n)
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * float(np.dot(dc[k:], dc[:-k]) / n)
    var /= n
    if not var > 0:
        return float("nan"), float("nan")
    stat = dbar / np.sqrt(var)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))


# Windowing (direct h-step samples)
def make_windows(data, L, h, n_train, n_end):
    """Sample for target t = rows [t-h-L+1, t-h] (origin t-h) -> y_t (column 0).
    train targets: t in [L+h-1, n_train) ; test targets: t in [n_train, n_end). data has n_end rows."""
    sw = np.lib.stride_tricks.sliding_window_view(data, L, axis=0)        # (n-L+1, F, L)
    def build(ts):
        idx = ts - h - L + 1
        return np.ascontiguousarray(sw[idx].transpose(0, 2, 1)).astype("float32"), data[ts, 0].astype("float32")
    Xtr, ytr = build(np.arange(L + h - 1, n_train))
    Xte, yte = build(np.arange(n_train, n_end))
    return Xtr, ytr, Xte, yte


# Deep models
_ADDPOS = None


def get_addpos():
    global _ADDPOS
    if _ADDPOS is None:
        from tensorflow.keras import layers

        class AddPos(layers.Layer):
            """x + learned positional embedding (L x d), broadcast over the batch."""
            def __init__(self, L, d, **kw):
                super().__init__(**kw); self.L, self.d = L, d
            def build(self, input_shape):
                self.pe = self.add_weight(name="pos_embedding", shape=(self.L, self.d), initializer="random_normal",
                                          trainable=True)
            def call(self, x):
                return x + self.pe
        _ADDPOS = AddPos
    return _ADDPOS


def build_keras(name, L, F, conv_padding):
    from tensorflow.keras import layers, models, optimizers
    inp = layers.Input((L, F))
    if name == "LSTM":
        x = layers.Dropout(0.2)(layers.LSTM(64)(inp))
    elif name == "GRU":
        x = layers.Dropout(0.2)(layers.GRU(64)(inp))
    elif name == "CNN-LSTM":
        x = layers.Conv1D(32, 3, activation="relu", padding=conv_padding)(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, activation="relu", padding=conv_padding)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(layers.LSTM(64)(x))
    elif name in ("Transformer", "Transformer+PE"):
        x = layers.Dense(D_MODEL)(inp)                              # input projection (F may be 1)
        if name == "Transformer+PE":                                # Positional signal is used only in this variant.
            x = get_addpos()(L, D_MODEL)(x)
        a = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=KEY_DIM)(x, x)
        x = layers.LayerNormalization()(layers.Add()([x, a]))
        f = layers.Dense(D_MODEL)(layers.Dense(64, activation="relu")(x))
        x = layers.LayerNormalization()(layers.Add()([x, f]))
        x = layers.Dropout(0.2)(layers.GlobalAveragePooling1D()(x))
    else:
        raise ValueError(name)
    m = models.Model(inp, layers.Dense(1)(x))
    m.compile(optimizer=optimizers.Adam(LR), loss="mse")
    return m


def deep_fold(ds, model, h, start, mat, seeds, dry):
    cfg = CFG[ds]; L = cfg["L"][model]; end = start + cfg["test"]
    data = mat[:end]                                                # nothing beyond the test block is touched
    mn = data[:start].min(0); rg = data[:start].max(0) - mn; rg[rg == 0] = 1.0   # Scale with pre-test observations.
    sc = (data - mn) / rg
    Xtr, ytr, Xte, _ = make_windows(sc, L, h, start, end)
    preds, epochs, nparams = [], [], None
    for sd in seeds:
        if dry:
            from sklearn.linear_model import Ridge
            p = Ridge(alpha=1.0).fit(Xtr.reshape(len(Xtr), -1), ytr).predict(Xte.reshape(len(Xte), -1)); ep = None
        else:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(sd)
            tf.keras.backend.clear_session()
            m = build_keras(model, L, mat.shape[1], cfg["conv_padding"])
            nv = max(1, int(len(Xtr) * cfg["val_frac"]))             # chronological tail of the training windows
            es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                                  restore_best_weights=True)
            hist = m.fit(Xtr[:-nv], ytr[:-nv], validation_data=(Xtr[-nv:], ytr[-nv:]), epochs=cfg["max_epochs"],
                         batch_size=cfg["batch"], callbacks=[es], verbose=0, shuffle=True)
            p = m.predict(Xte, verbose=0, batch_size=512).ravel(); ep = len(hist.history["loss"])
            nparams = int(m.count_params())
        preds.append((p * rg[0] + mn[0]).astype(float)); epochs.append(ep)
    return preds, dict(n_windows=int(len(Xtr)), epochs=epochs), nparams


# Naive / classical models
def naive_fold(y, start, T, horizons, lag):
    assert lag >= max(horizons)
    return {h: np.array([y[t - lag] for t in range(start, start + T)], float) for h in horizons}


def fourier(n, period=12, K=FOURIER_K):
    t = np.arange(n)
    return np.column_stack([f(2 * np.pi * k * t / period) for k in range(1, K + 1) for f in (np.sin, np.cos)])


def rolling_hstep(predict_fn, n, ntr, T, horizons):
    """predict_fn(s, e) -> forecasts for positions s..e (inclusive) issued at origin s-1 (dynamic multi-step).
    For each target t in [ntr, ntr+T) and horizon h the forecast comes from origin t-h. One call per origin."""
    hmax = max(horizons); targets = np.arange(ntr, ntr + T)
    cache = {}
    for o in sorted({int(t - h) for t in targets for h in horizons}):
        m = min(hmax, n - 1 - o)
        cache[o] = np.asarray(predict_fn(o + 1, o + m), dtype=float)
    return {h: np.array([cache[int(t) - h][h - 1] for t in targets]) for h in horizons}


def classical_fold(kind, ds, y, exog, start, T, horizons, arima_train, dry):
    lo = max(0, start - arima_train) if (ds == "bike" and arima_train) else 0
    yy = y[lo:start + T]; ee = None if exog is None else exog[lo:start + T]; ntr = start - lo
    if dry:
        return {h: np.array([yy[t - h] for t in range(ntr, ntr + T)], float) for h in horizons}, True, ntr
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    s = CFG[ds]["season"]
    order, sorder = ((1, 1, 1), (0, 1, 1, s)) if kind == "SARIMA" else ((2, 1, 2), (0, 0, 0, 0))
    mod = SARIMAX(yy[:ntr], exog=None if ee is None else ee[:ntr], order=order, seasonal_order=sorder,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False, maxiter=200)                          # Estimate parameters on the pre-test sample.
    ret = getattr(res, "mle_retvals", None)
    converged = bool(ret.get("converged", True)) if isinstance(ret, dict) else True
    full = res.apply(yy, exog=ee)                                   # Update filter states without refitting parameters.
    out = rolling_hstep(lambda a, b: full.predict(start=a, end=b, dynamic=0), len(yy), ntr, T, horizons)
    return out, converged, ntr


# Driver
def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def lib_version(name):
    try:
        return __import__(name).__version__
    except Exception:
        return "n/a"


def run(args, champ, bike):
    os.makedirs(args.out_dir, exist_ok=True)
    rpath = os.path.join(args.out_dir, "results.json")
    seeds = [42 + i for i in range(args.seeds)]
    series = {}
    yc = champ["sales"].to_numpy(float)
    series["champagne"] = dict(y=yc, mat=yc[:, None], ex=fourier(len(yc)))
    yb = bike["cnt"].to_numpy(float)
    series["bike"] = dict(y=yb, mat=bike[["cnt"] + BIKE_COVARS].to_numpy(float), ex=bike[BIKE_COVARS].to_numpy(float))
    folds = {ds: make_folds(ds, len(series[ds]["y"]), args.champ_folds, args.bike_folds) for ds in series}
    sig = json.loads(json.dumps(dict(
        seeds=seeds, folds=folds, arima_train=args.arima_train, dry_run=args.dry_run, quick=args.quick,
        d_model=D_MODEL, heads=N_HEADS, key_dim=KEY_DIM, lr=LR, fourier_k=FOURIER_K,
        cfg={ds: {k: v for k, v in CFG[ds].items()} for ds in CFG})))
    if os.path.exists(rpath):
        if not args.resume:
            sys.exit(f"{rpath} exists. Use --resume to continue it, or choose another --out-dir.")
        R = json.load(open(rpath))
        if R["config"] != sig:
            sys.exit("Settings differ from the stored run (seeds / folds / epochs / arima-train / dry-run / quick). "
                     "Refusing to mix results; use a fresh --out-dir.")
    else:
        R = dict(config=sig, results={}, params={}, meta={})
    try:
        import tensorflow as tf
        gpu = bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        gpu = False
    R["meta"].update(started=datetime.datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
                     numpy=np.__version__, pandas=pd.__version__, tensorflow=lib_version("tensorflow"),
                     statsmodels=lib_version("statsmodels"), gpu=gpu, dry_run=bool(args.dry_run), quick=bool(args.quick))
    print(f"TensorFlow {R['meta']['tensorflow']} | statsmodels {R['meta']['statsmodels']} | GPU: {gpu}")
    if not args.dry_run and not gpu:
        print("WARNING: no GPU detected - bike deep models will be slow (Colab: Runtime > Change runtime type > GPU)")

    datasets = ["champagne", "bike"] if args.dataset == "both" else [args.dataset]
    models = ALL_MODELS if not args.models else [m for m in args.models.split(",")]
    for m in models:
        assert m in ALL_MODELS, f"unknown model {m}; choose from {ALL_MODELS}"
    todo = 0
    for ds in datasets:
        for m in models:
            for h in CFG[ds]["horizons"]:
                for k in range(len(folds[ds])):
                    if str(k) not in R["results"].get(ds, {}).get(m, {}).get(str(h), {}):
                        todo += len(seeds) if m in DL_MODELS else 0
    done_fits, t_start = 0, time.time()

    def store(ds, m, h, k, rec):
        R["results"].setdefault(ds, {}).setdefault(m, {}).setdefault(str(h), {})[str(k)] = rec
        save_json(R, rpath)

    for ds in datasets:
        S, cfg, T = series[ds], CFG[ds], CFG[ds]["test"]
        hs = cfg["horizons"]; y = S["y"]
        for m in models:
            have = lambda h, k: str(k) in R["results"].get(ds, {}).get(m, {}).get(str(h), {})
            if m in NAIVE + CLASSICAL:
                for k, st in enumerate(folds[ds]):
                    if all(have(h, k) for h in hs):
                        continue
                    t0 = time.time(); extra = {}
                    if m in NAIVE:
                        out = naive_fold(y, st, T, hs, cfg["naive_lag"])
                    else:
                        out, conv, ntr = classical_fold(m, ds, y, None if m == "SARIMA" else S["ex"], st, T, hs,
                                                        args.arima_train, args.dry_run)
                        extra = dict(converged=conv, n_train=int(ntr))
                    for h in hs:
                        store(ds, m, h, k, dict(seeds=[None], preds=[np.round(out[h], 4).tolist()],
                                                metrics=[metrics(y[st:st + T], out[h])], **extra))
                    print(f"[{ds}] {m} fold {k + 1}/{len(folds[ds])} done in {time.time() - t0:.0f}s"
                          + (f" (converged={extra['converged']})" if extra else ""), flush=True)
            else:
                for h in hs:
                    for k, st in enumerate(folds[ds]):
                        if have(h, k):
                            continue
                        t0 = time.time()
                        preds, info, npar = deep_fold(ds, m, h, st, S["mat"], seeds, args.dry_run)
                        yt = y[st:st + T]
                        store(ds, m, h, k, dict(seeds=seeds, preds=[np.round(p, 4).tolist() for p in preds],
                                                metrics=[metrics(yt, p) for p in preds], **info))
                        if npar is not None:
                            R["params"].setdefault(ds, {})[m] = npar
                        done_fits += len(seeds)
                        el = time.time() - t_start
                        eta = el / done_fits * (todo - done_fits) if done_fits else 0
                        print(f"[{ds}] {m} h={h} fold {k + 1}/{len(folds[ds])} ({len(seeds)} seeds) "
                              f"{time.time() - t0:.0f}s | fits {done_fits}/{todo} | elapsed {el / 60:.0f} min, "
                              f"rough ETA {eta / 60:.0f} min", flush=True)
    R["meta"]["last_update"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_json(R, rpath)
    return R


# Summary
def pm(m, s, d=2):
    return f"{m:.{d}f} ± {s:.{d}f}" if s > 0 else f"{m:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def data_facts(champ, bike):
    mon = champ.groupby(champ["month"].dt.month)["sales"].mean()
    hr = bike.groupby("hr")["cnt"].mean().sort_values(ascending=False)
    yr = bike.groupby(bike["ts"].dt.year)["cnt"].mean()
    span = int((bike["ts"].iloc[-1] - bike["ts"].iloc[0]) / pd.Timedelta(hours=1)) + 1
    return [
        f"champagne: {len(champ)} months, {champ['month'].iloc[0]} .. {champ['month'].iloc[-1]}; "
        f"min {champ['sales'].min():,.0f}, max {champ['sales'].max():,.0f}",
        f"champagne monthly means: highest {calendar.month_name[int(mon.idxmax())]} ({mon.max():,.0f}), "
        f"lowest {calendar.month_name[int(mon.idxmin())]} ({mon.min():,.0f})",
        f"bike: {len(bike):,} records, {bike['ts'].iloc[0].date()} .. {bike['ts'].iloc[-1].date()}; "
        f"calendar span {span:,} h => {span - len(bike)} hours missing",
        f"bike top-3 mean-demand hours: " + ", ".join(f"{int(i)}:00 ({v:.1f})" for i, v in hr.head(3).items()),
        "bike mean hourly demand by year: " + ", ".join(f"{int(i)}: {v:.1f}" for i, v in yr.items()),
    ]


def write_summary(R, champ, bike, out_dir):
    cfgs = R["config"]; meta = R["meta"]; res = R["results"]
    series_y = {"champagne": champ["sales"].to_numpy(float), "bike": bike["cnt"].to_numpy(float)}
    L = []
    if cfgs["dry_run"]:
        L += ["#" * 80, "# DRY RUN - trivial stand-in models (ridge / persistence). NOT RESULTS. DO NOT REPORT.", "#" * 80]
    if cfgs["quick"]:
        L += ["# QUICK RUN (reduced epochs/folds/seeds): smoke test only - NOT FOR THE PAPER."]
    L += ["WALK-FORWARD BENCHMARK SUMMARY", "",
          f"run: started {meta.get('started')}, last update {meta.get('last_update')}",
          f"software: python {meta.get('python')}, numpy {meta.get('numpy')}, pandas {meta.get('pandas')}, "
          f"tensorflow {meta.get('tensorflow')}, statsmodels {meta.get('statsmodels')}, GPU {meta.get('gpu')}",
          f"seeds (deep models): {cfgs['seeds']} | arima training block (bike): {cfgs['arima_train']} records | "
          f"fourier K (champagne ARIMAX): {cfgs['fourier_k']}",
          f"transformer: d_model {cfgs['d_model']}, heads {cfgs['heads']}, key_dim {cfgs['key_dim']}; Adam lr {cfgs['lr']}",
          "", "DATA FACTS (computed from the supplied files)"] + ["- " + s for s in data_facts(champ, bike)]
    csv = ["dataset,model,horizon,level,fold,seed,mae,rmse,mape,smape"]

    L += ["", "DEEP-MODEL TRAINING SETTINGS"]
    for ds in ("champagne", "bike"):
        c = cfgs["cfg"][ds]
        L.append(f"- {ds}: window L {c['L']}, conv padding {c['conv_padding']}, batch {c['batch']}, "
                 f"max epochs {c['max_epochs']}, patience {c['patience']}, validation fraction {c['val_frac']}")
    L += ["", "TRAINABLE PARAMETERS (counted by TensorFlow)"]
    pr = [[m] + [f"{R['params'].get(ds, {}).get(m, 'n/a'):,}" if m in R["params"].get(ds, {}) else "n/a"
                 for ds in ("champagne", "bike")] for m in DL_MODELS]
    L += md_table(["model", "champagne (F=1)", "bike (F=8)"], pr)

    for ds in ("champagne", "bike"):
        if ds not in res:
            continue
        c = cfgs["cfg"][ds]; T = c["test"]; hs = c["horizons"]; starts = cfgs["folds"][ds]; nf = len(starts)
        y = series_y[ds]
        L += ["", "=" * 80, f"DATASET: {ds}  ({c['unit']}), {nf} walk-forward folds, test block {T} each", "=" * 80]
        for k, st in enumerate(starts):
            if ds == "bike":
                a, b = bike["ts"].iloc[st], bike["ts"].iloc[st + T - 1]
                L.append(f"fold {k + 1}: training records 0..{st - 1} ({st:,}), test records {st}..{st + T - 1} "
                         f"({a} .. {b})")
            else:
                a, b = champ["month"].iloc[st], champ["month"].iloc[st + T - 1]
                L.append(f"fold {k + 1}: training months 0..{st - 1} ({st}), test months {st}..{st + T - 1} ({a} .. {b})")
        ypool = np.concatenate([y[st:st + T] for st in starts])
        L.append(f"pooled test targets per model/horizon: {len(ypool)}")

        done = {}      # (model, h) -> dict
        for m in ALL_MODELS:
            for h in hs:
                recs = [res.get(ds, {}).get(m, {}).get(str(h), {}).get(str(k)) for k in range(nf)]
                if any(r is None for r in recs):
                    continue
                ns = len(recs[0]["preds"])
                P = np.array([np.concatenate([np.array(recs[k]["preds"][s]) for k in range(nf)]) for s in range(ns)])
                pooled = [metrics(ypool, P[s]) for s in range(ns)]
                fold_mae = np.array([np.mean([metrics(y[starts[k]:starts[k] + T], np.array(recs[k]["preds"][s]))["mae"]
                                              for s in range(ns)]) for k in range(nf)])
                done[(m, h)] = dict(P=P, pooled=pooled, fold_mae=fold_mae, recs=recs, ns=ns)
                for s in range(ns):
                    sd = recs[0]["seeds"][s]
                    csv.append(f"{ds},{m},{h},pooled,,{sd if sd is not None else ''},"
                               + ",".join(f"{pooled[s][q]:.6f}" for q in ("mae", "rmse", "mape", "smape")))
                for k in range(nf):
                    for s in range(ns):
                        mm = recs[k]["metrics"][s]; sd = recs[k]["seeds"][s]
                        csv.append(f"{ds},{m},{h},fold,{k + 1},{sd if sd is not None else ''},"
                                   + ",".join(f"{mm[q]:.6f}" for q in ("mae", "rmse", "mape", "smape")))
        missing = [(m, h) for m in ALL_MODELS for h in hs if (m, h) not in done]
        if missing:
            L.append(f"INCOMPLETE (not in this summary): {sorted(set(m for m, _ in missing))}")
        models = [m for m in ALL_MODELS if any((m, h) in done for h in hs)]

        def stat(m, h, q):
            v = np.array([p[q] for p in done[(m, h)]["pooled"]])
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

        # training-set sizes / early stopping
        L += ["", "TRAINING WINDOWS PER FOLD AND EARLY STOPPING (deep models)"]
        r = []
        for m in DL_MODELS:
            if (m, hs[0]) in done:
                nw1 = [done[(m, hs[0])]["recs"][k]["n_windows"] for k in range(nf)]
                nwl = [done[(m, hs[-1])]["recs"][k]["n_windows"] for k in range(nf)] if (m, hs[-1]) in done else []
                ep = [e for h in hs if (m, h) in done for rc in done[(m, h)]["recs"] for e in rc["epochs"] if e is not None]
                hit = sum(1 for e in ep if e >= c["max_epochs"])
                r.append([m, f"{nw1}", f"{nwl}", f"{np.mean(ep):.1f}" if ep else "n/a", f"{hit}/{len(ep)}" if ep else "n/a"])
        if r:
            L += md_table(["model", f"windows/fold (h={hs[0]})", f"windows/fold (h={hs[-1]})", "mean epochs",
                           "runs at max epochs"], r)
        for m in CLASSICAL:
            if (m, hs[0]) in done:
                cv = [rc.get("converged") for rc in done[(m, hs[0])]["recs"]]
                L.append(f"{m}: optimiser converged in folds {cv}; training-block length {done[(m, hs[0])]['recs'][0].get('n_train')}")

        for h in hs:
            L += ["", f"TABLE: {ds}, horizon t+{h} - pooled over all folds (mean ± sd over seeds; no ± = deterministic)"]
            rows = []
            for m in models:
                if (m, h) in done:
                    fm = done[(m, h)]["fold_mae"]
                    rows.append([m] + [pm(*stat(m, h, q)) for q in ("mae", "rmse", "mape", "smape")]
                                + [f"{fm.std(ddof=1):.2f}"])
            L += md_table(["model", "MAE", "RMSE", "MAPE %", "sMAPE %", "sd of per-fold MAE"], rows)
        L += ["", f"TABLE: {ds}, MAE by horizon (pooled; mean ± sd over seeds)"]
        L += md_table(["model"] + [f"t+{h}" for h in hs],
                      [[m] + [pm(*stat(m, h, "mae")) if (m, h) in done else "n/a" for h in hs] for m in models])
        for h in hs:
            L += ["", f"TABLE: {ds}, horizon t+{h} - MAE per fold (mean over seeds)"]
            L += md_table(["model"] + [f"fold {k + 1}" for k in range(nf)],
                          [[m] + [f"{v:.2f}" for v in done[(m, h)]["fold_mae"]] for m in models if (m, h) in done])

        # Diebold-Mariano: best model vs every other
        L += ["", f"DIEBOLD-MARIANO-TYPE TESTS ({ds}): lowest-mean-MAE model vs each other model.",
              "Forecasts of deep models = mean over seeds (seed-ensemble). Errors pooled over folds. "
              "stat<0 means the best model has lower loss. No multiplicity correction."]
        for h in hs:
            ms = [m for m in models if (m, h) in done]
            best = min(ms, key=lambda m: stat(m, h, "mae")[0])
            eb = ypool - done[(best, h)]["P"].mean(0)
            rows = []
            for m in ms:
                if m == best:
                    continue
                e2 = ypool - done[(m, h)]["P"].mean(0)
                s1, p1 = dm_test(eb, e2, h, 1); s2, p2 = dm_test(eb, e2, h, 2)
                rows.append([m, f"{s1:.2f}", f"{p1:.3f}", f"{s2:.2f}", f"{p2:.3f}"])
            L += ["", f"t+{h}: best by mean MAE = {best}"]
            L += md_table(["vs", "stat (abs err)", "p (abs err)", "stat (sq err)", "p (sq err)"], rows)

    path = os.path.join(out_dir, "summary.txt")
    open(path, "w").write("\n".join(L) + "\n")
    open(os.path.join(out_dir, "metrics_long.csv"), "w").write("\n".join(csv) + "\n")
    print(f"\nWrote {path} and metrics_long.csv - send these two files back.")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--dataset", choices=["both", "champagne", "bike"], default="both")
    ap.add_argument("--models", default="", help="comma-separated subset, e.g. LSTM,GRU")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--champ-folds", type=int, default=4)
    ap.add_argument("--bike-folds", type=int, default=5)
    ap.add_argument("--arima-train", type=int, default=720, help="training-block length (records) for bike SARIMA/ARIMAX")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if args.quick:                                       # Use a separate output folder for the smoke test.
        args.out_dir = args.out_dir.rstrip("/") + "_quick"
        args.seeds, args.champ_folds, args.bike_folds = 1, 1, 1
        for ds in CFG:
            CFG[ds]["max_epochs"] = 3
    champ, bike = load_champagne(args.data_dir), load_bike(args.data_dir)
    print(f"champagne: {len(champ)} months | bike: {len(bike):,} records | out-dir: {args.out_dir}")
    if args.summary_only:
        R = json.load(open(os.path.join(args.out_dir, "results.json")))
    else:
        R = run(args, champ, bike)
    lines = write_summary(R, champ, bike, args.out_dir)
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
