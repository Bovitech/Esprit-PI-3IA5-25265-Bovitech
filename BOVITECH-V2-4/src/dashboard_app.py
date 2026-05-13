"""
Bovitech dashboard: metrics, prediction CSV explorer, accelerometer simulator.

Run from project root:
  streamlit run src/dashboard_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import numpy as np

# Project root = parent of src/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from predict_core import load_model_bundle, predict_from_immu  # noqa: E402

# Libellés par défaut (1–7) si behavior_map.json absent — à aligner avec ton projet
_DEFAULT_BEHAVIOR_LABELS: dict[int, str] = {
    1: "Walking",
    2: "Standing",
    3: "Feeding head up",
    4: "Feeding head down",
    5: "Licking",
    6: "Drinking",
    7: "Lying",
}


def _merged_behavior_labels(bmap: dict | None) -> dict[int, str]:
    out = dict(_DEFAULT_BEHAVIOR_LABELS)
    if bmap:
        out.update(bmap)
    return out


def _enrich_pred_columns(df: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    """Ajoute *_name pour pred_behavior et pred_behavior_smooth."""
    out = df.copy()
    for col in ("pred_behavior_smooth", "pred_behavior"):
        if col not in out.columns:
            continue
        name_col = col + "_name"
        def _lbl(v: object) -> str:
            if pd.isna(v):
                return ""
            try:
                k = int(round(float(v)))
                return labels.get(k, str(k))
            except (TypeError, ValueError):
                return str(v)

        out[name_col] = out[col].apply(_lbl)
    return out


def _percent_time_by_class(series: pd.Series, labels: dict[int, str]) -> pd.DataFrame:
    """Pourcentage du nombre de lignes (secondes) par classe."""
    s = pd.to_numeric(series, errors="coerce").dropna().astype("int64")
    if len(s) == 0:
        return pd.DataFrame(columns=["Comportement", "% du temps"])
    vc = s.value_counts(normalize=True).sort_index() * 100.0
    rows = [{"Comportement": labels.get(int(k), str(k)), "% du temps": round(float(v), 2)} for k, v in vc.items()]
    return pd.DataFrame(rows)


def _render_behavior_breakdown(df: pd.DataFrame, labels: dict[int, str], title: str) -> None:
    st.subheader(title)
    col_pred = None
    for c in ("pred_behavior_smooth", "pred_behavior"):
        if c in df.columns:
            col_pred = c
            break
    if col_pred is None:
        st.caption("Aucune colonne pred_behavior trouvée.")
        return
    pct_df = _percent_time_by_class(df[col_pred], labels)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(pct_df, use_container_width=True, hide_index=True)
    with c2:
        try:
            import plotly.express as px

            if len(pct_df) > 0:
                ymax = max(100.0, float(pct_df["% du temps"].max()) * 1.15)
                fig = px.bar(
                    pct_df,
                    x="Comportement",
                    y="% du temps",
                    title="% du temps par comportement (fichier entier)",
                    text="% du temps",
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(xaxis_tickangle=-35, yaxis_range=[0, ymax])
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            if len(pct_df) > 0:
                st.bar_chart(pct_df.set_index("Comportement")["% du temps"])

    if "ts_sec" not in df.columns:
        return
    dt = pd.to_datetime(df["ts_sec"], unit="s", utc=False)
    by_day = df.assign(_date=dt.dt.date)
    dates = sorted(by_day["_date"].dropna().unique())
    if len(dates) == 0:
        return
    st.markdown("**Par jour (date calendaire dérivée de `ts_sec`)** — ex. une journée type `0721` dans le fichier")
    for d in dates:
        sub = by_day[by_day["_date"] == d]
        p2 = _percent_time_by_class(sub[col_pred], labels)
        with st.expander(
            f"{d} — {len(sub)} s — % par comportement",
            expanded=(len(dates) == 1),
        ):
            st.dataframe(p2, use_container_width=True, hide_index=True)
            try:
                import plotly.express as px

                if len(p2) > 0:
                    fig2 = px.bar(
                        p2,
                        x="Comportement",
                        y="% du temps",
                        title=f"{d} — % du temps",
                        text="% du temps",
                    )
                    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig2.update_layout(xaxis_tickangle=-35)
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass


st.set_page_config(page_title="Bovitech — Comportement vache", layout="wide")


def _default_model_dir() -> Path:
    p = Path(r"C:\bovitech_artifacts\model")
    if p.exists():
        return p
    return _ROOT / "artifacts" / "model"


with st.sidebar:
    st.header("Configuration")
    model_dir = st.text_input("Dossier modèle (`model_dir`)", value=str(_default_model_dir()))
    sensor_root = st.text_input(
        "Racine capteurs (optionnel)",
        value=r"C:\sensor_data\sensor_data",
        help="Pour charger behavior_map.json par défaut",
    )
    behavior_map_path = st.text_input(
        "behavior_map.json (optionnel)",
        value="",
        help="Laisse vide pour essayer artifacts/model/behavior_map.json",
    )
    smooth_default = st.slider("Lissage prédictions (secondes)", 1, 15, 7)


def load_behavior_map_optional() -> dict | None:
    if behavior_map_path.strip():
        p = Path(behavior_map_path.strip())
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in raw.items()}
    for candidate in (
        Path(model_dir) / "behavior_map.json",
        _ROOT / "artifacts" / "model" / "behavior_map.json",
    ):
        if candidate.exists():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            return {int(k): str(v) for k, v in raw.items()}
    return None


def _collect_training_json_files(model_dir: str, sensor_root: str) -> list[Path]:
    """
    behavior: metadata*.json dans le dossier modèle.
    lait XGBoost (ex.): *_metrics.json souvent sous model_outputs à côté de sensor_data.
    """
    mp = Path(model_dir)
    extras: list[Path] = []
    sr = Path(sensor_root.strip()) if sensor_root.strip() else Path(".")
    if sr.exists():
        extras.append(sr.parent / "model_outputs")
    extras.append(Path(r"C:\sensor_data\model_outputs"))

    files: list[Path] = []
    for base in [mp, *extras]:
        if not base.exists():
            continue
        files.extend(base.glob("metadata*.json"))
        files.extend(base.glob("*_metrics.json"))

    seen: set[str] = set()
    out: list[Path] = []
    for p in files:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return sorted(out, key=lambda x: (x.name.lower(), str(x).lower()))


def _render_training_json_doc(meta: dict) -> None:
    """Affiche soit les métadonnées RF comportement, soit un JSON métriques lait (ex. XGBoost)."""
    if "feature_columns" in meta:
        st.json(
            {
                k: meta[k]
                for k in ("metrics", "history_seconds", "include_mag", "cows", "dates", "model_params")
                if k in meta
            }
        )
        st.caption(f"Nombre de features: {len(meta['feature_columns'])}")
    elif "features" in meta and isinstance(meta["features"], list):
        st.json(meta)
        st.caption(f"Nombre de features: {len(meta['features'])}")
    else:
        st.json(meta)


def _load_milk_feature_list_from_metrics(metrics_path: Path) -> list[str] | None:
    if not metrics_path.exists():
        return None
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        feats = m.get("features")
        if isinstance(feats, list) and all(isinstance(x, str) for x in feats):
            return feats
    except Exception:
        return None
    return None


@st.cache_resource(show_spinner=False)
def _load_milk_pipeline(model_path: str):
    import joblib

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    return joblib.load(p)


def _thi_from_temp_rh_celsius(temp_c: float, rh_pct: float) -> float:
    """
    Common dairy THI approximation using air temperature (°C) and relative humidity (%).
    THI = (1.8*T + 32) - (0.55 - 0.0055*RH) * (1.8*T - 26)
    """
    t = float(temp_c)
    rh = float(rh_pct)
    return (1.8 * t + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * t - 26.0)


@st.cache_data(show_spinner=False)
def _load_behavior_daily_df(path: str) -> pd.DataFrame:
    p = Path(path.strip())
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _try_autofill_behavior_daily(behavior_daily_csv: str, cow_id: str, date_str: str) -> dict:
    """
    Charge behavior_n / behavior_mean / behavior_std pour (cow_id, date) depuis le CSV
    généré par build_behavior_daily_features.py (prédictions IMMU agrégées par jour).

    Comparaison par **date calendaire** (évite les bugs timezone / heure).
    """
    try:
        df = _load_behavior_daily_df(behavior_daily_csv)
        if df.empty or "cow_id" not in df.columns or "date" not in df.columns:
            return {}
        cid = str(cow_id).strip().upper()
        df["_cow"] = df["cow_id"].astype(str).str.strip().str.upper()
        df["_day"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize().dt.date
        tdt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(tdt):
            return {}
        target_day = tdt.normalize().date()
        sub = df[(df["_cow"] == cid) & (df["_day"] == target_day)]
        if len(sub) < 1:
            return {}
        r = sub.iloc[0]
        out = {}
        for k in ("behavior_n", "behavior_mean", "behavior_std"):
            if k in r.index and pd.notna(r[k]):
                out[k] = float(r[k])
        return out
    except Exception:
        return {}


def _try_autofill_milk_history(sensor_root: str, cow_id: str, date_str: str) -> dict:
    """
    Compute milk_lag1 and milk_roll3_mean from local milk CSVs if available.
    Returns {} if not possible.
    """
    try:
        base = Path(sensor_root)
        milk_path = base / "main_data" / "milk" / f"{cow_id}.csv"
        if not milk_path.exists():
            return {}
        df = pd.read_csv(milk_path)
        if "timestamp" not in df.columns or "milk_weight_kg" not in df.columns:
            return {}
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df["date"] = df["datetime"].dt.floor("D")
        df["milk_weight_kg"] = pd.to_numeric(df["milk_weight_kg"], errors="coerce")
        daily = (
            df.groupby("date", as_index=False)
            .agg(milk_weight_kg=("milk_weight_kg", "mean"))
            .sort_values("date")
            .reset_index(drop=True)
        )
        target_date = pd.to_datetime(date_str, errors="coerce").floor("D")
        if pd.isna(target_date):
            return {}
        daily["milk_lag1"] = daily["milk_weight_kg"].shift(1)
        daily["milk_roll3_mean"] = daily["milk_weight_kg"].rolling(window=3, min_periods=1).mean()
        row = daily[daily["date"] == target_date]
        if len(row) != 1:
            return {}
        r = row.iloc[0]
        out = {}
        if pd.notna(r.get("milk_lag1")):
            out["milk_lag1"] = float(r["milk_lag1"])
        if pd.notna(r.get("milk_roll3_mean")):
            out["milk_roll3_mean"] = float(r["milk_roll3_mean"])
        return out
    except Exception:
        return {}


def _milk_predict_single(pipeline, features: dict, feature_cols: list[str]) -> float:
    X = pd.DataFrame([{c: features.get(c, np.nan) for c in feature_cols}])
    return float(pipeline.predict(X)[0])


def _resolve_local_artifact_path(raw: str, *, default_dir: Path) -> Path:
    """
    Streamlit text inputs sometimes end up with just a filename.
    Resolve to default_dir/filename if raw is not an existing path.
    """
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        return Path("")
    p = Path(s)
    if p.exists():
        return p
    # if user pasted only a file name, try default_dir
    cand = default_dir / p.name
    return cand


tab_metrics, tab_pred, tab_sim, tab_milk = st.tabs(
    [
        "Résultats & métriques",
        "Fichier prédictions",
        "Simulateur accéléromètre (x,y,z)",
        "Milk prediction (kg/jour)",
    ]
)

with tab_metrics:
    st.subheader("Modèle et métriques d’entraînement")
    mp = Path(model_dir)
    if not mp.exists():
        st.warning(f"Dossier modèle introuvable (comportement / importances): {mp}")
    meta_files = _collect_training_json_files(model_dir, sensor_root)
    if not meta_files:
        st.warning(
            "Aucun fichier JSON trouvé (metadata*.json ou *_metrics.json). "
            "Vérifie le dossier modèle ou model_outputs (ex. C:\\sensor_data\\model_outputs)."
        )
    else:
        choice = st.selectbox(
            "Fichier metadata / métriques",
            options=[str(p) for p in meta_files],
            format_func=lambda s: f"{Path(s).name} — {s}",
        )
        meta = json.loads(Path(choice).read_text(encoding="utf-8"))
        _render_training_json_doc(meta)

    if mp.exists():
        fi = mp / "feature_importance_multimodal.csv"
        if not fi.exists():
            fi = mp / "feature_importance_immu.csv"
        if fi.exists():
            st.subheader("Importance des variables (top 25)")
            fdf = pd.read_csv(fi).head(25)
            st.dataframe(fdf, use_container_width=True)
            try:
                import plotly.express as px

                fig = px.bar(fdf, x="importance", y="feature", orientation="h", title="Feature importance")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        cm_candidates = [mp / "confusion_matrix_multimodal.csv", mp / "confusion_matrix_immu.csv"]
        cm_path = next((p for p in cm_candidates if p.exists()), None)
        if cm_path:
            st.subheader("Matrice de confusion (CSV sauvegardé à l’entraînement)")
            cm = pd.read_csv(cm_path)
            st.dataframe(cm, use_container_width=True)

with tab_pred:
    st.subheader("Visualiser un CSV de prédictions")
    uploaded = st.file_uploader("Uploader un CSV (ts_sec, pred_behavior, …)", type=["csv"])
    pred_path = st.text_input("Ou chemin vers un CSV existant", value=str(_ROOT / "artifacts" / "predictions" / "T09_0725_pred_final.csv"))
    df_plot = None
    if uploaded is not None:
        df_plot = pd.read_csv(uploaded)
    elif Path(pred_path).exists():
        df_plot = pd.read_csv(pred_path)
    if df_plot is not None and len(df_plot) > 0:
        labels = _merged_behavior_labels(load_behavior_map_optional())
        df_view = _enrich_pred_columns(df_plot, labels)
        show_cols = [c for c in df_view.columns if c in ("ts_sec", "pred_behavior_smooth_name", "pred_behavior_name", "pred_behavior_smooth", "pred_behavior")]
        other = [c for c in df_view.columns if c not in show_cols]
        st.dataframe(df_view[show_cols + other].head(200), use_container_width=True)
        col_ts = "ts_sec" if "ts_sec" in df_plot.columns else df_plot.columns[0]
        col_pred = None
        for c in ("pred_behavior_smooth", "pred_behavior"):
            if c in df_plot.columns:
                col_pred = c
                break
        if col_pred:
            try:
                import plotly.graph_objects as go

                yv = pd.to_numeric(df_plot[col_pred], errors="coerce").dropna().astype(int)
                u = sorted(yv.unique())
                tick_vals = u
                tick_text = [labels.get(k, str(k)) for k in u]

                if col_pred + "_name" in df_view.columns:
                    hover_arr = df_view[col_pred + "_name"].tolist()
                else:
                    hover_arr = df_plot[col_pred].apply(
                        lambda x: labels.get(int(round(float(x))), str(x)) if pd.notna(x) else ""
                    ).tolist()

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_plot[col_ts],
                        y=df_plot[col_pred],
                        mode="lines",
                        name=col_pred,
                        line=dict(width=1),
                        hovertext=hover_arr,
                        hovertemplate="%{hovertext}<br>"
                        + col_ts
                        + "=%{x}<br>classe=%{y}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="Comportement prédit dans le temps (axe Y = libellés)",
                    xaxis_title=col_ts,
                    yaxis_title="Comportement",
                    yaxis=dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Graphique indisponible: {e}")
        _render_behavior_breakdown(df_plot, labels, "Répartition du temps par comportement (% des secondes)")
    else:
        st.info("Charge un CSV ou indique un chemin valide.")

with tab_sim:
    st.markdown(
        """
Saisissez des échantillons **bruts** comme dans le fichier IMMU : au minimum  
`timestamp`, `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2` (plusieurs lignes par seconde possibles).  
Ajoutez `mag_x_uT`, `mag_y_uT`, `mag_z_uT` pour un **yaw** réaliste en mode synthèse tête.  
Le pipeline agrège **par seconde** puis applique le modèle (lags + multimodal si entraîné ainsi).

**Astuce :** si le modèle utilise des lags (`history_seconds` non nul dans les métadonnées), ajoute **plusieurs secondes consécutives** (ex. `1700000000`, `1700000001`, `1700000002`…) avec des IMU réalistes ; une seule seconde reste plus difficile à classer.
"""
    )
    try:
        _, _meta_sim = load_model_bundle(Path(model_dir))
        default_mm = _meta_sim.get("notes") == "multimodal"
        meta_head_src = _meta_sim.get("head_source")
    except Exception:
        default_mm = True
        meta_head_src = None
    st.caption(
        f"Modèle détecté comme multimodal (IMMU+Head) : **{default_mm}**"
        + (f" — entraînement tête : `{meta_head_src}`" if meta_head_src else "")
    )
    head_path_txt = st.text_input(
        "Optionnel — CSV direction de tête (si rempli et fichier existant : utilisé à la place de la synthèse IMU)",
        value="",
    )
    synth_head = st.checkbox(
        "Synthétiser la tête depuis l’IMU (défaut : oui si pas de CSV ci-dessus)",
        value=not head_path_txt.strip(),
        help="Décochez seulement si vous fournissez un CSV tête valide. Sinon roll/pitch/yaw viennent de imu_head_synthesis.",
    )

    st.subheader("Données IMMU")
    st.caption("Optionnel : uploade un CSV IMMU pour remplir la table automatiquement.")
    sim_uploaded = st.file_uploader(
        "Uploader un CSV IMMU (timestamp, accel_*, mag_*)",
        type=["csv"],
        key="sim_immu_uploader",
    )
    sim_path = st.text_input(
        "Ou chemin vers un CSV IMMU existant",
        value="",
        key="sim_immu_path",
        help="Exemple: examples/real_test_T10_0725_2beh_120s.csv",
    )
    st.caption(
        "Éditez le tableau puis cliquez sur **Prédire le comportement** (bouton bleu sous la table)."
    )
    default_rows = None
    try:
        if sim_uploaded is not None:
            default_rows = pd.read_csv(sim_uploaded)
        elif sim_path.strip() and Path(sim_path.strip()).exists():
            default_rows = pd.read_csv(Path(sim_path.strip()))
    except Exception as e:
        st.warning(f"Impossible de charger le CSV IMMU: {e}")
        default_rows = None

    if default_rows is None or len(default_rows) == 0:
        default_rows = pd.DataFrame(
            {
                "timestamp": [1700000000.0, 1700000000.02, 1700000000.04],
                "accel_x_mps2": [0.2, 0.3, 0.1],
                "accel_y_mps2": [-0.1, 0.0, 0.2],
                "accel_z_mps2": [9.7, 9.8, 9.6],
                "mag_x_uT": [22.0, 22.1, 21.9],
                "mag_y_uT": [-8.0, -8.1, -7.9],
                "mag_z_uT": [-42.0, -41.8, -42.2],
            }
        )

    with st.form("simulator_predict_form"):
        edited = st.data_editor(
            default_rows,
            num_rows="dynamic",
            use_container_width=True,
            key="sim_immu_editor",
        )
        submitted = st.form_submit_button(
            "Prédire le comportement",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if edited is None or len(edited) < 1:
            st.error("Ajoutez au moins une ligne.")
        else:
            missing = [c for c in ("timestamp", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2") if c not in edited.columns]
            if missing:
                st.error(f"Colonnes manquantes: {missing}")
            else:
                try:
                    bmap = load_behavior_map_optional()
                    head_p = Path(head_path_txt.strip()) if head_path_txt.strip() else None
                    if head_p is not None and not head_p.exists():
                        st.warning("Fichier tête introuvable — synthèse depuis l’IMMU.")
                        head_p = None
                    use_csv = head_p is not None and head_p.exists() and not synth_head
                    out = predict_from_immu(
                        model_dir=Path(model_dir),
                        immu_file=None,
                        immu_df=edited,
                        use_multimodal=default_mm,
                        head_file=head_p if use_csv else None,
                        smooth_window=int(smooth_default),
                        behavior_map=bmap,
                    )
                    st.success(f"{len(out)} seconde(s) prédite(s).")
                    lbl = _merged_behavior_labels(bmap)
                    out_named = _enrich_pred_columns(out, lbl)
                    show_c = [c for c in ("ts_sec", "pred_behavior_smooth_name", "pred_behavior_name", "pred_behavior_smooth", "pred_behavior") if c in out_named.columns]
                    rest = [c for c in out_named.columns if c not in show_c]
                    st.dataframe(out_named[show_c + rest], use_container_width=True)
                    _render_behavior_breakdown(out, lbl, "Répartition du temps (simulateur)")
                except Exception as e:
                    st.exception(e)

with tab_milk:
    st.subheader("Prédire la production de lait (kg/jour)")
    st.caption(
        "Modèle par défaut : XGBoost régularisé entraîné sur **CBT + THI + behavior journalier** "
        "(moyennes issues des prédictions IMMU) + **milk_lag1** / roll3. Prédit `milk_weight_kg` (kg/jour)."
    )
    st.info(
        "Entrée minimale : **THI** (temp °C + humidité %), **CBT** (temp °C). "
        "Le **behavior** peut être auto-rempli depuis `behavior_daily_from_predictions.csv` (aligné à l’entraînement) ; "
        "sinon utilise un **behavior id (1–7)** simplifié."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        milk_model_path_raw = st.text_input(
            "Chemin modèle lait (joblib)",
            value=r"C:\sensor_data\model_outputs\milk_xgb_pred_behavior_daily_milkhist_venv_pipeline.joblib",
            key="milk_model_joblib_path",
        )
    with c2:
        milk_metrics_path_raw = st.text_input(
            "Chemin metrics.json (pour la liste des features)",
            value=r"C:\sensor_data\model_outputs\milk_xgb_pred_behavior_daily_milkhist_venv_metrics.json",
            key="milk_model_metrics_path",
        )

    default_out_dir = Path(r"C:\sensor_data\model_outputs")
    milk_model_path = _resolve_local_artifact_path(milk_model_path_raw, default_dir=default_out_dir)
    milk_metrics_path = _resolve_local_artifact_path(milk_metrics_path_raw, default_dir=default_out_dir)

    # Guardrails: users sometimes paste metrics.json into model path (or vice versa)
    if milk_model_path.suffix.lower() == ".json" and milk_model_path.name.endswith("_metrics.json"):
        st.warning("Tu as mis un fichier `*_metrics.json` dans le champ modèle. Le modèle doit être un `*_pipeline.joblib`.")
    if milk_metrics_path.suffix.lower() == ".joblib":
        st.warning("Tu as mis un fichier `.joblib` dans le champ metrics. Les metrics doivent être un `*_metrics.json`.")

    feature_cols = _load_milk_feature_list_from_metrics(Path(milk_metrics_path))
    if not feature_cols:
        st.error(
            "Impossible de lire `features` depuis metrics.json. Vérifie le chemin et que le fichier existe."
        )
        st.stop()

    try:
        milk_pipeline = _load_milk_pipeline(str(milk_model_path))
    except Exception as e:
        st.error(f"Impossible de charger le modèle: {e}")
        st.stop()

    st.markdown("**Entrée** (tu peux laisser vide certains champs; ils seront imputés).")

    left, right = st.columns([1, 1])
    with left:
        cow_id = st.text_input("cow_id", value="C01")
        date_str = st.date_input("date", value=pd.Timestamp("2023-08-03")).strftime("%Y-%m-%d")
        st.markdown("**Contexte lactation / calendrier**")
        dim = st.number_input(
            "DIM (Days In Milk)",
            value=220.0,
            step=1.0,
            help="Nombre de jours depuis le début de lactation (vient de tes fichiers milk).",
        )
        st.markdown("**THI (air)**")
        thi_temp_c = st.number_input("température (°C)", value=25.0, step=0.1)
        thi_rh_pct = st.number_input("humidity_per (%)", value=60.0, step=1.0)

        st.markdown("**CBT (body)**")
        cbt_temp_c = st.number_input("cbt temperature_C (°C)", value=38.4, step=0.01)

        st.markdown("**Historique lait (optionnel mais recommandé)**")
        auto_milk_hist = st.checkbox(
            "Auto-remplir milk_lag1/roll3 depuis les fichiers milk",
            value=True,
            help="Cherche dans `sensor_root/main_data/milk/{cow_id}.csv`.",
        )
        milk_lag1_manual = st.number_input(
            "milk_lag1 (kg, hier) — manuel",
            value=0.0,
            step=0.1,
            help="Si tu connais la production d’hier, mets-la ici. Sinon laisse 0 (sera imputé si auto indisponible).",
        )
        milk_roll3_manual = st.number_input("milk_roll3_mean (kg, 3j) — manuel", value=0.0, step=0.1)

        behavior_daily_csv = st.text_input(
            "CSV behavior journalier (prédictions IMMU)",
            value=r"C:\sensor_data\model_outputs\behavior_daily_from_predictions.csv",
            help="Produit par build_behavior_daily_features.py à partir de behavior_pred_seconds/*.csv",
        )
        auto_behavior_daily = st.checkbox(
            "Auto-remplir behavior_n / behavior_mean / behavior_std depuis ce CSV",
            value=True,
        )

    with right:
        st.markdown("**Behavior (manuel si pas d’auto)**")
        behavior_id = st.number_input(
            "behavior id simplifié (1–7), si auto behavior indisponible",
            value=7,
            step=1,
            help="Utilisé seulement si la ligne (cow_id, date) n’est pas dans le CSV behavior journalier.",
        )
        st.caption(
            "Mapping (README): 1 Walking, 2 Standing, 3 Feeding head up, 4 Feeding head down, "
            "5 Licking, 6 Drinking, 7 Lying."
        )

    dt = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(dt):
        st.error("Date invalide.")
        st.stop()

    dow_val = int(dt.dayofweek)
    month_val = int(dt.month)
    st.caption(f"Calendrier: dow={dow_val} (0=Lun..6=Dim), month={month_val}")

    thi_val = _thi_from_temp_rh_celsius(thi_temp_c, thi_rh_pct)
    st.caption(f"THI calculé: **{thi_val:.2f}**")

    beh_auto: dict = {}
    if auto_behavior_daily:
        beh_auto = _try_autofill_behavior_daily(behavior_daily_csv, cow_id, date_str)

    st.markdown("**Behavior journalier (moyenne sur la journée — comme à l’entraînement)**")
    if auto_behavior_daily and beh_auto and "behavior_mean" in beh_auto:
        st.caption(
            "Source : ligne **(cow_id, date)** du CSV behavior daily — "
            "**behavior_mean** = moyenne des classes 1–7 sur toutes les secondes du jour."
        )
    elif auto_behavior_daily:
        st.warning(
            "Aucune ligne pour cette vache/date dans le CSV — comportement simplifié : "
            "**behavior_mean = behavior id** (1–7), sans vraie moyenne journalière."
        )

    # Valeurs par défaut : CSV (moyenne réelle) ou fallback id 1–7
    if beh_auto and "behavior_mean" in beh_auto and "behavior_n" in beh_auto:
        _dm = float(beh_auto["behavior_mean"])
        _dn = float(beh_auto["behavior_n"])
        _ds = float(beh_auto.get("behavior_std", 0.0) or 0.0)
    else:
        _dm = float(behavior_id)
        _dn = 86400.0
        _ds = 0.0

    _key = f"bh_{str(cow_id).strip()}_{date_str}".replace(" ", "_")
    # Inclut auto + id pour réinitialiser les champs si on passe CSV → mode simplifié
    _key_beh = f"{_key}_auto{int(auto_behavior_daily)}_bid{int(behavior_id)}"

    behavior_mean = st.number_input(
        "behavior_mean (moyenne du comportement sur la journée)",
        min_value=0.0,
        max_value=10.0,
        value=_dm,
        step=0.0001,
        format="%.4f",
        key=f"bm_{_key_beh}",
        help="Entraînement : moyenne des classes 1–7 sur toutes les secondes du jour (pas un seul id).",
    )
    behavior_n = st.number_input(
        "behavior_n (nombre de secondes utilisées)",
        min_value=0.0,
        value=_dn,
        step=1.0,
        key=f"bn_{_key_beh}",
    )
    behavior_std = st.number_input(
        "behavior_std (écart-type des classes sur la journée)",
        min_value=0.0,
        value=max(0.0, _ds),
        step=0.0001,
        format="%.4f",
        key=f"bs_{_key_beh}",
    )

    cbt_mean = float(cbt_temp_c)
    cbt_std = 0.0
    cbt_min = float(cbt_temp_c)
    cbt_max = float(cbt_temp_c)

    env_temp = float(thi_temp_c)
    env_hum = float(thi_rh_pct)

    milk_hist = {}
    if auto_milk_hist:
        milk_hist = _try_autofill_milk_history(sensor_root=sensor_root, cow_id=cow_id, date_str=date_str)
        if milk_hist:
            st.caption(f"Auto milk history: milk_lag1={milk_hist.get('milk_lag1')}, roll3={milk_hist.get('milk_roll3_mean')}")
        else:
            st.caption("Auto milk history: non disponible (utilise manuel / imputation).")

    def _manual_or_nan(x: float) -> float:
        return float(x) if float(x) != 0.0 else float(np.nan)

    milk_lag1 = float(milk_hist.get("milk_lag1", _manual_or_nan(milk_lag1_manual)))
    milk_roll3 = float(milk_hist.get("milk_roll3_mean", _manual_or_nan(milk_roll3_manual)))
    st.caption(f"Features lait utilisées: milk_lag1={milk_lag1}, milk_roll3_mean={milk_roll3}")

    # Build the feature dict exactly as training expected
    payload = {
        "cow_id": cow_id,
        "DIM": dim,
        "milk_lag1": milk_lag1,
        "milk_roll3_mean": milk_roll3,
        "cbt_temp_mean": cbt_mean,
        "cbt_temp_std": cbt_std,
        "cbt_temp_min": cbt_min,
        "cbt_temp_max": cbt_max,
        "behavior_n": behavior_n,
        "behavior_mean": behavior_mean,
        "behavior_std": behavior_std,
        "thi_mean": thi_val,
        "thi_std": 0.0,
        "thi_max": thi_val,
        "env_temp_mean": env_temp,
        "env_humidity_mean": env_hum,
        "dow": dow_val,
        "month": month_val,
    }

    extra_json = st.text_area(
        "Optionnel: JSON extra features (ex: behavior_class_0_ratio, ...)",
        value="{}",
        height=120,
        help="Si ton modèle a des colonnes en plus, tu peux les ajouter ici.",
    )
    try:
        extra = json.loads(extra_json) if extra_json.strip() else {}
        if isinstance(extra, dict):
            payload.update(extra)
    except Exception:
        st.warning("JSON extra invalide — ignoré.")

    if st.button("Prédire milk (kg/jour)", type="primary", use_container_width=True):
        try:
            pred = _milk_predict_single(milk_pipeline, payload, feature_cols)
            st.success(f"Prédiction: **{pred:.2f} kg/jour**")
            st.json({"date": date_str, "cow_id": cow_id, "prediction_milk_kg_day": pred})
        except Exception as e:
            st.exception(e)
