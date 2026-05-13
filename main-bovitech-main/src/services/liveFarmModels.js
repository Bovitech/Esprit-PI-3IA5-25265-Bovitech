import { fetchBarnSensor } from './barnSensors';
import { predictBehavior, predictIllness, predictMilk, predictStress, simulateTick } from './predictionApi';

/** Heure locale 24 h pour axe du graphe (ex. 23:53:07). */
export function formatBehaviorSampleTimeFr(date = new Date()) {
  const h = String(date.getHours()).padStart(2, '0');
  const m = String(date.getMinutes()).padStart(2, '0');
  const s = String(date.getSeconds()).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

/** Classe comportementale renvoyée par le modèle (1–7). */
export function normalizeBehaviorClassId(raw) {
  const n = Number(raw);
  if (Number.isFinite(n) && n >= 1 && n <= 7) return n;
  return 2;
}
export function thiFromTempRhCelsius(tempC, rhPct) {
  const t = Number(tempC);
  const rh = Number(rhPct);
  if (!Number.isFinite(t) || !Number.isFinite(rh)) return NaN;
  return (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 26);
}

export function cowNumericIdToApiCowId(id) {
  const digits = String(id).replace(/\D/g, '') || '1';
  return `C${digits.padStart(2, '0')}`;
}

function isoDateDaysAgo(daysAgo) {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

/** Oldest → newest (7 dates ending today). */
export function last7IsoDatesOldestFirst() {
  const dates = [];
  for (let i = 6; i >= 0; i -= 1) {
    dates.push(isoDateDaysAgo(i));
  }
  return dates;
}

export function weekdayLettersFrForDates(isoDates) {
  return isoDates.map((iso) => {
    const dt = new Date(`${iso}T12:00:00`);
    const monFirst = ['L', 'M', 'M', 'J', 'V', 'S', 'D'];
    return monFirst[(dt.getDay() + 6) % 7];
  });
}

/**
 * Niveau Stress V3 aligné sur le serveur : 0 Normal, 1 At risk, 2 Stressed.
 * Utilise `pred_stress` si présent (fiable), sinon le libellé (casse / FR tolérés).
 */
export function stressAlertLevel(stress) {
  if (!stress || stress.ok === false) return null;
  const k = Number(stress.pred_stress);
  if (Number.isFinite(k)) {
    if (k === 2) return 'high';
    if (k === 1) return 'risk';
    if (k === 0) return null;
  }
  const raw = String(stress.pred_stress_name || '').trim().toLowerCase();
  const name = raw.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
  if (name.includes('stressed') || raw.includes('stressé')) return 'high';
  if (name.includes('at risk') || name.includes('a risque')) return 'risk';
  return null;
}

export function isStressAlertActive(stress) {
  return stressAlertLevel(stress) != null;
}

export function stressHealthPctDisplay(stress) {
  if (!stress || stress.ok === false) return '91%';
  const level = stressAlertLevel(stress);
  if (level === 'high') return '61%';
  if (level === 'risk') return '76%';
  const k = Number(stress.pred_stress);
  if (k === 0) return '94%';
  return '91%';
}

export function stressSubtitleFr(stress) {
  if (!stress) return 'Modèle non disponible';
  if (stress.ok === false && stress.error) return 'Modèle stress indisponible';
  const level = stressAlertLevel(stress);
  if (level === 'high') return 'Stress thermique élevé';
  if (level === 'risk') return 'Stress thermique — vigilance';
  const k = Number(stress.pred_stress);
  if (k === 0) return 'Confort thermique';
  const name = String(stress.pred_stress_name || '');
  if (/normal/i.test(name)) return 'Confort thermique';
  return name || '—';
}

export function stressRiskLabelFr(stress) {
  if (!stress || stress.ok === false) return '—';
  const k = Number(stress.pred_stress);
  if (k === 0) return 'Normal';
  if (k === 1) return 'À risque';
  if (k === 2) return 'Stressé';
  const raw = String(stress.pred_stress_name || '').trim();
  const name = raw.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  if (name.includes('normal')) return 'Normal';
  if (name.includes('at risk') || raw.toLowerCase().includes('risque')) return 'À risque';
  if (name.includes('stressed') || raw.toLowerCase().includes('stress')) return 'Stressé';
  return raw || '—';
}

/** Réponse POST /predict/illness : PPO + score 0–100 pour la carte Santé (détail vache). */
export function illnessSanteValueFr(illness) {
  if (!illness || illness.ok === false) return null;
  const score = illness.health_score;
  const label = illness.class_name_fr || illness.class_name || '—';
  if (typeof score === 'number' && Number.isFinite(score)) {
    return `${label} · ${Math.round(score)}/100`;
  }
  return label;
}

export function illnessSanteStatColor(score) {
  const s = Number(score);
  if (!Number.isFinite(s)) return null;
  if (s >= 75) return '#4F8F63';
  if (s >= 55) return '#D9962F';
  return '#C84C4C';
}

export function behaviorLabelFr(predBehavior) {
  const id = Number(predBehavior);
  const map = {
    1: 'Marche',
    2: 'Debout',
    3: 'Alimentation tête haute',
    4: 'Alimentation tête baissée',
    5: 'Léchage',
    6: 'Abreuvement',
    7: 'Couchée',
  };
  return map[id] || 'Activité';
}

/** Une valeur dominante par classe modèle 1–7 → série de 7 points (indices 0–6 = classes 1–7).
 * Bas à 0 : seules les classes non prédites restent au plancher du graphe (évite l’illusion de plusieurs activités dont « Abreuvement »). */
function behaviorPredToSevenValues(predId) {
  const pid = normalizeBehaviorClassId(predId);
  const dominant = pid - 1;
  const low = 0;
  const high = 42;
  return [0, 1, 2, 3, 4, 5, 6].map((idx) => (idx === dominant ? high : low));
}

/** Rows for ActivityChart : une ligne par pas de temps, `values` alignés sur ACTIVITY_TYPES (7 classes). */
export function buildActivityChartRowsFromPredictions(predBehaviorIds, timeLabels) {
  return predBehaviorIds.map((pid, i) => ({
    label: timeLabels?.[i] ?? `Δ${i + 1}`,
    values: behaviorPredToSevenValues(pid),
  }));
}

function defaultMilkFallback() {
  return [176, 184, 181, 190, 186, 180, 186];
}

/**
 * One simulate tick + barn + 7 milk predictions (dates only change dow/month for the model).
 */
export async function fetchHomeDashboardModelPayload(cowId = '12') {
  const apiCow = cowNumericIdToApiCowId(cowId);
  let barn = null;
  try {
    barn = await fetchBarnSensor();
  } catch {
    barn = null;
  }

  let tick = null;
  try {
    tick = await simulateTick();
  } catch {
    tick = null;
  }

  /** Même flux que PredictionsScreen (live) : POST /predict/stress + tampon lying côté serveur. */
  let stress = tick?.stress ?? null;
  if (tick?.sensor) {
    try {
      const s = tick.sensor;
      const merged = await predictStress({
        cow_id: apiCow,
        cbt_temp_c: Number(s.cbt_temp_c ?? 38.4),
        temp_c: barn?.ok ? Number(barn.temp_c) : Number(s.temp_c ?? 25),
        humidity_per: barn?.ok ? Number(barn.humidity) : Number(s.humidity_per ?? 60),
        thi: barn?.ok && barn.thi != null ? Number(barn.thi) : undefined,
        pred_behavior: tick?.behavior?.pred_behavior,
        use_server_lying_buffer: true,
      });
      if (merged && merged.ok !== false) {
        stress = merged;
      }
    } catch {
      /* conserve stress du tick si POST échoue */
    }
  }

  const sensor = tick?.sensor || {};
  const dates = last7IsoDatesOldestFirst();
  const dayLetters = weekdayLettersFrForDates(dates);

  const milkKgByDay = [];
  for (const dateStr of dates) {
    try {
      const m = await predictMilk({
        cow_id: apiCow,
        date: dateStr,
        DIM: Number(sensor.DIM) || 220,
        temp_c: barn?.ok ? barn.temp_c : sensor.temp_c,
        humidity_per: barn?.ok ? barn.humidity : sensor.humidity_per,
        thi_mean: barn?.ok ? barn.thi : undefined,
        cbt_temp_c: sensor.cbt_temp_c,
        behavior_mean: sensor.behavior_mean,
        behavior_n: sensor.behavior_n,
        behavior_std: sensor.behavior_std,
        milk_lag1: sensor.milk_lag1,
        milk_roll3_mean: sensor.milk_roll3_mean,
      });
      milkKgByDay.push(Math.max(0, Number(m.prediction_milk_kg_day ?? 0)));
    } catch {
      milkKgByDay.push(0);
    }
  }

  if (milkKgByDay.every((v) => v === 0)) {
    const fb = defaultMilkFallback();
    for (let i = 0; i < fb.length; i += 1) milkKgByDay[i] = fb[i];
  }

  const milkAvg =
    milkKgByDay.reduce((a, b) => a + b, 0) / Math.max(milkKgByDay.length, 1);

  let thiStr = '--';
  if (barn?.ok && barn.thi != null) {
    thiStr = Number(barn.thi).toFixed(1);
  } else if (sensor.temp_c != null && sensor.humidity_per != null) {
    const t = thiFromTempRhCelsius(sensor.temp_c, sensor.humidity_per);
    if (Number.isFinite(t)) thiStr = t.toFixed(1);
  }

  const milkTodayKg =
    milkKgByDay.length > 0 ? milkKgByDay[milkKgByDay.length - 1] : 0;

  return {
    barnOk: !!barn?.ok,
    tickOk: !!tick,
    stress,
    thiStr,
    stressSubtitle: stressSubtitleFr(stress),
    milkKgByDay,
    milkDayLabels: dayLetters,
    milkAvgKg: milkAvg,
    milkTodayKg,
    latestBehaviorId: tick?.behavior?.pred_behavior,
    behaviorLabel: behaviorLabelFr(tick?.behavior?.pred_behavior),
  };
}

export function buildStressAlertItems(stress, { cowName = 'Rosette', cowNum = '12' } = {}) {
  const level = stressAlertLevel(stress);
  if (!level) return [];
  const urgent = level === 'high';
  const sid = urgent ? 'high' : 'risk';
  return [
    {
      id: `stress-${sid}-${cowNum}`,
      cow: `${cowName} #${cowNum}`,
      title: urgent ? 'Stress thermique élevé' : 'Stress thermique — vigilance',
      description:
        'Classification modèle (THI + température collier + temps couchée). Vérifier ventilation et point d’eau.',
      time: 'À l’instant',
      status: urgent ? 'Urgent' : 'Surveiller',
      color: urgent ? '#C84C4C' : '#D9962F',
      icon: 'pulse-outline',
      type: 'ion',
    },
  ];
}

/**
 * Sept échantillons consécutifs : à chaque pas, /simulate/tick fournit les capteurs IMU,
 * puis POST /predict/behavior recalcule la classe (même pipeline que l’écran Prédictions).
 * L’axe X affiche l’heure locale réelle au moment de la réponse (HH:mm:ss).
 */
export async function fetchRosetteLiveBundle(rosetteId = '12') {
  const apiCow = cowNumericIdToApiCowId(rosetteId);
  let barn = null;
  try {
    barn = await fetchBarnSensor();
  } catch {
    barn = null;
  }

  const activityRows = [];
  let lastTick = null;
  let lastBehaviorPid = 2;

  for (let i = 0; i < 7; i += 1) {
    try {
      lastTick = await simulateTick();
      const s = lastTick?.sensor;
      let pid = normalizeBehaviorClassId(lastTick?.behavior?.pred_behavior);

      if (
        s &&
        [
          s.accel_x_mps2,
          s.accel_y_mps2,
          s.accel_z_mps2,
          s.mag_x_uT,
          s.mag_y_uT,
          s.mag_z_uT,
        ].every((v) => v !== undefined && v !== null)
      ) {
        try {
          const b = await predictBehavior({
            accel_x_mps2: Number(s.accel_x_mps2),
            accel_y_mps2: Number(s.accel_y_mps2),
            accel_z_mps2: Number(s.accel_z_mps2),
            mag_x_uT: Number(s.mag_x_uT),
            mag_y_uT: Number(s.mag_y_uT),
            mag_z_uT: Number(s.mag_z_uT),
          });
          pid = normalizeBehaviorClassId(b?.pred_behavior ?? pid);
        } catch {
          /* conserve classe renvoyée par le tick */
        }
      }

      lastBehaviorPid = pid;
      const sampledAt = new Date();
      activityRows.push({
        label: formatBehaviorSampleTimeFr(sampledAt),
        values: behaviorPredToSevenValues(pid),
      });
    } catch {
      activityRows.push({
        label: formatBehaviorSampleTimeFr(new Date()),
        values: behaviorPredToSevenValues(2),
      });
    }
  }

  const sensor = lastTick?.sensor || {};
  let milkKg = Number(sensor.milk_lag1) || 18;
  try {
    const todayIso = isoDateDaysAgo(0);
    const m = await predictMilk({
      cow_id: apiCow,
      date: todayIso,
      DIM: Number(sensor.DIM) || 220,
      temp_c: barn?.ok ? barn.temp_c : sensor.temp_c,
      humidity_per: barn?.ok ? barn.humidity : sensor.humidity_per,
      thi_mean: barn?.ok ? barn.thi : undefined,
      cbt_temp_c: sensor.cbt_temp_c,
      behavior_mean: sensor.behavior_mean,
      behavior_n: sensor.behavior_n,
      behavior_std: sensor.behavior_std,
      milk_lag1: sensor.milk_lag1,
      milk_roll3_mean: sensor.milk_roll3_mean,
    });
    milkKg = Math.max(0, Number(m.prediction_milk_kg_day ?? milkKg));
  } catch {
    /* keep fallback */
  }

  const tempC = Number(sensor.cbt_temp_c);
  const behaviorLabel = behaviorLabelFr(lastBehaviorPid);

  /** Même flux que fetchHomeDashboardModelPayload / Prédictions : POST /predict/stress. */
  let stress = lastTick?.stress ?? null;
  if (lastTick?.sensor) {
    try {
      const s = lastTick.sensor;
      const merged = await predictStress({
        cow_id: apiCow,
        cbt_temp_c: Number(s.cbt_temp_c ?? 38.4),
        temp_c: barn?.ok ? Number(barn.temp_c) : Number(s.temp_c ?? 25),
        humidity_per: barn?.ok ? Number(barn.humidity) : Number(s.humidity_per ?? 60),
        thi: barn?.ok && barn.thi != null ? Number(barn.thi) : undefined,
        pred_behavior: lastBehaviorPid,
        use_server_lying_buffer: true,
      });
      if (merged && merged.ok !== false) {
        stress = merged;
      }
    } catch {
      /* conserve stress du tick si présent */
    }
  }

  const stressLabelFr = stressRiskLabelFr(stress);

  let illness = null;
  if (lastTick?.sensor) {
    try {
      const todayIso = isoDateDaysAgo(0);
      const s = lastTick.sensor;
      illness = await predictIllness({
        cow_id: apiCow,
        timestamp: new Date().toISOString(),
        accel_x_mps2: s.accel_x_mps2,
        accel_y_mps2: s.accel_y_mps2,
        accel_z_mps2: s.accel_z_mps2,
        mag_x_uT: s.mag_x_uT,
        mag_y_uT: s.mag_y_uT,
        mag_z_uT: s.mag_z_uT,
        temp_c: barn?.ok ? barn.temp_c : s.temp_c,
        humidity_per: barn?.ok ? barn.humidity : s.humidity_per,
        thi: barn?.ok && barn.thi != null ? Number(barn.thi) : undefined,
        cbt_temp_c: s.cbt_temp_c,
        DIM: Number(s.DIM) || 220,
        date: todayIso,
        milk_lag1: s.milk_lag1,
        milk_roll3_mean: s.milk_roll3_mean,
        behavior_mean: s.behavior_mean,
        behavior_n: s.behavior_n,
        behavior_std: s.behavior_std,
        pred_behavior: lastBehaviorPid,
        stress,
        use_server_lying_buffer: true,
      });
    } catch {
      illness = null;
    }
  }

  return {
    milkKg: Math.round(milkKg * 10) / 10,
    tempC: Number.isFinite(tempC) ? Math.round(tempC * 10) / 10 : 38.4,
    behaviorLabelFr: behaviorLabel,
    stress,
    stressLabelFr,
    activityRows,
    apiReachable: !!lastTick,
    illness,
  };
}
