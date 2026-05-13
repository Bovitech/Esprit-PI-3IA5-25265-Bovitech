import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import { LineChart } from 'react-native-chart-kit';
import { colors } from '../theme/colors';
import { API_BASE_URL } from '../config/api';
import {
  checkPredictionApi,
  predictBehavior,
  predictIllness,
  predictMilk,
  predictStress,
  predictVocal,
  simulateTick,
} from '../services/predictionApi';
import { fetchBarnSensor } from '../services/barnSensors';
import { illnessSanteValueFr } from '../services/liveFarmModels';
import { recordingUriToBase64 } from '../services/vocalRecordBase64';

const cardShadow =
  Platform.OS === 'ios'
    ? {
        shadowColor: '#1B4332',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 20,
      }
    : { elevation: 3 };

const greenBorder = 'rgba(45, 106, 79, 0.32)';
const chartWidth = Math.max(280, Dimensions.get('window').width - 64);

function InputField({ label, value, onChangeText, keyboardType = 'numeric', placeholder }) {
  return (
    <View style={styles.inputGroup}>
      <Text style={styles.inputLabel}>{label}</Text>
      <TextInput
        style={styles.input}
        value={value}
        onChangeText={onChangeText}
        keyboardType={keyboardType}
        placeholder={placeholder}
        placeholderTextColor={colors.textMuted}
      />
    </View>
  );
}

function LiveMetricCard({ icon, label, value, unit }) {
  return (
    <View style={styles.liveMetricCard}>
      <View style={styles.liveMetricIconWrap}>
        <Ionicons name={icon} size={16} color={colors.green} />
      </View>
      <Text style={styles.liveMetricLabel}>{label}</Text>
      <Text style={styles.liveMetricValue}>
        {value}
        <Text style={styles.liveMetricUnit}> {unit}</Text>
      </Text>
    </View>
  );
}

function SensorValueRow({ label, value }) {
  return (
    <View style={styles.sensorRow}>
      <Text style={styles.sensorLabel}>{label}</Text>
      <Text style={styles.sensorValue}>{value}</Text>
    </View>
  );
}

export default function PredictionsScreen() {
  const [apiState, setApiState] = React.useState({ status: 'checking', message: '' });
  const [loadingBehavior, setLoadingBehavior] = React.useState(false);
  const [loadingMilk, setLoadingMilk] = React.useState(false);
  const [behaviorResult, setBehaviorResult] = React.useState(null);
  const [milkResult, setMilkResult] = React.useState(null);
  const [error, setError] = React.useState('');
  const [liveRunning, setLiveRunning] = React.useState(false);
  const [liveData, setLiveData] = React.useState(null);
  const [behaviorSeries, setBehaviorSeries] = React.useState([]);
  const [milkSeries, setMilkSeries] = React.useState([]);
  const [stressSeries, setStressSeries] = React.useState([]);
  const [illnessScoreSeries, setIllnessScoreSeries] = React.useState([]);
  const [vocalIsRecording, setVocalIsRecording] = React.useState(false);
  const [vocalLoading, setVocalLoading] = React.useState(false);
  const [vocalResult, setVocalResult] = React.useState(null);
  const vocalRecordingRef = React.useRef(null);

  const stressLabelFr = (name) => {
    if (name === 'At risk') return 'A risque';
    if (name === 'Stressed') return 'Stressé';
    if (name === 'Normal') return 'Normal';
    return name || '--';
  };

  const vocalLabelDisplayFr = (name) => {
    if (!name) return '--';
    if (name === 'besoin_nourriture') return 'Besoin de nourriture';
    if (name === 'toux') return 'Toux';
    if (name === 'normal') return 'Normal';
    if (name === 'ovulation') return 'Ovulation';
    return name;
  };

  const ensureMicPermission = async () => {
    const perm = await Audio.requestPermissionsAsync();
    if (perm.granted) return true;
    setError(
      perm.status === 'denied'
        ? 'Permission micro refusée. Activez le micro dans les réglages.'
        : 'Permission micro nécessaire pour enregistrer.'
    );
    return false;
  };

  const startVocalRecording = async () => {
    setError('');
    setVocalResult(null);
    const ok = await ensureMicPermission();
    if (!ok) return;
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      vocalRecordingRef.current = recording;
      setVocalIsRecording(true);
    } catch (e) {
      setError(e?.message || 'Impossible de démarrer l’enregistrement');
      setVocalIsRecording(false);
      vocalRecordingRef.current = null;
    }
  };

  const stopVocalAndPredict = async () => {
    const rec = vocalRecordingRef.current;
    if (!rec) return;
    setVocalLoading(true);
    setError('');
    try {
      await rec.stopAndUnloadAsync();
      const uri = rec.getURI();
      vocalRecordingRef.current = null;
      setVocalIsRecording(false);
      if (!uri) {
        throw new Error('Fichier audio introuvable');
      }
      const b64 = await recordingUriToBase64(uri);
      const out = await predictVocal(b64);
      setVocalResult(out);
      if (!out?.ok) {
        setError(out?.error || 'Erreur prédiction vocale');
      }
    } catch (e) {
      setVocalResult(null);
      setError(e?.response?.data?.error || e.message || 'Erreur audio / API');
    } finally {
      setVocalLoading(false);
      try {
        await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
      } catch {
        /* ignore */
      }
    }
  };

  const [behaviorInputs, setBehaviorInputs] = React.useState({
    accel_x_mps2: '0.20',
    accel_y_mps2: '-0.10',
    accel_z_mps2: '9.70',
    mag_x_uT: '22.0',
    mag_y_uT: '-8.0',
    mag_z_uT: '-42.0',
  });

  const [milkInputs, setMilkInputs] = React.useState({
    cow_id: 'C01',
    date: '2023-08-03',
    DIM: '220',
    temp_c: '25.0',
    humidity_per: '60.0',
    cbt_temp_c: '38.4',
    behavior_mean: '7.0',
    behavior_n: '86400',
    behavior_std: '0.0',
    milk_lag1: '18.0',
    milk_roll3_mean: '18.5',
  });

  const [barn, setBarn] = React.useState(null);
  const barnRef = React.useRef(null);
  const milkInputsRef = React.useRef(milkInputs);

  React.useEffect(() => {
    milkInputsRef.current = milkInputs;
  }, [milkInputs]);

  React.useEffect(() => {
    return () => {
      const r = vocalRecordingRef.current;
      if (!r) return;
      vocalRecordingRef.current = null;
      r.stopAndUnloadAsync().catch(() => {});
    };
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await fetchBarnSensor();
        if (!cancelled) {
          setBarn(data);
          barnRef.current = data;
        }
      } catch {
        if (!cancelled) {
          setBarn(null);
          barnRef.current = null;
        }
      }
    };
    load();
    const id = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  /** T, RH (et affichage THI) depuis POST /barn_sensor — CBT reste manuel / simulé */
  React.useEffect(() => {
    if (!barn?.ok) return;
    setMilkInputs((prev) => ({
      ...prev,
      temp_c: String(Number(barn.temp_c).toFixed(2)),
      humidity_per: String(Number(barn.humidity).toFixed(2)),
    }));
  }, [barn]);

  const updateBehaviorInput = (key, value) => {
    setBehaviorInputs((prev) => ({ ...prev, [key]: value }));
  };

  const updateMilkInput = (key, value) => {
    setMilkInputs((prev) => ({ ...prev, [key]: value }));
  };

  const parseNumeric = (v, fallback = 0) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  };

  React.useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        await checkPredictionApi();
        if (mounted) {
          setApiState({ status: 'ok', message: 'API connectee' });
        }
      } catch (e) {
        if (mounted) {
          setApiState({
            status: 'down',
            message: "API non joignable. Lancez `python src/model_http_api.py` dans BOVITECH-V2-4.",
          });
        }
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  const runBehaviorPrediction = async () => {
    setError('');
    setLoadingBehavior(true);
    try {
      const payload = Object.fromEntries(
        Object.entries(behaviorInputs).map(([k, v]) => [k, parseNumeric(v, 0)])
      );
      const result = await predictBehavior(payload);
      setBehaviorResult(result);
    } catch (e) {
      setError(e?.response?.data?.error || e.message || 'Erreur prediction behavior');
    } finally {
      setLoadingBehavior(false);
    }
  };

  const runMilkPrediction = async () => {
    setError('');
    setLoadingMilk(true);
    try {
      const payload = {
        cow_id: milkInputs.cow_id,
        date: milkInputs.date,
        DIM: parseNumeric(milkInputs.DIM, 220),
        temp_c: parseNumeric(milkInputs.temp_c, 25),
        humidity_per: parseNumeric(milkInputs.humidity_per, 60),
        cbt_temp_c: parseNumeric(milkInputs.cbt_temp_c, 38.4),
        behavior_mean: parseNumeric(milkInputs.behavior_mean, 7),
        behavior_n: parseNumeric(milkInputs.behavior_n, 86400),
        behavior_std: parseNumeric(milkInputs.behavior_std, 0),
        milk_lag1: parseNumeric(milkInputs.milk_lag1, 0),
        milk_roll3_mean: parseNumeric(milkInputs.milk_roll3_mean, 0),
      };
      if (barn?.ok) {
        payload.thi_mean = Number(barn.thi);
      }
      const result = await predictMilk(payload);
      setMilkResult(result);
    } catch (e) {
      setError(e?.response?.data?.error || e.message || 'Erreur prediction milk');
    } finally {
      setLoadingMilk(false);
    }
  };

  const runLiveTick = async () => {
    const data = await simulateTick();
    const s = data.sensor;
    const br = barnRef.current;
    const mi = milkInputsRef.current;
    let milk = data.milk;
    if (br?.ok) {
      try {
        milk = await predictMilk({
          cow_id: mi.cow_id,
          date: mi.date,
          DIM: parseNumeric(mi.DIM, 220),
          temp_c: br.temp_c,
          humidity_per: br.humidity,
          thi_mean: br.thi,
          cbt_temp_c: s.cbt_temp_c,
          behavior_mean: s.behavior_mean,
          behavior_n: s.behavior_n,
          behavior_std: s.behavior_std,
          milk_lag1: s.milk_lag1,
          milk_roll3_mean: s.milk_roll3_mean,
        });
      } catch {
        /* garde le milk du tick si l’appel échoue */
      }
    }
    let stress = data.stress;
    try {
      stress = await predictStress({
        cow_id: mi.cow_id,
        cbt_temp_c: parseNumeric(mi.cbt_temp_c, 38.4),
        temp_c: br?.ok ? br.temp_c : parseNumeric(mi.temp_c, 25),
        humidity_per: br?.ok ? br.humidity : parseNumeric(mi.humidity_per, 60),
        thi: br?.ok && br.thi != null ? br.thi : undefined,
        pred_behavior: data?.behavior?.pred_behavior,
        use_server_lying_buffer: true,
      });
    } catch {
      /* garde le stress du tick */
    }

    let illness = null;
    try {
      illness = await predictIllness({
        cow_id: mi.cow_id,
        timestamp: new Date().toISOString(),
        accel_x_mps2: s.accel_x_mps2,
        accel_y_mps2: s.accel_y_mps2,
        accel_z_mps2: s.accel_z_mps2,
        mag_x_uT: s.mag_x_uT,
        mag_y_uT: s.mag_y_uT,
        mag_z_uT: s.mag_z_uT,
        temp_c: br?.ok ? br.temp_c : s.temp_c,
        humidity_per: br?.ok ? br.humidity : s.humidity_per,
        thi: br?.ok && br.thi != null ? Number(br.thi) : undefined,
        cbt_temp_c: s.cbt_temp_c,
        DIM: parseNumeric(mi.DIM, 220),
        date: mi.date,
        milk_lag1: s.milk_lag1,
        milk_roll3_mean: s.milk_roll3_mean,
        behavior_mean: s.behavior_mean,
        behavior_n: s.behavior_n,
        behavior_std: s.behavior_std,
        pred_behavior: data?.behavior?.pred_behavior,
        stress,
        use_server_lying_buffer: true,
      });
    } catch {
      illness = null;
    }

    setLiveData({ ...data, milk, stress, illness });
    setBehaviorResult(data?.behavior || null);
    setMilkResult(milk || null);
    setBehaviorSeries((prev) => {
      const next = [...prev, Number(data?.behavior?.pred_behavior ?? 0)];
      return next.slice(-30);
    });
    setMilkSeries((prev) => {
      const next = [...prev, Number(milk?.prediction_milk_kg_day ?? 0)];
      return next.slice(-30);
    });
    if (stress?.pred_stress != null && Number.isFinite(Number(stress.pred_stress))) {
      setStressSeries((prev) => [...prev, Number(stress.pred_stress)].slice(-30));
    }
    if (illness?.ok && typeof illness.health_score === 'number') {
      setIllnessScoreSeries((prev) => [...prev, Number(illness.health_score)].slice(-30));
    }
  };

  React.useEffect(() => {
    let timer = null;
    if (liveRunning) {
      runLiveTick().catch((e) => {
        setError(e?.response?.data?.error || e.message || 'Erreur live simulation');
      });
      timer = setInterval(() => {
        runLiveTick().catch((e) => {
          setError(e?.response?.data?.error || e.message || 'Erreur live simulation');
        });
      }, 1000);
    }
    return () => {
      if (timer) {
        clearInterval(timer);
      }
    };
  }, [liveRunning]);

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.topHeader}>
          <View style={styles.topHeaderDecor} />
          <View style={styles.heroRow}>
            <View style={styles.heroTitleBlock}>
              <View style={styles.aiChip}>
                <Ionicons name="analytics" size={14} color="rgba(255,255,255,0.95)" />
                <Text style={styles.aiChipText}>MODELS ONLINE</Text>
              </View>
              <Text style={styles.screenTitle}>Predictions IA</Text>
              <Text style={styles.screenSub}>
                Lait, comportement, stress, santé PPO (maladie + score 0–100) et vocal
              </Text>
            </View>
          </View>
        </View>

        <View style={styles.body}>
          <View style={[styles.apiCard, cardShadow]}>
            <View style={styles.apiHeader}>
              <Ionicons
                name={apiState.status === 'ok' ? 'checkmark-circle' : 'warning'}
                size={18}
                color={apiState.status === 'ok' ? colors.moss : colors.amber}
              />
              <Text style={styles.apiText}>
                {apiState.status === 'checking' ? 'Verification API...' : apiState.message}
              </Text>
            </View>
            <Text style={styles.apiUrl}>Endpoint: {API_BASE_URL}</Text>
          </View>

          <Text style={styles.sectionKicker}>Live (IMU + lait + stress + santé)</Text>
          <View style={[styles.card, cardShadow]}>
            <View style={styles.liveHeaderRow}>
              <View>
                <Text style={styles.cardTitle}>Simulation 1s + environnement</Text>
                <Text style={styles.inputHint}>
                  IMU: T10_0725.csv. THI:{' '}
                  {barn?.ok
                    ? 'ferme (GET /barn_sensor).'
                    : 'dérivé T/RH du formulaire lait (ou sim).'}
                  {' '}Santé: POST /predict/illness (PPO + fusion score). CBT/lying: buffer 8h
                  côté API (classe 7 = couchée).
                </Text>
              </View>
              <View style={[styles.liveBadge, liveRunning ? styles.liveBadgeOn : styles.liveBadgeOff]}>
                <View style={[styles.liveDot, liveRunning ? styles.liveDotOn : styles.liveDotOff]} />
                <Text style={styles.liveBadgeText}>{liveRunning ? 'LIVE' : 'PAUSE'}</Text>
              </View>
            </View>

            <TouchableOpacity
              style={[styles.primaryBtn, liveRunning ? styles.stopBtn : null]}
              onPress={() => setLiveRunning((v) => !v)}
              activeOpacity={0.9}
            >
              <Ionicons name={liveRunning ? 'pause' : 'play'} size={18} color="#fff" />
              <Text style={styles.primaryBtnText}>
                {liveRunning ? 'Arreter live' : 'Demarrer live (1s)'}
              </Text>
            </TouchableOpacity>

            {liveData?.sensor && (
              <>
                <View style={styles.liveMetricsGrid}>
                  <LiveMetricCard
                    icon="thermometer-outline"
                    label={barn?.ok ? 'Temp. (ferme)' : 'Temp. (sim)'}
                    value={
                      barn?.ok
                        ? Number(barn.temp_c).toFixed(2)
                        : Number(liveData.sensor.temp_c).toFixed(2)
                    }
                    unit="C"
                  />
                  <LiveMetricCard
                    icon="water-outline"
                    label={barn?.ok ? 'Humid. (ferme)' : 'Humid. (sim)'}
                    value={
                      barn?.ok
                        ? Number(barn.humidity).toFixed(2)
                        : Number(liveData.sensor.humidity_per).toFixed(2)
                    }
                    unit="%"
                  />
                  <LiveMetricCard
                    icon="body-outline"
                    label="CBT (lait / sim)"
                    value={Number(parseNumeric(milkInputs.cbt_temp_c, liveData.sensor.cbt_temp_c)).toFixed(2)}
                    unit="C"
                  />
                  <LiveMetricCard
                    icon="calculator-outline"
                    label={barn?.ok ? 'THI (ferme)' : 'THI (sim)'}
                    value={
                      barn?.ok
                        ? Number(barn.thi).toFixed(2)
                        : liveData?.milk
                          ? Number(liveData.milk.thi_mean).toFixed(2)
                          : '--'
                    }
                    unit=""
                  />
                </View>

                <View style={styles.sensorPanel}>
                  <Text style={styles.resultTitle}>IMU live values</Text>
                  <View style={styles.sensorGrid}>
                    <SensorValueRow
                      label="Accel X"
                      value={`${Number(liveData.sensor.accel_x_mps2).toFixed(3)} m/s²`}
                    />
                    <SensorValueRow
                      label="Accel Y"
                      value={`${Number(liveData.sensor.accel_y_mps2).toFixed(3)} m/s²`}
                    />
                    <SensorValueRow
                      label="Accel Z"
                      value={`${Number(liveData.sensor.accel_z_mps2).toFixed(3)} m/s²`}
                    />
                    <SensorValueRow
                      label="Mag X"
                      value={`${Number(liveData.sensor.mag_x_uT).toFixed(3)} uT`}
                    />
                    <SensorValueRow
                      label="Mag Y"
                      value={`${Number(liveData.sensor.mag_y_uT).toFixed(3)} uT`}
                    />
                    <SensorValueRow
                      label="Mag Z"
                      value={`${Number(liveData.sensor.mag_z_uT).toFixed(3)} uT`}
                    />
                  </View>
                </View>
              </>
            )}

            {(liveData?.behavior || liveData?.milk || liveData?.stress || liveData?.illness) && (
              <View style={styles.livePredGrid}>
                {liveData?.behavior && (
                  <View style={[styles.resultCard, styles.livePredCard]}>
                    <Text style={styles.resultTitle}>Behavior (auto)</Text>
                    <Text style={styles.resultMain}>{liveData.behavior.pred_behavior_name}</Text>
                    <Text style={styles.resultSub}>Classe: {liveData.behavior.pred_behavior}</Text>
                  </View>
                )}
                {liveData?.milk && (
                  <View style={[styles.resultCard, styles.livePredCard]}>
                    <Text style={styles.resultTitle}>Milk (auto)</Text>
                    <Text style={styles.resultMain}>
                      {Number(liveData.milk.prediction_milk_kg_day).toFixed(2)} kg/jour
                    </Text>
                    <Text style={styles.resultSub}>
                      Pred. continue basée sur capteurs live
                    </Text>
                  </View>
                )}
                {liveData?.stress && (
                  <View style={[styles.resultCard, styles.livePredCard, styles.stressCard]}>
                    <Text style={styles.resultTitle}>Stress V3 (THI + CBT + lying 8h)</Text>
                    {liveData.stress.ok ? (
                      <>
                        <Text style={styles.resultMain}>
                          {stressLabelFr(liveData.stress.pred_stress_name)}
                        </Text>
                        <Text style={styles.resultSub}>
                          p ={' '}
                          {liveData.stress.probabilities
                            ? Object.entries(liveData.stress.probabilities)
                                .map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`)
                                .join('  ')
                            : ''}
                        </Text>
                      </>
                    ) : (
                      <Text style={styles.resultSub}>
                        {liveData.stress.error || 'Modèle indisponible. Copiez StressDetectionV3_trained.pt dans finale_model/ ou définissez STRESS_V3_CHECKPOINT.'}
                      </Text>
                    )}
                  </View>
                )}
                {liveData?.illness && (
                  <View style={[styles.resultCard, styles.livePredCard, styles.stressCard]}>
                    <Text style={styles.resultTitle}>Santé maladie (PPO + score temporel)</Text>
                    {liveData.illness.ok ? (
                      <>
                        <Text style={styles.resultMain}>
                          {illnessSanteValueFr(liveData.illness) || '—'}
                        </Text>
                        <Text style={styles.resultSub}>
                          {liveData.illness.probabilities
                            ? Object.entries(liveData.illness.probabilities)
                                .map(([k, v]) => `${k}: ${(Number(v) * 100).toFixed(0)}%`)
                                .join('  ')
                            : '—'}
                        </Text>
                        {liveData.illness.confidence != null && Number.isFinite(Number(liveData.illness.confidence)) && (
                          <Text style={styles.resultSub}>
                            Confiance choix: {(Number(liveData.illness.confidence) * 100).toFixed(0)}%
                          </Text>
                        )}
                      </>
                    ) : (
                      <Text style={styles.resultSub}>
                        {liveData.illness.error ||
                          'Modèle indisponible. Placez illness_ppo sous finale_model/artifacts/illness_model/ (voir API /health → illness_ppo).'}
                      </Text>
                    )}
                  </View>
                )}
              </View>
            )}

            {behaviorSeries.length > 1 && (
              <View style={styles.chartCard}>
                <Text style={styles.resultTitle}>Behavior per second (live)</Text>
                <LineChart
                  data={{
                    labels: behaviorSeries.map((_, i) => String(i + 1)).filter((_, i, arr) => i % 5 === 0 || i === arr.length - 1),
                    datasets: [{ data: behaviorSeries }],
                  }}
                  width={chartWidth}
                  height={180}
                  yAxisSuffix=""
                  fromZero
                  withDots={false}
                  withInnerLines
                  withOuterLines={false}
                  chartConfig={{
                    backgroundColor: '#FFFFFF',
                    backgroundGradientFrom: '#FFFFFF',
                    backgroundGradientTo: '#FFFFFF',
                    decimalPlaces: 0,
                    color: (opacity = 1) => `rgba(45, 106, 79, ${opacity})`,
                    labelColor: (opacity = 1) => `rgba(92, 92, 87, ${opacity})`,
                    propsForBackgroundLines: { stroke: 'rgba(45,106,79,0.12)' },
                  }}
                  bezier
                  style={styles.chart}
                />
              </View>
            )}

            {milkSeries.length > 1 && (
              <View style={styles.chartCard}>
                <Text style={styles.resultTitle}>Milk prediction per second (live)</Text>
                <LineChart
                  data={{
                    labels: milkSeries.map((_, i) => String(i + 1)).filter((_, i, arr) => i % 5 === 0 || i === arr.length - 1),
                    datasets: [{ data: milkSeries }],
                  }}
                  width={chartWidth}
                  height={180}
                  yAxisSuffix=""
                  fromZero={false}
                  withDots={false}
                  withInnerLines
                  withOuterLines={false}
                  chartConfig={{
                    backgroundColor: '#FFFFFF',
                    backgroundGradientFrom: '#FFFFFF',
                    backgroundGradientTo: '#FFFFFF',
                    decimalPlaces: 2,
                    color: (opacity = 1) => `rgba(82, 183, 136, ${opacity})`,
                    labelColor: (opacity = 1) => `rgba(92, 92, 87, ${opacity})`,
                    propsForBackgroundLines: { stroke: 'rgba(45,106,79,0.12)' },
                  }}
                  bezier
                  style={styles.chart}
                />
              </View>
            )}

            {stressSeries.length > 1 && (
              <View style={styles.chartCard}>
                <Text style={styles.resultTitle}>Classe stress (0=normal, 1=a risque, 2=stress) — live</Text>
                <LineChart
                  data={{
                    labels: stressSeries.map((_, i) => String(i + 1)).filter((_, i, arr) => i % 5 === 0 || i === arr.length - 1),
                    datasets: [{ data: stressSeries }],
                  }}
                  width={chartWidth}
                  height={180}
                  yAxisSuffix=""
                  fromZero
                  segments={2}
                  withDots={false}
                  withInnerLines
                  withOuterLines={false}
                  chartConfig={{
                    backgroundColor: '#FFFFFF',
                    backgroundGradientFrom: '#FFFFFF',
                    backgroundGradientTo: '#FFFFFF',
                    decimalPlaces: 0,
                    color: (opacity = 1) => `rgba(196, 69, 54, ${opacity})`,
                    labelColor: (opacity = 1) => `rgba(92, 92, 87, ${opacity})`,
                    propsForBackgroundLines: { stroke: 'rgba(45,106,79,0.12)' },
                  }}
                  bezier
                  style={styles.chart}
                />
              </View>
            )}

            {illnessScoreSeries.length > 1 && (
              <View style={styles.chartCard}>
                <Text style={styles.resultTitle}>Score santé 0–100 (fusion PPO + règles) — live</Text>
                <LineChart
                  data={{
                    labels: illnessScoreSeries
                      .map((_, i) => String(i + 1))
                      .filter((_, i, arr) => i % 5 === 0 || i === arr.length - 1),
                    datasets: [{ data: illnessScoreSeries }],
                  }}
                  width={chartWidth}
                  height={180}
                  yAxisSuffix=""
                  fromZero
                  withDots={false}
                  withInnerLines
                  withOuterLines={false}
                  chartConfig={{
                    backgroundColor: '#FFFFFF',
                    backgroundGradientFrom: '#FFFFFF',
                    backgroundGradientTo: '#FFFFFF',
                    decimalPlaces: 0,
                    color: (opacity = 1) => `rgba(46, 65, 45, ${opacity})`,
                    labelColor: (opacity = 1) => `rgba(92, 92, 87, ${opacity})`,
                    propsForBackgroundLines: { stroke: 'rgba(45,106,79,0.12)' },
                  }}
                  bezier
                  style={styles.chart}
                />
              </View>
            )}
          </View>

          <Text style={styles.sectionKicker}>Son / vocal</Text>
          <View style={[styles.card, cardShadow]}>
            <Text style={styles.cardTitle}>État vache (audio)</Text>
            <Text style={styles.inputHint}>
              Enregistrez un son de vache (toux, normal, ovulation, besoin de nourriture). Le serveur envoie
              l’audio au modèle Keras (mel 128×128). Formats typiques: M4A (mobile). Sur navigateur web, l’audio
              est lu via le navigateur (souvent WebM) puis encodé en base64 sans expo-file-system. Sur Windows,
              installez ffmpeg si le décodage M4A côté serveur échoue.
            </Text>
            <View style={styles.vocalActionsRow}>
              <TouchableOpacity
                style={[styles.secondaryBtn, vocalIsRecording ? styles.recordBtnActive : null]}
                onPress={vocalIsRecording ? undefined : startVocalRecording}
                disabled={vocalLoading || vocalIsRecording}
                activeOpacity={0.85}
              >
                <Ionicons name="mic" size={18} color={vocalIsRecording ? '#fff' : colors.green} />
                <Text style={[styles.secondaryBtnText, vocalIsRecording ? styles.recordBtnTextOn : null]}>
                  {vocalIsRecording ? 'En cours…' : 'Enregistrer'}
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.primaryBtn, styles.vocalStopBtn]}
                onPress={stopVocalAndPredict}
                disabled={!vocalIsRecording || vocalLoading}
                activeOpacity={0.9}
              >
                {vocalLoading ? (
                  <ActivityIndicator color="#fff" />
                ) : (
                  <>
                    <Ionicons name="stop-circle" size={18} color="#fff" />
                    <Text style={styles.primaryBtnText}>Arrêter & prédire</Text>
                  </>
                )}
              </TouchableOpacity>
            </View>
            {vocalResult?.ok && (
              <View style={styles.resultCard}>
                <Text style={styles.resultTitle}>Résultat</Text>
                <Text style={styles.resultMain}>{vocalLabelDisplayFr(vocalResult.pred_vocal_name)}</Text>
                <Text style={styles.resultSub}>
                  {vocalResult.probabilities
                    ? Object.entries(vocalResult.probabilities)
                        .map(([k, v]) => `${vocalLabelDisplayFr(k)}: ${(Number(v) * 100).toFixed(0)}%`)
                        .join('   ')
                    : ''}
                </Text>
              </View>
            )}
          </View>

          <Text style={styles.sectionKicker}>Behavior model</Text>
          <View style={[styles.card, cardShadow]}>
            <Text style={styles.cardTitle}>Prediction comportement (RF multimodal)</Text>
            <Text style={styles.inputHint}>Données IMMU: accel_x/y/z + mag_x/y/z</Text>
            <View style={styles.grid}>
              <InputField
                label="accel_x_mps2"
                value={behaviorInputs.accel_x_mps2}
                onChangeText={(v) => updateBehaviorInput('accel_x_mps2', v)}
              />
              <InputField
                label="accel_y_mps2"
                value={behaviorInputs.accel_y_mps2}
                onChangeText={(v) => updateBehaviorInput('accel_y_mps2', v)}
              />
              <InputField
                label="accel_z_mps2"
                value={behaviorInputs.accel_z_mps2}
                onChangeText={(v) => updateBehaviorInput('accel_z_mps2', v)}
              />
              <InputField
                label="mag_x_uT"
                value={behaviorInputs.mag_x_uT}
                onChangeText={(v) => updateBehaviorInput('mag_x_uT', v)}
              />
              <InputField
                label="mag_y_uT"
                value={behaviorInputs.mag_y_uT}
                onChangeText={(v) => updateBehaviorInput('mag_y_uT', v)}
              />
              <InputField
                label="mag_z_uT"
                value={behaviorInputs.mag_z_uT}
                onChangeText={(v) => updateBehaviorInput('mag_z_uT', v)}
              />
            </View>

            <TouchableOpacity style={styles.primaryBtn} onPress={runBehaviorPrediction} activeOpacity={0.9}>
              {loadingBehavior ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <>
                  <Ionicons name="pulse" size={18} color="#fff" />
                  <Text style={styles.primaryBtnText}>Predire comportement</Text>
                </>
              )}
            </TouchableOpacity>

            {behaviorResult && (
              <View style={styles.resultCard}>
                <Text style={styles.resultTitle}>Resultat behavior</Text>
                <Text style={styles.resultMain}>{behaviorResult.pred_behavior_name}</Text>
                <Text style={styles.resultSub}>Classe: {behaviorResult.pred_behavior}</Text>
              </View>
            )}
          </View>

          <Text style={styles.sectionKicker}>Milk model</Text>
          <View style={[styles.card, cardShadow]}>
            <Text style={styles.cardTitle}>Prediction lait (XGBoost pipeline)</Text>
            <Text style={styles.inputHint}>Contexte vache</Text>
            <InputField
              label="cow_id"
              value={milkInputs.cow_id}
              onChangeText={(v) => updateMilkInput('cow_id', v)}
              keyboardType="default"
              placeholder="C01"
            />
            <InputField
              label="date (YYYY-MM-DD)"
              value={milkInputs.date}
              onChangeText={(v) => updateMilkInput('date', v)}
              keyboardType="default"
              placeholder="2023-08-03"
            />
            <Text style={styles.inputHint}>Contexte lactation / calendrier</Text>
            <View style={styles.grid}>
              <InputField label="DIM" value={milkInputs.DIM} onChangeText={(v) => updateMilkInput('DIM', v)} />
            </View>
            <Text style={styles.inputHint}>
              T° & humidité (air) — remplies depuis le capteur ferme quand /barn_sensor est OK. THI: envoyé
              au modèle quand le capteur est actif.
            </Text>
            <View style={styles.grid}>
              <InputField
                label="temperature (C)"
                value={milkInputs.temp_c}
                onChangeText={(v) => updateMilkInput('temp_c', v)}
              />
              <InputField
                label="humidity_per"
                value={milkInputs.humidity_per}
                onChangeText={(v) => updateMilkInput('humidity_per', v)}
              />
            </View>
            {barn?.ok && (
              <Text style={styles.barnHint}>
                Capteur ferme: THI = {Number(barn.thi).toFixed(2)} (inclus dans prédit lait)
              </Text>
            )}
            <Text style={styles.inputHint}>
              CBT (corps) — pas de capteur: valeur par défaut ou saisie manuelle
            </Text>
            <View style={styles.grid}>
              <InputField
                label="cbt temperature_C (C)"
                value={milkInputs.cbt_temp_c}
                onChangeText={(v) => updateMilkInput('cbt_temp_c', v)}
              />
            </View>
            <Text style={styles.inputHint}>Historique lait (optionnel mais recommande)</Text>
            <View style={styles.grid}>
              <InputField
                label="milk_lag1 (kg, hier)"
                value={milkInputs.milk_lag1}
                onChangeText={(v) => updateMilkInput('milk_lag1', v)}
              />
              <InputField
                label="milk_roll3_mean (kg, 3j)"
                value={milkInputs.milk_roll3_mean}
                onChangeText={(v) => updateMilkInput('milk_roll3_mean', v)}
              />
            </View>
            <Text style={styles.inputHint}>Behavior journalier (moyenne sur la journee)</Text>
            <Text style={styles.inputHintSub}>
              Source: ligne (cow_id, date) du CSV behavior daily, behavior_mean = moyenne des classes 1-7.
            </Text>
            <View style={styles.grid}>
              <InputField
                label="behavior_mean"
                value={milkInputs.behavior_mean}
                onChangeText={(v) => updateMilkInput('behavior_mean', v)}
              />
              <InputField
                label="behavior_n"
                value={milkInputs.behavior_n}
                onChangeText={(v) => updateMilkInput('behavior_n', v)}
              />
              <InputField
                label="behavior_std"
                value={milkInputs.behavior_std}
                onChangeText={(v) => updateMilkInput('behavior_std', v)}
              />
            </View>

            <TouchableOpacity style={styles.primaryBtn} onPress={runMilkPrediction} activeOpacity={0.9}>
              {loadingMilk ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <>
                  <Ionicons name="water" size={18} color="#fff" />
                  <Text style={styles.primaryBtnText}>Predire lait (kg/jour)</Text>
                </>
              )}
            </TouchableOpacity>

            {milkResult && (
              <View style={styles.resultCard}>
                <Text style={styles.resultTitle}>Resultat milk</Text>
                <Text style={styles.resultMain}>{Number(milkResult.prediction_milk_kg_day).toFixed(2)} kg/jour</Text>
                <Text style={styles.resultSub}>THI calcule: {Number(milkResult.thi_mean).toFixed(2)}</Text>
              </View>
            )}
          </View>

          {!!error && (
            <View style={[styles.errorCard, cardShadow]}>
              <Ionicons name="alert-circle" size={18} color={colors.red} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F4F7F5' },
  scroll: { flex: 1, backgroundColor: '#F4F7F5' },
  content: { paddingBottom: 110 },

  topHeader: {
    backgroundColor: colors.green,
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 22,
    overflow: 'hidden',
    position: 'relative',
  },
  topHeaderDecor: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: 'rgba(255,255,255,0.08)',
    top: -80,
    right: -50,
  },
  heroRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', zIndex: 1 },
  heroTitleBlock: { flex: 1, paddingRight: 12 },
  aiChip: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: 6,
    backgroundColor: 'rgba(255,255,255,0.18)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    marginBottom: 10,
  },
  aiChipText: { fontSize: 11, fontWeight: '700', color: 'rgba(255,255,255,0.95)', letterSpacing: 0.3 },
  screenTitle: { fontSize: 24, fontWeight: '700', color: '#fff', letterSpacing: -0.5, lineHeight: 28 },
  screenSub: { fontSize: 13, color: 'rgba(255,255,255,0.78)', marginTop: 6, lineHeight: 18 },

  body: { paddingHorizontal: 16, marginTop: -8 },
  sectionKicker: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.greenMid,
    textTransform: 'uppercase',
    letterSpacing: 0.9,
    marginBottom: 8,
    marginTop: 14,
    opacity: 0.92,
  },
  apiCard: {
    marginTop: 14,
    backgroundColor: colors.card,
    borderRadius: 18,
    borderWidth: 1.5,
    borderColor: greenBorder,
    padding: 14,
  },
  apiHeader: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  apiText: { color: colors.text, fontWeight: '600', flex: 1 },
  apiUrl: { marginTop: 6, color: colors.grayMid, fontSize: 12 },

  card: {
    backgroundColor: colors.card,
    borderRadius: 18,
    borderWidth: 1.5,
    borderColor: greenBorder,
    padding: 14,
    marginBottom: 6,
  },
  cardTitle: { fontSize: 14, fontWeight: '700', color: colors.text, marginBottom: 10 },
  liveHeaderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  liveBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
    marginTop: 2,
  },
  liveBadgeOn: { backgroundColor: 'rgba(64,145,108,0.14)', borderWidth: 1, borderColor: 'rgba(64,145,108,0.32)' },
  liveBadgeOff: { backgroundColor: 'rgba(138,137,114,0.12)', borderWidth: 1, borderColor: 'rgba(138,137,114,0.26)' },
  liveDot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  liveDotOn: { backgroundColor: colors.moss },
  liveDotOff: { backgroundColor: colors.grayMid },
  liveBadgeText: { color: colors.text, fontWeight: '700', fontSize: 11, letterSpacing: 0.4 },
  inputHint: { color: colors.grayMid, fontSize: 12, marginBottom: 8, marginTop: 6, fontWeight: '600' },
  barnHint: {
    color: colors.greenMid,
    fontSize: 12,
    marginBottom: 8,
    fontWeight: '600',
  },
  inputHintSub: { color: colors.grayMid, fontSize: 11, marginBottom: 8 },
  liveMetricsGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginTop: 12 },
  liveMetricCard: {
    width: '47.5%',
    backgroundColor: '#FAFCFA',
    borderWidth: 1,
    borderColor: 'rgba(45,106,79,0.22)',
    borderRadius: 12,
    padding: 10,
  },
  liveMetricIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 10,
    backgroundColor: colors.greenLight,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  liveMetricLabel: { color: colors.grayMid, fontSize: 11, marginBottom: 2 },
  liveMetricValue: { color: colors.text, fontWeight: '800', fontSize: 18 },
  liveMetricUnit: { color: colors.grayMid, fontWeight: '600', fontSize: 12 },
  sensorPanel: {
    marginTop: 12,
    backgroundColor: colors.greenLight,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(45,106,79,0.22)',
    padding: 12,
  },
  sensorGrid: { marginTop: 6, gap: 6 },
  sensorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 4,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(45,106,79,0.2)',
  },
  sensorLabel: { color: colors.gray, fontSize: 12, fontWeight: '600' },
  sensorValue: { color: colors.text, fontSize: 12, fontWeight: '700' },
  livePredGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginTop: 2 },
  livePredCard: { width: '47.5%' },
  stressCard: { width: '100%' },
  chartCard: {
    marginTop: 12,
    backgroundColor: colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(45,106,79,0.22)',
    padding: 10,
  },
  chart: {
    marginTop: 6,
    borderRadius: 8,
  },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  inputGroup: { width: '47.5%' },
  inputLabel: { color: colors.grayMid, fontSize: 12, marginBottom: 6 },
  input: {
    backgroundColor: '#FAFCFA',
    borderWidth: 1,
    borderColor: 'rgba(45,106,79,0.22)',
    borderRadius: 12,
    color: colors.text,
    paddingHorizontal: 10,
    paddingVertical: 9,
  },
  vocalActionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 6,
    alignItems: 'center',
  },
  secondaryBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: greenBorder,
    backgroundColor: '#FAFCFA',
    minHeight: 44,
  },
  secondaryBtnText: { color: colors.green, fontWeight: '700', fontSize: 14 },
  recordBtnActive: {
    backgroundColor: colors.green,
    borderColor: colors.green,
  },
  recordBtnTextOn: { color: '#fff' },
  vocalStopBtn: {
    marginTop: 0,
    flex: 1,
    flexGrow: 1,
    minWidth: 160,
  },
  primaryBtn: {
    marginTop: 14,
    backgroundColor: colors.green,
    borderRadius: 12,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
  },
  stopBtn: {
    backgroundColor: colors.red,
  },
  primaryBtnText: { color: '#fff', fontWeight: '700' },
  resultCard: {
    marginTop: 12,
    backgroundColor: colors.greenLight,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(45,106,79,0.22)',
    padding: 12,
  },
  resultTitle: { color: colors.grayMid, fontSize: 12 },
  resultMain: { color: colors.green, fontSize: 22, fontWeight: '800', marginTop: 2 },
  resultSub: { color: colors.text, marginTop: 4, fontSize: 13 },
  errorCard: {
    marginTop: 10,
    backgroundColor: colors.redLight,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(184,74,74,0.25)',
    padding: 12,
    flexDirection: 'row',
    gap: 8,
    alignItems: 'center',
  },
  errorText: { color: colors.red, flex: 1, fontWeight: '600' },
});
