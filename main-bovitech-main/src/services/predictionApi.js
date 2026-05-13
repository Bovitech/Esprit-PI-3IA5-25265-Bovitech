import axios from 'axios';
import { API_BASE_URL } from '../config/api';

/** Cold start / ML + Tick can exceed 12s on CPU or LAN; overrides via Expo env (ms). */
const DEFAULT_HTTP_TIMEOUT_MS =
  typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_API_HTTP_TIMEOUT_MS
    ? Number(process.env.EXPO_PUBLIC_API_HTTP_TIMEOUT_MS)
    : 90000;

const VOCAL_HTTP_TIMEOUT_MS =
  typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_API_VOCAL_TIMEOUT_MS
    ? Number(process.env.EXPO_PUBLIC_API_VOCAL_TIMEOUT_MS)
    : 180000;

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: Number.isFinite(DEFAULT_HTTP_TIMEOUT_MS) ? DEFAULT_HTTP_TIMEOUT_MS : 90000,
});

const vocalClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: Number.isFinite(VOCAL_HTTP_TIMEOUT_MS) ? VOCAL_HTTP_TIMEOUT_MS : 180000,
});

export async function checkPredictionApi() {
  const { data } = await client.get('/health');
  return data;
}

export async function predictBehavior(payload) {
  const { data } = await client.post('/predict/behavior', payload);
  return data;
}

export async function predictMilk(payload) {
  const { data } = await client.post('/predict/milk', payload);
  return data;
}

/** THI, CBT (cbt_temp_c), optional lying_minutes_8h or pred_behavior, cow_id. */
export async function predictStress(payload) {
  const { data } = await client.post('/predict/stress', payload);
  return data;
}

/** Base64 WAV or M4A (expo-av HIGH_QUALITY); server uses librosa (M4A may need ffmpeg). */
export async function predictVocal(audioWavBase64) {
  const { data } = await vocalClient.post('/predict/vocal', {
    audio_wav_base64: audioWavBase64,
  });
  return data;
}

/** POST /predict/illness — PPO maladie + score santé temporel. */
export async function predictIllness(payload) {
  try {
    const { data } = await client.post('/predict/illness', payload);
    return data;
  } catch (e) {
    const msg = e?.response?.data?.error || e?.message || 'Illness API error';
    return { ok: false, error: String(msg) };
  }
}

export async function simulateTick() {
  const { data } = await client.get('/simulate/tick');
  return data;
}
