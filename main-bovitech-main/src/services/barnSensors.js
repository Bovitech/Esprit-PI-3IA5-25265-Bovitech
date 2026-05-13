import { API_BASE_URL } from '../config/api';

/**
 * Dernière mesure étable (DHT + THI) servie par model_http_api.py (GET /barn_sensor).
 */
export async function fetchBarnSensor() {
  const url = `${API_BASE_URL.replace(/\/$/, '')}/barn_sensor`;
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}
