import { Platform } from 'react-native';

const DEFAULT_BASE_URL =
  Platform.OS === 'android' ? 'http://10.0.2.2:8008' : 'http://localhost:8008';

/** Django chatbot (bovitech-chatbot-main). Émulateur Android: 10.0.2.2. Appareil physique: IP du PC (ex. http://192.168.1.10:8000). */
const DEFAULT_CHATBOT_BASE_URL =
  Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000';

export const API_BASE_URL =
  (typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_API_BASE_URL) || DEFAULT_BASE_URL;

/** HerdTrack static dashboard (served by model_http_api at /gps/dashboard/). Override if dashboard is hosted elsewhere. */
export const DASHBOARD_BASE_URL =
  (typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_DASHBOARD_BASE_URL) || API_BASE_URL;

export const CHATBOT_API_BASE_URL =
  (typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_CHATBOT_API_BASE_URL) ||
  DEFAULT_CHATBOT_BASE_URL;
