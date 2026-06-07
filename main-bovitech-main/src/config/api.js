import { Platform } from 'react-native';

const DEFAULT_BASE_URL =
  Platform.OS === 'android' ? 'http://10.0.2.2:8008' : 'http://localhost:8008';

const DEFAULT_AUTH_BASE_URL =
  Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000';

/** Django chatbot (bovitech-chatbot-main). Same port as PI_Backend — run one service at a time on :8000. */
const DEFAULT_CHATBOT_BASE_URL = DEFAULT_AUTH_BASE_URL;

export const API_BASE_URL =
  (typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_API_BASE_URL) || DEFAULT_BASE_URL;

/** PI_Backend — auth & herd CRUD (Django REST, JWT). */
export const AUTH_API_BASE_URL =
  (typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_AUTH_BASE_URL) ||
  DEFAULT_AUTH_BASE_URL;

/** HerdTrack static dashboard (served by model_http_api at /gps/dashboard/). Override if dashboard is hosted elsewhere. */
export const DASHBOARD_BASE_URL =
  (typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_DASHBOARD_BASE_URL) || API_BASE_URL;

export const CHATBOT_API_BASE_URL =
  (typeof process !== 'undefined' && process.env?.EXPO_PUBLIC_CHATBOT_API_BASE_URL) ||
  DEFAULT_CHATBOT_BASE_URL;
