import axios from 'axios';
import { AUTH_API_BASE_URL } from '../config/api';
import { getAccessToken, getRefreshToken } from './authStorage';

const client = axios.create({
  baseURL: AUTH_API_BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

client.interceptors.request.use(async (config) => {
  const token = await getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

function unwrap(response) {
  const body = response.data;
  if (body?.success === false) {
    throw new Error(body.error || 'Erreur serveur');
  }
  return body?.data ?? body;
}

export async function login({ email, password }) {
  const data = unwrap(await client.post('/api/auth/login/', { email, password }));
  return {
    token: data.token,
    refresh: data.refresh,
  };
}

export async function register({
  username,
  email,
  password,
  farmName,
  cowCount,
  region,
}) {
  const data = unwrap(
    await client.post('/api/auth/register/', {
      username,
      email,
      password,
      farm_name: farmName,
      cow_count: Number.parseInt(String(cowCount), 10) || 0,
      region,
    })
  );
  return {
    token: data.token,
    refresh: data.refresh,
  };
}

export async function fetchProfile() {
  return unwrap(await client.get('/api/auth/me/'));
}

export async function logout(refreshToken) {
  const refresh = refreshToken || (await getRefreshToken());
  if (!refresh) return;
  try {
    await client.post('/api/auth/logout/', { refresh });
  } catch {
    // Session cleared locally even if blacklist fails (expired token, etc.)
  }
}
