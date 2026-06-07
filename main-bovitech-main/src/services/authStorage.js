import AsyncStorage from '@react-native-async-storage/async-storage';

const ACCESS_KEY = '@bovitech/auth_token';
const REFRESH_KEY = '@bovitech/auth_refresh';
const USER_KEY = '@bovitech/auth_user';

export async function saveAuthSession({ token, refresh, user }) {
  await AsyncStorage.multiSet([
    [ACCESS_KEY, token],
    [REFRESH_KEY, refresh],
    [USER_KEY, user ? JSON.stringify(user) : ''],
  ]);
}

export async function getAccessToken() {
  return AsyncStorage.getItem(ACCESS_KEY);
}

export async function getRefreshToken() {
  return AsyncStorage.getItem(REFRESH_KEY);
}

export async function getStoredUser() {
  const raw = await AsyncStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export async function clearAuthSession() {
  await AsyncStorage.multiRemove([ACCESS_KEY, REFRESH_KEY, USER_KEY]);
}
