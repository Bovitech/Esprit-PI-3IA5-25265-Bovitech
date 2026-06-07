import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import * as authApi from '../services/authApi';
import {
  clearAuthSession,
  getAccessToken,
  getRefreshToken,
  saveAuthSession,
} from '../services/authStorage';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const persistSession = useCallback(async (tokens) => {
    await saveAuthSession({
      token: tokens.token,
      refresh: tokens.refresh,
      user: null,
    });
    const profile = await authApi.fetchProfile();
    setUser(profile);
    await saveAuthSession({
      token: tokens.token,
      refresh: tokens.refresh,
      user: profile,
    });
    return profile;
  }, []);

  const restoreSession = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) {
      setUser(null);
      return false;
    }
    try {
      const profile = await authApi.fetchProfile();
      setUser(profile);
      const refresh = await getRefreshToken();
      await saveAuthSession({ token, refresh, user: profile });
      return true;
    } catch {
      await clearAuthSession();
      setUser(null);
      return false;
    }
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        await restoreSession();
      } finally {
        if (alive) setIsLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [restoreSession]);

  const login = useCallback(
    async (email, password) => {
      const tokens = await authApi.login({ email, password });
      return persistSession(tokens);
    },
    [persistSession]
  );

  const register = useCallback(
    async (form) => {
      const tokens = await authApi.register(form);
      return persistSession(tokens);
    },
    [persistSession]
  );

  const logout = useCallback(async () => {
    await authApi.logout();
    await clearAuthSession();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({
      user,
      isLoading,
      isAuthenticated: Boolean(user),
      login,
      register,
      logout,
      restoreSession,
    }),
    [user, isLoading, login, register, logout, restoreSession]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return ctx;
}
