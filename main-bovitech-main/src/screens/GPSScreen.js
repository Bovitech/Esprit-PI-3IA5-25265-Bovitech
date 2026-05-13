import React, { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Platform,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';

import { API_BASE_URL, DASHBOARD_BASE_URL } from '../config/api';

const DASHBOARD_PATH = '/gps/dashboard/index.html';

function herdTrackUrl() {
  const base = (DASHBOARD_BASE_URL || API_BASE_URL || '').replace(/\/$/, '');
  return `${base}${DASHBOARD_PATH}`;
}

export default function GPSScreen() {
  const { width, height } = useWindowDimensions();
  const uri = useMemo(() => herdTrackUrl(), []);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const onLoadEnd = useCallback(() => {
    setLoading(false);
  }, []);

  const onFail = useCallback(() => {
    setLoading(false);
    setError(
      `Impossible de charger HerdTrack.\n\nURL : ${uri}\n\n` +
        `Démarrez l’API Python (model_http_api) sur le port 8008, ou définissez EXPO_PUBLIC_DASHBOARD_BASE_URL / EXPO_PUBLIC_API_BASE_URL.`
    );
  }, [uri]);

  const reservedBottom = Platform.OS === 'ios' ? 100 : 96;
  const webHeight = Math.max(200, height - reservedBottom);

  return (
    <SafeAreaView style={[styles.safe, { width }]} edges={['top']}>
      <View style={styles.header}>
        <Text style={styles.title}>GPS · HerdTrack</Text>
        <Text style={styles.sub} numberOfLines={1}>
          Tableau de bord Bovitech-GPS
        </Text>
      </View>

      {error ? (
        <View style={[styles.center, { minHeight: webHeight * 0.4 }]}>
          <Text style={styles.err}>{error}</Text>
        </View>
      ) : (
        <View style={[styles.webWrap, { height: webHeight }]}>
          {loading ? (
            <View style={styles.overlay}>
              <ActivityIndicator size="large" color="#1B4332" />
              <Text style={styles.loadingText}>Chargement du tableau de bord…</Text>
            </View>
          ) : null}
          <WebView
            source={{ uri }}
            style={styles.web}
            onLoadEnd={onLoadEnd}
            onHttpError={onFail}
            onError={onFail}
            originWhitelist={['*']}
            javaScriptEnabled
            domStorageEnabled
            mixedContentMode="always"
            allowsInlineMediaPlayback
            setSupportMultipleWindows={false}
            {...(Platform.OS === 'android' ? { androidLayerType: 'hardware' } : {})}
          />
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#f4f6f9',
  },
  header: {
    paddingHorizontal: 16,
    paddingBottom: 8,
  },
  title: {
    fontSize: 18,
    fontWeight: '800',
    color: '#1A3C2E',
  },
  sub: {
    marginTop: 4,
    fontSize: 13,
    color: '#5c6f62',
  },
  webWrap: {
    flex: 1,
    position: 'relative',
    backgroundColor: '#fff',
  },
  web: {
    flex: 1,
    backgroundColor: '#fff',
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.85)',
    zIndex: 2,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: '#1A3C2E',
  },
  center: {
    paddingHorizontal: 20,
    paddingVertical: 24,
    justifyContent: 'center',
  },
  err: {
    fontSize: 14,
    lineHeight: 22,
    color: '#333',
  },
});
