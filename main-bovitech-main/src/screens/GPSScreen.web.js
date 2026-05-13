import React, { useMemo } from 'react';
import { StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { API_BASE_URL, DASHBOARD_BASE_URL } from '../config/api';

const DASHBOARD_PATH = '/gps/dashboard/index.html';

function herdTrackUrl() {
  const base = (DASHBOARD_BASE_URL || API_BASE_URL || '').replace(/\/$/, '');
  return `${base}${DASHBOARD_PATH}`;
}

/**
 * Expo web: use a real <iframe> instead of react-native-webview (avoids blank / zero-height issues).
 */
export default function GPSScreen() {
  const { width, height } = useWindowDimensions();
  const uri = useMemo(() => herdTrackUrl(), []);

  const innerH =
    typeof window !== 'undefined'
      ? Math.max(360, window.innerHeight - 140)
      : Math.max(360, height - 140);

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.header}>
        <Text style={styles.title}>GPS · HerdTrack</Text>
        <Text style={styles.sub} numberOfLines={2}>
          Même tableau que sous `Bovitech-GPS/dashboard` (servi par l’API sur le port 8008).
        </Text>
      </View>

      <View style={[styles.frame, { width, height: innerH }]}>
        {typeof document !== 'undefined'
          ? React.createElement('iframe', {
              title: 'HerdTrack',
              src: uri,
              allow: 'geolocation',
              style: {
                width: '100%',
                height: '100%',
                border: 'none',
                display: 'block',
                backgroundColor: '#fff',
              },
            })
          : null}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#f4f6f9',
    minHeight: '100%',
  },
  header: {
    paddingHorizontal: 16,
    paddingBottom: 10,
  },
  title: {
    fontSize: 18,
    fontWeight: '800',
    color: '#1A3C2E',
  },
  sub: {
    marginTop: 4,
    fontSize: 12,
    color: '#5c6f62',
  },
  frame: {
    alignSelf: 'stretch',
    backgroundColor: '#fff',
  },
});
