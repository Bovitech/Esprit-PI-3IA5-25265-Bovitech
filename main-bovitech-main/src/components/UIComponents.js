import React from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { colors } from '../theme/colors';

// ─── Badge ────────────────────────────────────────────────────────────────────
export function Badge({ type = 'ok', label }) {
  const styles = badgeStyles[type] || badgeStyles.ok;
  return (
    <View style={[badge.container, styles.container]}>
      <Text style={[badge.text, styles.text]}>{label}</Text>
    </View>
  );
}

const badge = StyleSheet.create({
  container: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
    alignSelf: 'flex-start',
  },
  text: { fontSize: 10, fontWeight: '700' },
});

const badgeStyles = {
  ok: {
    container: { backgroundColor: colors.greenLight },
    text: { color: colors.green },
  },
  warn: {
    container: { backgroundColor: colors.amberLight },
    text: { color: colors.amber },
  },
  danger: {
    container: { backgroundColor: colors.redLight },
    text: { color: colors.red },
  },
  info: {
    container: { backgroundColor: colors.blueLight },
    text: { color: colors.blue },
  },
};

// ─── Card ─────────────────────────────────────────────────────────────────────
export function Card({ children, style }) {
  return <View style={[card.container, style]}>{children}</View>;
}

const cardShadow =
  Platform.OS === 'ios'
    ? {
        shadowColor: '#1B4332',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.06,
        shadowRadius: 24,
      }
    : { elevation: 4 };

const card = StyleSheet.create({
  container: {
    backgroundColor: colors.card,
    borderRadius: 20,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: colors.divider,
    ...cardShadow,
  },
});

// ─── SectionTitle ─────────────────────────────────────────────────────────────
export function SectionTitle({ children }) {
  return <Text style={sectionTitle.text}>{children}</Text>;
}

const sectionTitle = StyleSheet.create({
  text: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.grayMid,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginTop: 16,
    marginBottom: 8,
  },
});

// ─── ProgressBar ──────────────────────────────────────────────────────────────
export function ProgressBar({ value, color = colors.greenMid }) {
  return (
    <View style={progress.track}>
      <View
        style={[
          progress.fill,
          { width: `${Math.min(value, 100)}%`, backgroundColor: color },
        ]}
      />
    </View>
  );
}

const progress = StyleSheet.create({
  track: {
    height: 7,
    backgroundColor: colors.divider,
    borderRadius: 6,
    overflow: 'hidden',
    marginVertical: 4,
  },
  fill: { height: '100%', borderRadius: 6 },
});

// ─── AlertDot ─────────────────────────────────────────────────────────────────
export function AlertDot({ type }) {
  const dotColors = {
    danger: colors.red,
    warn: colors.amberMid,
    ok: colors.greenMid,
    info: colors.blue,
  };
  return (
    <View
      style={{
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: dotColors[type] || colors.grayMid,
        marginTop: 5,
      }}
    />
  );
}
