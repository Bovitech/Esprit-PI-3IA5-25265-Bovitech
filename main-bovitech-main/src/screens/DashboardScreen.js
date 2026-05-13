// src/screens/DashboardScreen.js
import React from 'react';
import {
  View, Text, StyleSheet, ScrollView,
  TouchableOpacity, RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, SPACING, RADIUS, SHADOWS, FONTS } from '../constants/theme';
import { COWS, FARM_STATS } from '../constants/mockData';

const StatusBadge = ({ status }) => {
  const config = {
    healthy: { color: COLORS.success, label: 'Saine', icon: 'checkmark-circle' },
    alert: { color: COLORS.danger, label: 'Alerte', icon: 'warning' },
    out_of_zone: { color: COLORS.warning, label: 'Hors zone', icon: 'location' },
  };
  const c = config[status] || config.healthy;
  return (
    <View style={[styles.badge, { backgroundColor: c.color + '20' }]}>
      <Ionicons name={c.icon} size={12} color={c.color} />
      <Text style={[styles.badgeText, { color: c.color }]}>{c.label}</Text>
    </View>
  );
};

const StatCard = ({ icon, label, value, unit, color, trend }) => (
  <View style={[styles.statCard, SHADOWS.card]}>
    <View style={[styles.statIcon, { backgroundColor: color + '20' }]}>
      <Ionicons name={icon} size={22} color={color} />
    </View>
    <Text style={styles.statValue}>{value}<Text style={styles.statUnit}> {unit}</Text></Text>
    <Text style={styles.statLabel}>{label}</Text>
    {trend !== undefined && (
      <View style={styles.trendRow}>
        <Ionicons
          name={trend >= 0 ? 'trending-up' : 'trending-down'}
          size={14}
          color={trend >= 0 ? COLORS.success : COLORS.danger}
        />
        <Text style={[styles.trendText, { color: trend >= 0 ? COLORS.success : COLORS.danger }]}>
          {Math.abs(trend)}%
        </Text>
      </View>
    )}
  </View>
);

export default function DashboardScreen({ navigation }) {
  const [refreshing, setRefreshing] = React.useState(false);
  const milkTrend = (((FARM_STATS.totalMilkToday - FARM_STATS.totalMilkYesterday) / FARM_STATS.totalMilkYesterday) * 100).toFixed(1);
  const alerts = COWS.flatMap(c => c.alerts);

  const onRefresh = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 1500);
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={COLORS.primary} />}
    >
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.greeting}>Bonjour 👋</Text>
          <Text style={styles.farmName}>Ferme BoviTech</Text>
        </View>
        <TouchableOpacity style={styles.notifBtn} onPress={() => navigation.navigate('Alerts')}>
          <Ionicons name="notifications" size={22} color={COLORS.white} />
          {alerts.length > 0 && (
            <View style={styles.notifBadge}>
              <Text style={styles.notifCount}>{alerts.length}</Text>
            </View>
          )}
        </TouchableOpacity>
      </View>

      {/* Stats Row */}
      <View style={styles.statsGrid}>
        <StatCard icon="cow" label="Vaches totales" value={FARM_STATS.totalCows} unit="" color={COLORS.primary} />
        <StatCard icon="heart" label="En bonne santé" value={FARM_STATS.healthyCows} unit="" color={COLORS.success} />
        <StatCard icon="warning" label="Alertes" value={FARM_STATS.alertCows} unit="" color={COLORS.danger} />
        <StatCard
          icon="water"
          label="Lait aujourd'hui"
          value={FARM_STATS.totalMilkToday}
          unit="L"
          color={COLORS.info}
          trend={parseFloat(milkTrend)}
        />
      </View>

      {/* Alerts Banner */}
      {alerts.length > 0 && (
        <TouchableOpacity style={styles.alertBanner} onPress={() => navigation.navigate('Alerts')}>
          <Ionicons name="warning" size={18} color={COLORS.white} />
          <Text style={styles.alertBannerText}>{alerts.length} alerte(s) nécessitent votre attention</Text>
          <Ionicons name="chevron-forward" size={18} color={COLORS.white} />
        </TouchableOpacity>
      )}

      {/* Cow List */}
      <Text style={styles.sectionTitle}>Mon troupeau</Text>
      {COWS.map(cow => (
        <TouchableOpacity
          key={cow.id}
          style={[styles.cowCard, SHADOWS.card]}
          onPress={() => navigation.navigate('CowDetail', { cowId: cow.id })}
        >
          <View style={styles.cowAvatar}>
            <Text style={styles.cowEmoji}>🐄</Text>
          </View>
          <View style={styles.cowInfo}>
            <View style={styles.cowHeader}>
              <Text style={styles.cowName}>{cow.name}</Text>
              <StatusBadge status={cow.status} />
            </View>
            <Text style={styles.cowBreed}>{cow.breed} · {cow.tag}</Text>
            <View style={styles.cowStats}>
              <View style={styles.cowStat}>
                <Ionicons name="thermometer" size={13} color={COLORS.textSecondary} />
                <Text style={styles.cowStatText}>{cow.health.temperature}°C</Text>
              </View>
              <View style={styles.cowStat}>
                <Ionicons name="water" size={13} color={COLORS.info} />
                <Text style={styles.cowStatText}>{cow.milk.today}L</Text>
              </View>
              <View style={styles.cowStat}>
                <Ionicons name="flash" size={13} color={COLORS.accent} />
                <Text style={styles.cowStatText}>{cow.health.activityLevel}%</Text>
              </View>
            </View>
          </View>
          <Ionicons name="chevron-forward" size={18} color={COLORS.textLight} />
        </TouchableOpacity>
      ))}
      <View style={{ height: 24 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  header: {
    backgroundColor: COLORS.primary,
    paddingTop: 54,
    paddingBottom: SPACING.lg,
    paddingHorizontal: SPACING.lg,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  greeting: { color: COLORS.accentLight, fontSize: FONTS.sizes.sm },
  farmName: { color: COLORS.white, fontSize: FONTS.sizes.xxl, fontWeight: '700' },
  notifBtn: {
    width: 44, height: 44, borderRadius: 22,
    backgroundColor: COLORS.primaryLight,
    justifyContent: 'center', alignItems: 'center',
  },
  notifBadge: {
    position: 'absolute', top: 0, right: 0,
    backgroundColor: COLORS.danger,
    borderRadius: 10, minWidth: 18, height: 18,
    justifyContent: 'center', alignItems: 'center',
  },
  notifCount: { color: COLORS.white, fontSize: 10, fontWeight: '700' },
  statsGrid: {
    flexDirection: 'row', flexWrap: 'wrap',
    paddingHorizontal: SPACING.md, paddingTop: SPACING.md, gap: SPACING.sm,
  },
  statCard: {
    width: '47%', backgroundColor: COLORS.surface,
    borderRadius: RADIUS.md, padding: SPACING.md, alignItems: 'flex-start',
  },
  statIcon: { borderRadius: RADIUS.sm, padding: 8, marginBottom: SPACING.sm },
  statValue: { fontSize: FONTS.sizes.xl, fontWeight: '700', color: COLORS.textPrimary },
  statUnit: { fontSize: FONTS.sizes.sm, fontWeight: '400', color: COLORS.textSecondary },
  statLabel: { fontSize: FONTS.sizes.xs, color: COLORS.textSecondary, marginTop: 2 },
  trendRow: { flexDirection: 'row', alignItems: 'center', marginTop: 4, gap: 3 },
  trendText: { fontSize: FONTS.sizes.xs, fontWeight: '600' },
  alertBanner: {
    margin: SPACING.md, backgroundColor: COLORS.danger,
    borderRadius: RADIUS.md, padding: SPACING.md,
    flexDirection: 'row', alignItems: 'center', gap: SPACING.sm,
  },
  alertBannerText: { flex: 1, color: COLORS.white, fontSize: FONTS.sizes.sm, fontWeight: '600' },
  sectionTitle: {
    fontSize: FONTS.sizes.lg, fontWeight: '700',
    color: COLORS.textPrimary, marginHorizontal: SPACING.lg,
    marginTop: SPACING.md, marginBottom: SPACING.sm,
  },
  cowCard: {
    marginHorizontal: SPACING.md, marginBottom: SPACING.sm,
    backgroundColor: COLORS.surface, borderRadius: RADIUS.md,
    padding: SPACING.md, flexDirection: 'row', alignItems: 'center', gap: SPACING.sm,
  },
  cowAvatar: {
    width: 52, height: 52, borderRadius: 26,
    backgroundColor: COLORS.surfaceAlt,
    justifyContent: 'center', alignItems: 'center',
  },
  cowEmoji: { fontSize: 26 },
  cowInfo: { flex: 1 },
  cowHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  cowName: { fontSize: FONTS.sizes.md, fontWeight: '700', color: COLORS.textPrimary },
  cowBreed: { fontSize: FONTS.sizes.xs, color: COLORS.textSecondary, marginTop: 2 },
  cowStats: { flexDirection: 'row', gap: SPACING.md, marginTop: SPACING.sm },
  cowStat: { flexDirection: 'row', alignItems: 'center', gap: 3 },
  cowStatText: { fontSize: FONTS.sizes.xs, color: COLORS.textSecondary },
  badge: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: 8, paddingVertical: 3, borderRadius: RADIUS.full,
  },
  badgeText: { fontSize: 11, fontWeight: '600' },
});
