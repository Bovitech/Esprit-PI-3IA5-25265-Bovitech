import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { colors } from '../theme/colors';
import { aiRisks, behaviorData } from '../theme/mockData';
import { Badge, Card, ProgressBar } from '../components/UIComponents';
import { t, getLanguage } from '../i18n';

const behaviorColors = {
  teal: colors.tealMid,
  green: colors.greenMid,
  amber: colors.amberMid,
};

function HealthSectionTitle({ children, style }) {
  return <Text style={[styles.sectionKicker, style]}>{children}</Text>;
}

const vitalIcons = {
  'Temp. moy. troupeau': 'thermometer-outline',
  'Activité normale': 'pulse-outline',
  'Risques détectés': 'warning-outline',
  'Chaleurs détectées': 'heart-outline',
};

const cardShadow =
  Platform.OS === 'ios'
    ? {
        shadowColor: '#1B4332',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.06,
        shadowRadius: 20,
      }
    : { elevation: 3 };

/** Bordure verte légère (cohérent avec Troupeau) */
const greenBorder = 'rgba(45, 106, 79, 0.32)';

export default function HealthScreen() {
  const isAr = getLanguage() === 'ar';
  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.topHeader}>
          <View style={styles.topHeaderDecor} />
          <View style={styles.heroRow}>
            <View style={styles.heroTitleBlock}>
              <View style={styles.aiChip}>
                <Ionicons name="sparkles" size={14} color="rgba(255,255,255,0.95)" />
                <Text style={styles.aiChipText}>{t('health.aiNow')}</Text>
              </View>
              <Text style={styles.screenTitle}>{t('health.title')}</Text>
              <Text style={styles.screenSub}>{t('health.sub')}</Text>
            </View>
            <View style={styles.scoreRing}>
              <Text style={styles.scoreVal}>A</Text>
              <Text style={styles.scoreLbl}>{t('health.index')}</Text>
            </View>
          </View>
        </View>

        <View style={styles.body}>
          <HealthSectionTitle style={styles.sectionKickerFirst}>
            {t('health.herdVitals')}
          </HealthSectionTitle>
          <View style={styles.vitalsGrid}>
            {[
              { val: '38.4°C', lbl: 'Temp. moy. troupeau', color: behaviorColors.green },
              { val: '87%', lbl: 'Activité normale', color: behaviorColors.teal },
              { val: '3', lbl: 'Risques détectés', color: behaviorColors.amber },
              { val: '4', lbl: 'Chaleurs détectées', color: colors.moss },
            ].map((v) => (
              <View key={v.lbl} style={styles.vitalTile}>
                <View style={[styles.vitalIconWrap, { borderColor: v.color + '44' }]}>
                  <Ionicons name={vitalIcons[v.lbl] || 'ellipse-outline'} size={20} color={v.color} />
                </View>
                <Text style={[styles.vitalVal, { color: v.color }]}>{v.val}</Text>
                <Text style={styles.vitalLbl}>{v.lbl}</Text>
              </View>
            ))}
          </View>

          <HealthSectionTitle>{t('health.aiRisks')}</HealthSectionTitle>
          {aiRisks.map((risk) => (
            <View key={risk.id} style={[styles.riskShell, cardShadow]}>
              <View
                style={[
                  styles.riskAccent,
                  risk.severity === 'warn' ? styles.riskAccentWarn : styles.riskAccentInfo,
                ]}
              />
              <View style={styles.riskInner}>
                <View style={styles.riskHeader}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.riskTitle}>{risk.cowName}</Text>
                    <Text style={styles.riskName}>{isAr ? risk.risk_ar : risk.risk}</Text>
                  </View>
                  <Badge type={risk.severity} label={`${risk.probability}%`} />
                </View>
                <Text style={styles.riskDesc}>
                  {isAr ? risk.description_ar : risk.description}
                </Text>
                {risk.severity === 'warn' && (
                  <TouchableOpacity style={styles.aiBtn} activeOpacity={0.85}>
                    <Ionicons name="chatbubbles-outline" size={18} color={colors.amber} />
                    <Text style={styles.aiBtnText}>{t('health.askVet')}</Text>
                  </TouchableOpacity>
                )}
              </View>
            </View>
          ))}

          <HealthSectionTitle>{t('health.behavior')}</HealthSectionTitle>
          <Card style={styles.accentCard}>
            {behaviorData.map((b, i) => (
              <View
                key={b.label}
                style={[styles.behaviorRow, i < behaviorData.length - 1 && styles.behaviorRowBorder]}
              >
                <View style={styles.progressLabelRow}>
                  <Text style={styles.progressLabel}>{isAr ? b.label_ar : b.label}</Text>
                  <Text
                    style={[
                      styles.progressVal,
                      { color: behaviorColors[b.color] || behaviorColors.green },
                    ]}
                  >
                    {b.value}%
                  </Text>
                </View>
                <ProgressBar
                  value={b.value}
                  color={behaviorColors[b.color] || behaviorColors.green}
                />
              </View>
            ))}
          </Card>

          <HealthSectionTitle>{t('health.summary30')}</HealthSectionTitle>
          <Card style={styles.accentCard}>
            {[
              { label: 'Consultations vétérinaires', val: '3', icon: '🩺' },
              { label: 'Traitements administrés', val: '5', icon: '💊' },
              { label: 'Anomalies détectées par IA', val: '8', icon: '🔍' },
              { label: 'Taux de détection précoce', val: '87%', icon: '🎯' },
            ].map((item, i) => (
              <View
                key={item.label}
                style={[styles.summaryRow, i < 3 && styles.summaryRowBorder]}
              >
                <Text style={styles.summaryIcon}>{item.icon}</Text>
                <Text style={styles.summaryLabel}>{item.label}</Text>
                <Text style={styles.summaryVal}>{item.val}</Text>
              </View>
            ))}
          </Card>

          <HealthSectionTitle>{t('health.events')}</HealthSectionTitle>
          <Card style={[styles.accentCard, styles.lastCard]}>
            {[
              {
                date: '27 Mar',
                event: "Insémination — Samira #05",
                type: 'info',
              },
              {
                date: '02 Avr',
                event: 'Contrôle laitier mensuel',
                type: 'ok',
              },
              {
                date: '15 Avr',
                event: 'Vélage prévu — Hana #18',
                type: 'warn',
              },
            ].map((e, i) => (
              <View
                key={e.date}
                style={[styles.eventRow, i < 2 && styles.eventRowBorder]}
              >
                <View style={styles.eventDate}>
                  <Text style={styles.eventDateText}>{e.date}</Text>
                </View>
                <Text style={styles.eventLabel}>{e.event}</Text>
              <Badge type={e.type} label={t('common.scheduled')} />
              </View>
            ))}
          </Card>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F4F7F5' },
  scroll: { flex: 1, backgroundColor: '#F4F7F5' },
  content: { paddingBottom: 110 },

  topHeader: {
    backgroundColor: colors.green,
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 22,
    overflow: 'hidden',
    position: 'relative',
  },
  topHeaderDecor: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: 'rgba(255,255,255,0.08)',
    top: -80,
    right: -50,
  },
  heroRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    zIndex: 1,
  },
  heroTitleBlock: { flex: 1, paddingRight: 12 },
  aiChip: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    gap: 6,
    backgroundColor: 'rgba(255,255,255,0.18)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    marginBottom: 10,
  },
  aiChipText: {
    fontSize: 11,
    fontWeight: '700',
    color: 'rgba(255,255,255,0.95)',
    letterSpacing: 0.3,
  },
  screenTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#fff',
    letterSpacing: -0.5,
    lineHeight: 28,
  },
  screenSub: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.78)',
    marginTop: 6,
    lineHeight: 18,
  },
  scoreRing: {
    width: 72,
    height: 72,
    borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.28)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreVal: {
    fontSize: 26,
    fontWeight: '800',
    color: '#fff',
  },
  scoreLbl: {
    fontSize: 9,
    fontWeight: '600',
    color: 'rgba(255,255,255,0.75)',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
    marginTop: 2,
  },

  body: {
    paddingHorizontal: 16,
    marginTop: -8,
  },

  sectionKicker: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.greenMid,
    textTransform: 'uppercase',
    letterSpacing: 0.9,
    marginBottom: 8,
    marginTop: 10,
    opacity: 0.92,
  },
  /** Décalé sous le bandeau vert pour ne pas paraître « dans » le header */
  sectionKickerFirst: {
    marginTop: 18,
  },

  vitalsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 6,
  },
  vitalTile: {
    width: '47.5%',
    backgroundColor: colors.card,
    borderRadius: 18,
    padding: 14,
    borderWidth: 1.5,
    borderColor: greenBorder,
    ...cardShadow,
  },
  vitalIconWrap: {
    width: 38,
    height: 38,
    borderRadius: 12,
    backgroundColor: colors.greenLight,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 10,
    borderWidth: 1,
  },
  vitalVal: {
    fontSize: 22,
    fontWeight: '700',
  },
  vitalLbl: {
    fontSize: 11,
    color: colors.grayMid,
    marginTop: 4,
    lineHeight: 15,
  },

  riskShell: {
    flexDirection: 'row',
    backgroundColor: colors.card,
    borderRadius: 20,
    borderWidth: 1.5,
    borderColor: greenBorder,
    marginBottom: 12,
    overflow: 'hidden',
  },
  accentCard: {
    borderColor: greenBorder,
    borderWidth: 1.5,
  },
  lastCard: { marginBottom: 8 },
  riskAccent: { width: 4 },
  riskAccentWarn: { backgroundColor: colors.amberMid },
  riskAccentInfo: { backgroundColor: colors.blue },
  riskInner: { flex: 1, padding: 16, paddingLeft: 14 },
  riskHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  riskTitle: { fontSize: 15, fontWeight: '700', color: colors.text },
  riskName: { fontSize: 13, color: colors.grayMid, marginTop: 3 },
  riskDesc: {
    fontSize: 13,
    color: colors.gray,
    lineHeight: 20,
  },
  aiBtn: {
    marginTop: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: colors.amberLight,
    borderRadius: 14,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: 'rgba(181, 122, 28, 0.2)',
  },
  aiBtnText: { fontSize: 13, fontWeight: '700', color: colors.amber },

  behaviorRow: { paddingVertical: 12 },
  behaviorRowBorder: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.divider,
  },
  progressLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  progressLabel: { fontSize: 13, color: colors.grayMid },
  progressVal: { fontSize: 12, fontWeight: '700' },

  summaryRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    gap: 12,
  },
  summaryRowBorder: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.divider,
  },
  summaryIcon: { fontSize: 18, width: 28, textAlign: 'center' },
  summaryLabel: { flex: 1, fontSize: 14, color: colors.text },
  summaryVal: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.green,
  },

  eventRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    gap: 10,
  },
  eventRowBorder: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.divider,
  },
  eventDate: {
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 12,
    backgroundColor: colors.greenLight,
    borderWidth: 1,
    borderColor: 'rgba(45, 106, 79, 0.22)',
    minWidth: 64,
    alignItems: 'center',
  },
  eventDateText: { fontSize: 11, fontWeight: '700', color: colors.green },
  eventLabel: { flex: 1, fontSize: 13, color: colors.text, lineHeight: 18 },
});
