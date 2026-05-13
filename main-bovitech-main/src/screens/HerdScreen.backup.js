import React, { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Modal,
  SafeAreaView,
  Platform,
} from 'react-native';
import { colors } from '../theme/colors';
import { cows } from '../theme/mockData';
import { Badge, ProgressBar } from '../components/UIComponents';
import { t, getLanguage } from '../i18n';

const metricColors = {
  green: colors.green,
  amber: colors.amber,
  red: colors.red,
  teal: colors.teal,
};

const avatarBg = {
  danger: { bg: colors.redLight, fg: colors.red, ring: colors.red },
  warn: { bg: colors.amberLight, fg: colors.amber, ring: colors.amberMid },
  ok: { bg: colors.greenLight, fg: colors.green, ring: colors.greenMid },
  info: { bg: colors.tealLight, fg: colors.teal, ring: colors.tealMid },
};

const cardShadow =
  Platform.OS === 'ios'
    ? {
        shadowColor: '#1B4332',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.07,
        shadowRadius: 14,
      }
    : { elevation: 2 };

export default function HerdScreen() {
  const [selected, setSelected] = useState(null);
  const isAr = getLanguage() === 'ar';

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.topHeader}>
          <View style={styles.topHeaderDecor} />
          <View style={styles.screenHeader}>
            <View>
              <Text style={styles.headerKicker}>{t('herd.headerKicker')}</Text>
              <Text style={styles.screenTitle}>{t('herd.title')}</Text>
              <Text style={styles.headerSub}>
                {cows.length} {t('herd.followed')}
              </Text>
            </View>
            <TouchableOpacity style={styles.addBtn} activeOpacity={0.85}>
              <Text style={styles.addBtnText}>{t('herd.add')}</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.listWrap}>
          {cows.map((cow) => {
            const av = avatarBg[cow.status] || avatarBg.ok;
            return (
              <TouchableOpacity
                key={cow.id}
                onPress={() => setSelected(cow)}
                activeOpacity={0.88}
                style={[styles.cowCard, cardShadow]}
              >
                <View
                  style={[
                    styles.avatar,
                    { backgroundColor: av.bg, borderColor: av.ring },
                  ]}
                >
                  <Text style={{ fontSize: 18 }}>🐄</Text>
                </View>

                <View style={styles.cowInfo}>
                  <Text style={styles.cowName}>
                    {cow.name}{' '}
                    <Text style={styles.cowId}>#{cow.id}</Text>
                  </Text>
                  <Text style={styles.cowMeta}>
                    {cow.breed} · {cow.age} ans · {isAr ? cow.activity_ar : cow.activity}
                  </Text>
                </View>

                <View style={styles.cowMetric}>
                  <Text
                    style={[
                      styles.metricVal,
                      { color: metricColors[cow.metricColor] || colors.green },
                    ]}
                  >
                    {cow.metric}
                  </Text>
                  {cow.metricUnit ? (
                    <Text style={styles.metricUnit}>
                      {isAr ? cow.metricUnit_ar : cow.metricUnit}
                    </Text>
                  ) : (
                    <Badge type={cow.status} label={isAr ? cow.statusLabel_ar : cow.statusLabel} />
                  )}
                </View>
              </TouchableOpacity>
            );
          })}

          <View style={styles.moreRow}>
            <Text style={styles.moreText}>{t('herd.others')}</Text>
          </View>
        </View>
      </ScrollView>

      <Modal
        visible={!!selected}
        animationType="slide"
        transparent
        onRequestClose={() => setSelected(null)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalSheet}>
            {selected && (
              <>
                <View style={styles.modalHandle} />
                <View style={styles.modalHeader}>
                  <View>
                    <Text style={styles.modalTitle}>
                      {selected.name} <Text style={styles.modalId}>#{selected.id}</Text>
                    </Text>
                    <Text style={styles.modalBreed}>
                      {selected.breed} · {selected.age} ans
                    </Text>
                  </View>
                  <Badge type={selected.status} label={selected.statusLabel} />
                </View>

                <View style={styles.vitalsGrid}>
                  <View style={styles.vitalCard}>
                    <Text style={[styles.vitalVal, { color: colors.green }]}>
                      {selected.temp}°C
                    </Text>
                    <Text style={styles.vitalLbl}>{t('herd.temperature')}</Text>
                  </View>
                  <View style={styles.vitalCard}>
                    <Text style={[styles.vitalVal, { color: colors.teal }]}>
                      {selected.milkToday}L
                    </Text>
                    <Text style={styles.vitalLbl}>{t('herd.milkToday')}</Text>
                  </View>
                  <View style={styles.vitalCard}>
                    <Text style={[styles.vitalVal, { color: colors.amber }]}>
                      {selected.gestationDay ? `J+${selected.gestationDay}` : '—'}
                    </Text>
                    <Text style={styles.vitalLbl}>{t('herd.gestation')}</Text>
                  </View>
                  <View style={styles.vitalCard}>
                    <Text style={[styles.vitalVal, { color: colors.green }]}>
                      {selected.gpsActive ? t('common.activeState') : t('common.inactiveState')}
                    </Text>
                    <Text style={styles.vitalLbl}>{t('herd.gpsCollar')}</Text>
                  </View>
                </View>

                <Text style={styles.detailSection}>{t('herd.activity6h')}</Text>
                {[
                  { label: 'Repos', value: 58, color: colors.tealMid },
                  { label: 'Rumination', value: 22, color: colors.greenMid },
                  { label: 'Marche', value: 15, color: colors.amberMid },
                  { label: 'Alimentation', value: 5, color: colors.greenMid },
                ].map((b) => (
                  <View key={b.label} style={{ marginBottom: 8 }}>
                    <View style={styles.progressLabelRow}>
                      <Text style={styles.progressLabel}>{b.label}</Text>
                      <Text style={{ fontSize: 11, fontWeight: '700', color: b.color }}>
                        {b.value}%
                      </Text>
                    </View>
                    <ProgressBar value={b.value} color={b.color} />
                  </View>
                ))}

                <TouchableOpacity
                  style={styles.closeBtn}
                  onPress={() => setSelected(null)}
                  activeOpacity={0.85}
                >
                  <Text style={styles.closeBtnText}>{t('common.close')}</Text>
                </TouchableOpacity>
              </>
            )}
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.bg },
  scroll: { flex: 1 },
  content: { paddingBottom: 100 },

  topHeader: {
    backgroundColor: colors.green,
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 20,
    overflow: 'hidden',
    position: 'relative',
  },
  topHeaderDecor: {
    position: 'absolute',
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: 'rgba(255,255,255,0.08)',
    top: -70,
    right: -40,
  },
  screenHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    zIndex: 1,
  },
  headerKicker: {
    fontSize: 10,
    fontWeight: '700',
    color: 'rgba(255,255,255,0.65)',
    letterSpacing: 1,
    marginBottom: 4,
  },
  screenTitle: {
    fontSize: 26,
    fontWeight: '700',
    color: '#fff',
    letterSpacing: -0.5,
  },
  headerSub: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.78)',
    marginTop: 4,
  },
  addBtn: {
    backgroundColor: 'rgba(255,255,255,0.22)',
    borderRadius: 14,
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.35)',
    marginTop: 4,
  },
  addBtnText: { fontSize: 13, color: '#fff', fontWeight: '700' },

  listWrap: {
    paddingHorizontal: 16,
    marginTop: -12,
  },

  cowCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: 18,
    paddingHorizontal: 14,
    paddingVertical: 14,
    gap: 12,
    marginBottom: 12,
    borderWidth: 1.5,
    borderColor: 'rgba(45, 106, 79, 0.35)',
  },
  avatar: {
    width: 46,
    height: 46,
    borderRadius: 23,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2.5,
  },
  cowInfo: { flex: 1 },
  cowName: { fontSize: 15, fontWeight: '700', color: colors.text },
  cowId: { fontSize: 11, color: colors.grayMid, fontWeight: '400' },
  cowMeta: { fontSize: 11, color: colors.grayMid, marginTop: 3, lineHeight: 15 },
  cowMetric: { alignItems: 'flex-end', gap: 4 },
  metricVal: { fontSize: 13, fontWeight: '700' },
  metricUnit: { fontSize: 10, color: colors.grayMid },
  moreRow: {
    paddingVertical: 14,
    alignItems: 'center',
  },
  moreText: {
    fontSize: 12,
    color: colors.greenMid,
    fontWeight: '600',
  },

  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  modalSheet: {
    backgroundColor: colors.card,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    padding: 20,
    paddingTop: 12,
    borderTopWidth: 4,
    borderTopColor: colors.green,
    overflow: 'hidden',
  },
  modalHandle: {
    width: 40,
    height: 4,
    backgroundColor: colors.border,
    borderRadius: 2,
    alignSelf: 'center',
    marginBottom: 16,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  modalTitle: { fontSize: 20, fontWeight: '700', color: colors.text },
  modalId: { fontSize: 14, color: colors.grayMid, fontWeight: '400' },
  modalBreed: { fontSize: 12, color: colors.grayMid, marginTop: 2 },

  vitalsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginBottom: 16,
  },
  vitalCard: {
    flex: 1,
    minWidth: '45%',
    backgroundColor: colors.greenLight,
    borderRadius: 16,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(45, 106, 79, 0.28)',
  },
  vitalVal: { fontSize: 24, fontWeight: '700' },
  vitalLbl: { fontSize: 10, color: colors.grayMid, marginTop: 3 },

  detailSection: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.green,
    textTransform: 'uppercase',
    letterSpacing: 0.4,
    marginBottom: 10,
  },
  progressLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  progressLabel: { fontSize: 11, color: colors.grayMid },

  closeBtn: {
    marginTop: 16,
    backgroundColor: colors.green,
    borderRadius: 14,
    padding: 14,
    alignItems: 'center',
  },
  closeBtnText: { fontSize: 14, fontWeight: '700', color: '#fff' },
});
