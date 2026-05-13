import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  ActivityIndicator,
  Dimensions,
  FlatList,
  Image,
  Modal,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { Audio } from 'expo-av';

import { cows as initialCows } from '../theme/mockData';
import { fetchRosetteLiveBundle, illnessSanteStatColor, illnessSanteValueFr, stressRiskLabelFr } from '../services/liveFarmModels';
import { predictVocal } from '../services/predictionApi';
import { recordingUriToBase64 } from '../services/vocalRecordBase64';
import { useFarmLocation } from '../hooks/useFarmLocation';

import cowV1 from '../../assets/v1.png';
import cowV2 from '../../assets/v2.png';
import cowV3 from '../../assets/v3.png';
import cowV4 from '../../assets/v4.png';

const { width } = Dimensions.get('window');

const COLORS = {
  primary: '#2E412D',
  earth: '#7C6144',
  cream: '#F6F6EC',
  sage: '#AFB2A1',
  dark: '#3E3E32',
  black: '#010101',
  white: '#FFFFFF',
  border: '#E5E3D6',
  success: '#4F8F63',
  warning: '#D9962F',
  danger: '#C84C4C',
  info: '#4E78A8',
  purple: '#8A63D2',
  teal: '#2A9D8F',
};

const cowImages = {
  '07': cowV1,
  '12': cowV2,
  '03': cowV3,
  '05': cowV4,
  '09': cowV1,
};

const FILTERS = [
  { key: 'all', label: 'Toutes', icon: 'apps-outline' },
  { key: 'danger', label: 'Hors zone', icon: 'location-outline' },
  { key: 'warn', label: 'Alertes', icon: 'warning-outline' },
  { key: 'info', label: 'Gestation', icon: 'leaf-outline' },
  { key: 'ok', label: 'OK', icon: 'checkmark-circle-outline' },
];

const STATUS_STYLES = {
  danger: {
    label: 'Hors zone',
    color: COLORS.danger,
    bg: 'rgba(200,76,76,0.10)',
  },
  warn: {
    label: 'Surveiller',
    color: COLORS.warning,
    bg: 'rgba(217,150,47,0.12)',
  },
  info: {
    label: 'Gestation',
    color: COLORS.info,
    bg: 'rgba(78,120,168,0.12)',
  },
  ok: {
    label: 'OK',
    color: COLORS.success,
    bg: 'rgba(79,143,99,0.12)',
  },
};

/** Classes comportementales du modèle (1–7), libellés FR — voir README / model_http_api BEHAVIOR_LABELS */
const ACTIVITY_TYPES = [
  { key: 'walking', label: 'Marche', color: '#C84C4C' },
  { key: 'standing', label: 'Debout', color: '#D9962F' },
  { key: 'feedingHeadUp', label: 'Alimentation tête haute', color: '#4E78A8' },
  { key: 'feedingHeadDown', label: 'Alimentation tête baissée', color: '#4F8F63' },
  { key: 'licking', label: 'Léchage', color: '#B98A32' },
  { key: 'drinking', label: 'Abreuvement', color: '#2A9D8F' },
  { key: 'lying', label: 'Couchée', color: '#8A63D2' },
];

const PERIODS = [
  { key: 'minute', label: 'Minute' },
  { key: 'day', label: 'Jour' },
  { key: 'week', label: 'Semaine' },
  { key: 'month', label: 'Mois' },
];

const BREED_OPTIONS = [
  'Holstein / Prim’Holstein',
  'Frisonne pie noire',
  'Brune Suisse',
  'Tarentaise',
  'Montbéliarde',
  'Brune de l’Atlas',
  'Blonde du Cap Bon',
  'Croisée / locale',
];

const THUMB_HEIGHT = 72;
const THUMB_GAP = 8;

function getStatusStyle(status) {
  return STATUS_STYLES[status] || STATUS_STYLES.ok;
}

function getEstimatedWeight(breed) {
  if (!breed) return 520;

  const value = breed.toLowerCase();

  if (value.includes('holstein')) return 620;
  if (value.includes('frisonne')) return 600;
  if (value.includes('brune suisse')) return 610;
  if (value.includes('tarentaise')) return 520;
  if (value.includes('montbéliarde')) return 590;
  if (value.includes('atlas')) return 450;
  if (value.includes('cap bon')) return 430;
  if (value.includes('locale')) return 460;

  return 520;
}

function calculateAge(dateOfBirth) {
  if (!dateOfBirth) return 0;

  const birth = new Date(dateOfBirth);
  const today = new Date();

  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();

  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age -= 1;
  }

  return Math.max(age, 0);
}

function generateInternalId() {
  return `COW-${Date.now().toString().slice(-6)}`;
}

function generateDIN(internalId) {
  const digits = internalId.replace(/\D/g, '').slice(-6);
  return `TN-BOV-${digits}`;
}

function formatDate(date) {
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, '0');
  const dd = String(date.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

function parseDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return new Date();
  return date;
}

function getActivityData(period) {
  if (period === 'minute') {
    return [
      { label: '00', values: [8, 10, 12, 18, 14, 20, 15] },
      { label: '10', values: [10, 8, 14, 16, 18, 16, 12] },
      { label: '20', values: [12, 12, 13, 15, 16, 18, 11] },
      { label: '30', values: [9, 13, 12, 14, 20, 16, 13] },
      { label: '40', values: [11, 9, 15, 13, 18, 18, 14] },
      { label: '50', values: [10, 12, 13, 14, 16, 20, 12] },
    ];
  }

  if (period === 'day') {
    return [
      { label: '00h', values: [20, 8, 12, 35, 5, 20, 14] },
      { label: '04h', values: [18, 10, 10, 40, 4, 18, 12] },
      { label: '08h', values: [24, 18, 20, 10, 12, 16, 15] },
      { label: '12h', values: [30, 20, 18, 8, 10, 14, 11] },
      { label: '16h', values: [22, 22, 16, 10, 14, 16, 13] },
      { label: '20h', values: [18, 12, 14, 28, 8, 20, 12] },
    ];
  }

  if (period === 'week') {
    return [
      { label: 'L', values: [24, 16, 18, 18, 10, 14, 12] },
      { label: 'M', values: [20, 18, 16, 22, 8, 16, 14] },
      { label: 'M', values: [22, 16, 20, 18, 12, 12, 11] },
      { label: 'J', values: [18, 20, 18, 20, 10, 14, 13] },
      { label: 'V', values: [26, 14, 16, 18, 12, 14, 12] },
      { label: 'S', values: [20, 16, 18, 24, 8, 14, 15] },
      { label: 'D', values: [22, 18, 16, 20, 10, 14, 11] },
    ];
  }

  return [
    { label: 'S1', values: [22, 16, 18, 20, 10, 14, 13] },
    { label: 'S2', values: [24, 14, 18, 18, 12, 14, 11] },
    { label: 'S3', values: [20, 18, 16, 22, 10, 14, 12] },
    { label: 'S4', values: [23, 16, 17, 19, 11, 14, 13] },
  ];
}

const BottomSheet = memo(function BottomSheet({
  visible,
  title,
  children,
  onClose,
  height = 430,
}) {
  const translateY = useRef(new Animated.Value(height)).current;
  const fade = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (visible) {
      translateY.setValue(height);
      fade.setValue(0);

      Animated.parallel([
        Animated.timing(fade, {
          toValue: 1,
          duration: 180,
          useNativeDriver: true,
        }),
        Animated.spring(translateY, {
          toValue: 0,
          friction: 9,
          tension: 90,
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [visible, height, fade, translateY]);

  if (!visible) return null;

  return (
    <Modal transparent visible={visible} animationType="none" onRequestClose={onClose}>
      <View style={sheet.overlay}>
        <Animated.View style={[sheet.backdrop, { opacity: fade }]}>
          <Pressable style={StyleSheet.absoluteFill} onPress={onClose} />
        </Animated.View>

        <Animated.View
          style={[
            sheet.container,
            {
              height,
              transform: [{ translateY }],
            },
          ]}
        >
          <View style={sheet.handle} />

          <View style={sheet.header}>
            <Text style={sheet.title}>{title}</Text>

            <TouchableOpacity style={sheet.closeBtn} onPress={onClose}>
              <Ionicons name="close" size={20} color={COLORS.primary} />
            </TouchableOpacity>
          </View>

          {children}
        </Animated.View>
      </View>
    </Modal>
  );
});

const AnimatedInfo = memo(function AnimatedInfo({ children, delay = 0, style }) {
  const anim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    anim.setValue(0);

    Animated.timing(anim, {
      toValue: 1,
      duration: 360,
      delay,
      useNativeDriver: true,
    }).start();
  }, [anim, delay, children]);

  return (
    <Animated.View
      style={[
        style,
        {
          opacity: anim,
          transform: [
            {
              translateY: anim.interpolate({
                inputRange: [0, 1],
                outputRange: [16, 0],
              }),
            },
          ],
        },
      ]}
    >
      {children}
    </Animated.View>
  );
});

const CowPhoto = memo(function CowPhoto({ status, size = 156, cowId }) {
  const { bg } = getStatusStyle(status);
  const imageSource = cowImages[cowId] || cowV1;

  return (
    <View
      style={[
        photoStyles.wrap,
        {
          width: size,
          height: size * 1.12,
          backgroundColor: bg,
        },
      ]}
    >
      <View style={photoStyles.glow} />
      <Image source={imageSource} style={photoStyles.cowImage} resizeMode="contain" />
    </View>
  );
});

const FilterPill = memo(function FilterPill({ item, active, onPress }) {
  return (
    <TouchableOpacity
      onPress={() => onPress(item.key)}
      activeOpacity={0.82}
      style={[pill.wrap, active && pill.active]}
    >
      <Ionicons name={item.icon} size={22} color={active ? COLORS.white : COLORS.primary} />
      <Text style={[pill.label, active && pill.labelActive]}>{item.label}</Text>
    </TouchableOpacity>
  );
});

const CowCard = memo(function CowCard({ cow, index, onPress }) {
  const anim = useRef(new Animated.Value(0)).current;
  const { color, label } = getStatusStyle(cow.status);

  useEffect(() => {
    Animated.timing(anim, {
      toValue: 1,
      duration: 300,
      delay: Math.min(index * 45, 300),
      useNativeDriver: true,
    }).start();
  }, [anim, index]);

  return (
    <Animated.View
      style={{
        opacity: anim,
        transform: [
          {
            translateY: anim.interpolate({
              inputRange: [0, 1],
              outputRange: [16, 0],
            }),
          },
        ],
      }}
    >
      <TouchableOpacity style={card.outer} activeOpacity={0.88} onPress={() => onPress(cow)}>
        <View style={card.photoBlock}>
          <CowPhoto status={cow.status} size={156} cowId={cow.id} />
        </View>

        <View style={card.infoCard}>
          <View style={card.topRow}>
            <View style={{ flex: 1 }}>
              <Text style={card.name} numberOfLines={1}>
                {cow.name}
              </Text>

              <Text style={card.breed} numberOfLines={1}>
                #{cow.id} · {cow.breed}
              </Text>
            </View>

            <Ionicons name="male-female-outline" size={16} color={COLORS.sage} />
          </View>

          <View style={card.metaRow}>
            <Ionicons name="calendar-outline" size={13} color={COLORS.sage} />
            <Text style={card.meta}>{cow.age} ans</Text>
          </View>

          <View style={card.metaRow}>
            <Ionicons
              name={cow.id === '12' ? 'pulse-outline' : 'location'}
              size={13}
              color={COLORS.primary}
            />
            <Text style={card.meta} numberOfLines={1}>
              {cow.activity}
            </Text>
          </View>

          <View style={card.bottomRow}>
            <View style={card.statusRow}>
              <View style={[card.dot, { backgroundColor: color }]} />
              <Text style={[card.statusText, { color }]} numberOfLines={1}>
                {cow.statusLabel || label}
              </Text>
            </View>

            <View style={card.milkPill}>
              <MaterialCommunityIcons name="water-outline" size={13} color={COLORS.primary} />
              <Text style={card.milkText}>{cow.milkToday || 18} kg</Text>
            </View>
          </View>
        </View>
      </TouchableOpacity>
    </Animated.View>
  );
});

const CalendarPickerContent = memo(function CalendarPickerContent({
  selectedDate,
  onSelect,
  minYear = 2015,
  maxYear = 2030,
}) {
  const selected = parseDate(selectedDate);
  const [visibleMonth, setVisibleMonth] = useState(
    new Date(selected.getFullYear(), selected.getMonth(), 1)
  );

  const monthName = visibleMonth.toLocaleDateString('fr-FR', {
    month: 'long',
    year: 'numeric',
  });

  const calendarDays = useMemo(() => {
    const year = visibleMonth.getFullYear();
    const month = visibleMonth.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);

    const startOffset = (firstDay.getDay() + 6) % 7;
    const days = [];

    for (let i = 0; i < startOffset; i += 1) {
      days.push(null);
    }

    for (let day = 1; day <= lastDay.getDate(); day += 1) {
      days.push(new Date(year, month, day));
    }

    while (days.length % 7 !== 0) {
      days.push(null);
    }

    return days;
  }, [visibleMonth]);

  const goPreviousMonth = () => {
    setVisibleMonth((prev) => {
      const next = new Date(prev.getFullYear(), prev.getMonth() - 1, 1);
      if (next.getFullYear() < minYear) return prev;
      return next;
    });
  };

  const goNextMonth = () => {
    setVisibleMonth((prev) => {
      const next = new Date(prev.getFullYear(), prev.getMonth() + 1, 1);
      if (next.getFullYear() > maxYear) return prev;
      return next;
    });
  };

  return (
    <View style={calendar.content}>
      <View style={calendar.monthHeader}>
        <TouchableOpacity style={calendar.monthBtn} onPress={goPreviousMonth}>
          <Ionicons name="chevron-back" size={20} color={COLORS.primary} />
        </TouchableOpacity>

        <Text style={calendar.monthTitle}>{monthName}</Text>

        <TouchableOpacity style={calendar.monthBtn} onPress={goNextMonth}>
          <Ionicons name="chevron-forward" size={20} color={COLORS.primary} />
        </TouchableOpacity>
      </View>

      <View style={calendar.weekRow}>
        {['L', 'M', 'M', 'J', 'V', 'S', 'D'].map((day, index) => (
          <Text key={`${day}-${index}`} style={calendar.weekText}>
            {day}
          </Text>
        ))}
      </View>

      <View style={calendar.grid}>
        {calendarDays.map((date, index) => {
          if (!date) {
            return <View key={`empty-${index}`} style={calendar.dayCell} />;
          }

          const dateString = formatDate(date);
          const active = selectedDate === dateString;
          const today = formatDate(new Date()) === dateString;

          return (
            <TouchableOpacity
              key={dateString}
              activeOpacity={0.84}
              style={[
                calendar.dayCell,
                active && calendar.dayCellActive,
                today && !active && calendar.dayCellToday,
              ]}
              onPress={() => onSelect(dateString)}
            >
              <Text style={[calendar.dayText, active && calendar.dayTextActive]}>
                {date.getDate()}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <View style={calendar.infoBox}>
        <Ionicons name="calendar-outline" size={20} color={COLORS.primary} />
        <Text style={calendar.infoText}>Date sélectionnée : {selectedDate}</Text>
      </View>
    </View>
  );
});

const ActivityChart = memo(function ActivityChart({ period, overrideSeries }) {
  const data = useMemo(() => {
    if (overrideSeries?.length) return overrideSeries;
    return getActivityData(period);
  }, [period, overrideSeries]);
  const [selectedActivities, setSelectedActivities] = useState(
    ACTIVITY_TYPES.map((item) => item.key)
  );

  /** Données live : montrer seulement les classes réellement prédites (évite Abreuvement / autres à faux positif visuel). */
  useEffect(() => {
    if (!overrideSeries?.length) {
      setSelectedActivities(ACTIVITY_TYPES.map((item) => item.key));
      return;
    }
    const keys = new Set();
    overrideSeries.forEach((row) => {
      if (!Array.isArray(row.values)) return;
      row.values.forEach((v, idx) => {
        if (idx < ACTIVITY_TYPES.length && Number(v) >= 18) keys.add(ACTIVITY_TYPES[idx].key);
      });
    });
    if (keys.size === 0) {
      setSelectedActivities(ACTIVITY_TYPES.map((item) => item.key));
      return;
    }
    setSelectedActivities(Array.from(keys));
  }, [overrideSeries]);

  const chartWidth = Math.min(
    width - (overrideSeries?.length ? 108 : 150),
    overrideSeries?.length ? 336 : 290
  );
  const chartHeight = 160;
  const paddingX = 18;
  const paddingY = 22;

  const visibleActivities = useMemo(() => {
    if (selectedActivities.length === 0) {
      return ACTIVITY_TYPES;
    }

    return ACTIVITY_TYPES.filter((item) => selectedActivities.includes(item.key));
  }, [selectedActivities]);

  const toggleActivity = (activityKey) => {
    setSelectedActivities((prev) => {
      const allKeys = ACTIVITY_TYPES.map((item) => item.key);
      const isGlobal = prev.length === ACTIVITY_TYPES.length;
      const isSelected = prev.includes(activityKey);

      if (isGlobal) {
        return [activityKey];
      }

      if (isSelected) {
        const next = prev.filter((item) => item !== activityKey);
        return next.length === 0 ? allKeys : next;
      }

      return [...prev, activityKey];
    });
  };

  const showAllActivities = () => {
    setSelectedActivities(ACTIVITY_TYPES.map((item) => item.key));
  };

  const getPoints = (activityIndex) => {
    const values = data.map((item) => item.values[activityIndex]);
    const maxValue = 45;

    return values.map((value, index) => {
      const x =
        paddingX +
        (index / Math.max(data.length - 1, 1)) * (chartWidth - paddingX * 2);

      const y =
        paddingY +
        (1 - value / maxValue) * (chartHeight - paddingY * 2);

      return {
        x,
        y,
        value,
        label: data[index].label,
      };
    });
  };

  return (
    <View style={activity.card}>
      <View style={activity.topRow}>
        <View style={{ flex: 1 }}>
          <Text style={activity.kicker}>ACTIVITÉ DE LA VACHE</Text>
          <Text style={activity.title}>Courbe comportementale globale</Text>
        </View>

        <TouchableOpacity
          activeOpacity={0.84}
          style={activity.totalBadge}
          onPress={showAllActivities}
        >
          <Text style={activity.totalText}>Toutes</Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={activity.lineLegendWrap}
      >
        {ACTIVITY_TYPES.map((item) => {
          const active = selectedActivities.includes(item.key);

          return (
            <TouchableOpacity
              key={item.key}
              activeOpacity={0.82}
              onPress={() => toggleActivity(item.key)}
              style={[
                activity.lineLegendBtn,
                active && {
                  backgroundColor: `${item.color}18`,
                  borderColor: item.color,
                },
              ]}
            >
              <View style={[activity.legendDot, { backgroundColor: item.color }]} />
              <Text style={[activity.lineLegendText, active && { color: item.color }]}>
                {item.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>

      <View style={activity.curveBox}>
        <View style={activity.curveGridLineTop} />
        <View style={activity.curveGridLineMid} />
        <View style={activity.curveGridLineBottom} />

        <View style={{ width: chartWidth, height: chartHeight }}>
          {visibleActivities.map((activityType) => {
            const activityIndex = ACTIVITY_TYPES.findIndex((item) => item.key === activityType.key);
            const points = getPoints(activityIndex);

            return (
              <View key={activityType.key}>
                {points.map((point, index) => {
                  if (index === points.length - 1) return null;

                  const next = points[index + 1];
                  const dx = next.x - point.x;
                  const dy = next.y - point.y;
                  const length = Math.sqrt(dx * dx + dy * dy);
                  const angle = `${Math.atan2(dy, dx)}rad`;

                  return (
                    <View
                      key={`${activityType.key}-line-${index}`}
                      style={[
                        activity.curveLine,
                        {
                          left: point.x + dx / 2 - length / 2,
                          top: point.y + dy / 2 - 1.4,
                          width: length,
                          backgroundColor: activityType.color,
                          transform: [{ rotate: angle }],
                        },
                      ]}
                    />
                  );
                })}

                {points.map((point, index) => (
                  <TouchableOpacity
                    key={`${activityType.key}-point-${index}`}
                    activeOpacity={0.85}
                    style={[
                      activity.curvePointTouch,
                      {
                        left: point.x - 12,
                        top: point.y - 12,
                      },
                    ]}
                  >
                    <View
                      style={[
                        activity.curvePoint,
                        {
                          backgroundColor: activityType.color,
                        },
                      ]}
                    />
                  </TouchableOpacity>
                ))}
              </View>
            );
          })}

          {data.map((item, index) => {
            const x =
              paddingX +
              (index / Math.max(data.length - 1, 1)) * (chartWidth - paddingX * 2);

            return (
              <Text
                key={`label-${item.label}-${index}`}
                style={[
                  activity.curveXAxisLabel,
                  overrideSeries?.length ? { fontSize: 8 } : null,
                  {
                    left: x - 18,
                    top: chartHeight - 2,
                  },
                ]}
              >
                {item.label}
              </Text>
            );
          })}
        </View>
      </View>

      <View style={activity.selectedPanel}>
        <Text style={activity.selectedTitle}>
          {selectedActivities.length === ACTIVITY_TYPES.length
            ? 'Vue globale'
            : `${selectedActivities.length} activité(s) affichée(s)`}
        </Text>

        <Text style={activity.curveDesc}>
          {overrideSeries?.length
            ? 'Échantillons horodatés à la réception : classe prédite via POST /predict/behavior (IMU du tick), comme sur l’écran Prédictions.'
            : 'Clique sur une activité pour l’afficher seule. Clique sur plusieurs activités pour comparer plusieurs courbes.'}
        </Text>
      </View>
    </View>
  );
});

const AddCowSheetContent = memo(function AddCowSheetContent({
  onAddCow,
  onClose,
  onSuccess,
}) {
  const stableInternalId = useRef(generateInternalId()).current;

  const [name, setName] = useState('');
  const [breed, setBreed] = useState(BREED_OPTIONS[0]);
  const [sex, setSex] = useState('Femelle');
  const [birthDate, setBirthDate] = useState('2021-05-14');
  const [breedSheetOpen, setBreedSheetOpen] = useState(false);
  const [dateSheetOpen, setDateSheetOpen] = useState(false);

  const generated = useMemo(() => {
    const internalId = stableInternalId;
    const din = generateDIN(internalId);
    const age = calculateAge(birthDate);
    const weight = getEstimatedWeight(breed);

    return {
      internalId,
      din,
      age,
      weight,
    };
  }, [birthDate, breed, stableInternalId]);

  const handleAdd = () => {
    const fallbackId = String(Math.floor(Math.random() * 80) + 20).padStart(2, '0');

    const newCow = {
      id: fallbackId,
      internalId: generated.internalId,
      din: generated.din,
      name: name.trim() || `Vache #${fallbackId}`,
      breed,
      sex,
      birthDate,
      age: generated.age || 0,
      estimatedWeight: generated.weight,
      status: 'ok',
      statusLabel: 'OK',
      activity: 'Zone sécurisée',
      milkToday: 21,
      temp: 38.4,
    };

    onAddCow(newCow);
    onClose();
    onSuccess?.();
  };

  return (
    <>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={addSheet.content}
        keyboardShouldPersistTaps="handled"
      >
        <View style={addSheet.photoBox}>
          <Ionicons name="camera-outline" size={24} color={COLORS.primary} />
          <Text style={addSheet.photoText}>Ajouter une photo</Text>
        </View>

        <Text style={addSheet.sectionLabel}>Champs remplis par l’utilisateur</Text>

        <View style={addSheet.inputGroup}>
          <Text style={addSheet.label}>Nom / numéro</Text>

          <TextInput
            value={name}
            onChangeText={setName}
            placeholder="Ex : Bella"
            placeholderTextColor={COLORS.sage}
            style={addSheet.input}
          />
        </View>

        <View style={addSheet.inputGroup}>
          <Text style={addSheet.label}>Race</Text>

          <TouchableOpacity
            activeOpacity={0.84}
            style={addSheet.selectInput}
            onPress={() => setBreedSheetOpen(true)}
          >
            <Text style={addSheet.selectText}>{breed}</Text>
            <Ionicons name="chevron-down" size={18} color={COLORS.primary} />
          </TouchableOpacity>
        </View>

        <View style={addSheet.inputGroup}>
          <Text style={addSheet.label}>Sexe</Text>

          <View style={addSheet.sexRow}>
            {['Femelle', 'Mâle'].map((item) => (
              <TouchableOpacity
                key={item}
                onPress={() => setSex(item)}
                style={[addSheet.sexBtn, sex === item && addSheet.sexBtnActive]}
              >
                <Text style={[addSheet.sexText, sex === item && addSheet.sexTextActive]}>
                  {item}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={addSheet.inputGroup}>
          <Text style={addSheet.label}>Date de naissance</Text>

          <TouchableOpacity
            activeOpacity={0.84}
            style={addSheet.selectInput}
            onPress={() => setDateSheetOpen(true)}
          >
            <Text style={addSheet.selectText}>{birthDate}</Text>
            <Ionicons name="calendar-outline" size={18} color={COLORS.primary} />
          </TouchableOpacity>
        </View>

        <Text style={addSheet.sectionLabel}>Généré automatiquement</Text>

        <View style={addSheet.generatedBox}>
          <View style={addSheet.generatedRow}>
            <Text style={addSheet.generatedLabel}>ID interne</Text>
            <Text style={addSheet.generatedValue}>{generated.internalId}</Text>
          </View>

          <View style={addSheet.generatedRow}>
            <Text style={addSheet.generatedLabel}>DIN officiel</Text>
            <Text style={addSheet.generatedValue}>{generated.din}</Text>
          </View>

          <View style={addSheet.generatedRow}>
            <Text style={addSheet.generatedLabel}>Âge</Text>
            <Text style={addSheet.generatedValue}>{generated.age || 0} ans</Text>
          </View>

          <View style={addSheet.generatedRow}>
            <Text style={addSheet.generatedLabel}>Poids estimé</Text>
            <Text style={addSheet.generatedValue}>{generated.weight} kg</Text>
          </View>
        </View>

        <TouchableOpacity activeOpacity={0.86} style={addSheet.submitBtn} onPress={handleAdd}>
          <Ionicons name="checkmark" size={19} color={COLORS.white} />
          <Text style={addSheet.submitText}>Confirmer l’ajout</Text>
        </TouchableOpacity>
      </ScrollView>

      <BottomSheet
        visible={breedSheetOpen}
        title="Choisir la race"
        onClose={() => setBreedSheetOpen(false)}
        height={470}
      >
        <ScrollView showsVerticalScrollIndicator={false}>
          {BREED_OPTIONS.map((item) => {
            const active = breed === item;

            return (
              <TouchableOpacity
                key={item}
                activeOpacity={0.84}
                style={[addSheet.optionRow, active && addSheet.optionRowActive]}
                onPress={() => {
                  setBreed(item);
                  setBreedSheetOpen(false);
                }}
              >
                <Text style={[addSheet.optionText, active && addSheet.optionTextActive]}>
                  {item}
                </Text>

                {active && (
                  <Ionicons name="checkmark-circle" size={20} color={COLORS.primary} />
                )}
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      </BottomSheet>

      <BottomSheet
        visible={dateSheetOpen}
        title="Date de naissance"
        onClose={() => setDateSheetOpen(false)}
        height={500}
      >
        <CalendarPickerContent
          selectedDate={birthDate}
          maxYear={new Date().getFullYear()}
          onSelect={(date) => {
            setBirthDate(date);
            setDateSheetOpen(false);
          }}
        />
      </BottomSheet>
    </>
  );
});

function vocalEtatLabelFr(name) {
  if (!name) return '—';
  if (name === 'besoin_nourriture') return 'Besoin de nourriture';
  if (name === 'toux') return 'Toux';
  if (name === 'normal') return 'Normal';
  if (name === 'ovulation') return 'Ovulation';
  return name;
}

function formatVocalProbabilities(probabilities) {
  if (!probabilities || typeof probabilities !== 'object') return '';
  return Object.entries(probabilities)
    .map(([k, v]) => `${vocalEtatLabelFr(k)}: ${(Number(v) * 100).toFixed(0)} %`)
    .join(' · ');
}

const VoiceSheetContent = memo(function VoiceSheetContent({ cow }) {
  const pulse = useRef(new Animated.Value(1)).current;
  const [answer, setAnswer] = useState('');
  const [err, setErr] = useState('');
  const [vocalIsRecording, setVocalIsRecording] = useState(false);
  const [vocalBusy, setVocalBusy] = useState(false);
  const vocalRecordingRef = useRef(null);

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, {
          toValue: 1.12,
          duration: 650,
          useNativeDriver: true,
        }),
        Animated.timing(pulse, {
          toValue: 1,
          duration: 650,
          useNativeDriver: true,
        }),
      ])
    );

    loop.start();

    return () => loop.stop();
  }, [pulse]);

  useEffect(() => {
    return () => {
      const r = vocalRecordingRef.current;
      if (!r) return;
      vocalRecordingRef.current = null;
      r.stopAndUnloadAsync().catch(() => {});
    };
  }, []);

  const ensureMicPermission = async () => {
    const perm = await Audio.requestPermissionsAsync();
    if (perm.granted) return true;
    setErr(
      perm.status === 'denied'
        ? 'Permission micro refusée.'
        : 'Permission micro nécessaire pour enregistrer.'
    );
    return false;
  };

  const toggleMicAnalyze = async () => {
    if (vocalBusy) return;

    if (!vocalIsRecording) {
      setErr('');
      setAnswer('');
      const ok = await ensureMicPermission();
      if (!ok) return;
      try {
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: true,
          playsInSilentModeIOS: true,
        });
        const { recording } = await Audio.Recording.createAsync(
          Audio.RecordingOptionsPresets.HIGH_QUALITY
        );
        vocalRecordingRef.current = recording;
        setVocalIsRecording(true);
      } catch (e) {
        setErr(e?.message || 'Impossible de démarrer l’enregistrement');
        setVocalIsRecording(false);
        vocalRecordingRef.current = null;
      }
      return;
    }

    const rec = vocalRecordingRef.current;
    if (!rec) return;

    setVocalBusy(true);
    setErr('');
    try {
      await rec.stopAndUnloadAsync();
      const uri = rec.getURI();
      vocalRecordingRef.current = null;
      setVocalIsRecording(false);

      if (!uri) throw new Error('Fichier audio introuvable');

      const b64 = await recordingUriToBase64(uri);
      const out = await predictVocal(b64);

      if (out?.ok) {
        const etat = vocalEtatLabelFr(out.pred_vocal_name);
        const probs = formatVocalProbabilities(out.probabilities);
        setAnswer(
          `${cow.name} — état vocal (IA) : ${etat}.${probs ? `\n\n${probs}` : ''}`
        );
      } else {
        const msg =
          typeof out?.error === 'string' ? out.error : 'Prédiction vocale impossible (API).';
        setErr(msg);
        setAnswer('');
      }
    } catch (e) {
      setAnswer('');
      setErr(e?.response?.data?.error || e.message || 'Erreur réseau / audio');
    } finally {
      setVocalBusy(false);
      try {
        await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
      } catch {
        /* ignore */
      }
    }
  };

  const recordingHint =
    vocalIsRecording && !vocalBusy ? 'Réappuyez sur le micro pour envoyer au modèle' : '';

  return (
    <View style={voice.content}>
      <Text style={voice.desc}>
        Enregistrez le vocal de cette vache. Modèle vocal (mel Keras): toux, normal, ovulation,
        besoin de nourriture — même flux que Prédictions (API port 8008).
      </Text>

      <Animated.View style={[voice.micOuter, { transform: [{ scale: pulse }] }]}>
        <TouchableOpacity
          activeOpacity={0.86}
          style={[
            voice.micBtn,
            vocalIsRecording && voice.micBtnRecording,
            vocalBusy && { opacity: 0.82 },
          ]}
          onPress={toggleMicAnalyze}
          disabled={vocalBusy}
        >
          {vocalBusy ? (
            <ActivityIndicator color={COLORS.white} size="large" />
          ) : (
            <Ionicons name="mic" size={34} color={COLORS.white} />
          )}
        </TouchableOpacity>
      </Animated.View>

      <Text style={voice.hint}>
        {recordingHint || 'Premier tap = enregistrer · second tap = analyse IA'}
      </Text>
      {recordingHint ? (
        <Text style={[voice.subHint, { marginBottom: 10 }]}>
          Visez 1–3 s de vocal net pour une meilleure détection.
        </Text>
      ) : (
        <View style={{ height: 8 }} />
      )}

      {!!err && (
        <Text style={voice.errMsg} selectable>
          {err}
        </Text>
      )}

      {!!answer && (
        <View style={voice.answerBox}>
          <View style={voice.answerIcon}>
            <MaterialCommunityIcons name="pulse" size={20} color={COLORS.primary} />
          </View>

          <Text style={voice.answerText} selectable>
            {answer}
          </Text>
        </View>
      )}
    </View>
  );
});

const CowDetailModal = memo(function CowDetailModal({
  visible,
  cow,
  allCows,
  onClose,
  onSelectCow,
  liveRosette,
}) {
  const contentFade = useRef(new Animated.Value(1)).current;
  const slideAnim = useRef(new Animated.Value(0)).current;
  const prevCowId = useRef(null);

  const [activityPeriod, setActivityPeriod] = useState('day');
  const [calendarOpen, setCalendarOpen] = useState(false);
  const [voiceOpen, setVoiceOpen] = useState(false);
  const [selectedDate, setSelectedDate] = useState(formatDate(new Date()));

  const displayCow = useMemo(() => {
    if (!cow) return null;
    if (cow.id !== '12' || !liveRosette?.apiReachable) return cow;
    return {
      ...cow,
      milkToday: liveRosette.milkKg,
      temp: liveRosette.tempC,
      activity: liveRosette.stressLabelFr,
    };
  }, [cow, liveRosette]);

  const activityChartRows =
    cow?.id === '12' && liveRosette?.activityRows?.length ? liveRosette.activityRows : null;

  const activeCowIndex = useMemo(() => {
    if (!cow) return -1;
    return allCows.findIndex((item) => item.id === cow.id);
  }, [cow, allCows]);

  useEffect(() => {
    if (!visible || activeCowIndex < 0) return;

    Animated.spring(slideAnim, {
      toValue: activeCowIndex,
      useNativeDriver: true,
      tension: 80,
      friction: 12,
    }).start();
  }, [activeCowIndex, slideAnim, visible]);

  useEffect(() => {
    if (!visible) return;

    if (prevCowId.current === null) {
      prevCowId.current = cow?.id;
      return;
    }

    if (prevCowId.current !== cow?.id) {
      prevCowId.current = cow?.id;

      Animated.sequence([
        Animated.timing(contentFade, {
          toValue: 0,
          duration: 120,
          useNativeDriver: true,
        }),
        Animated.timing(contentFade, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        }),
      ]).start();
    }
  }, [contentFade, cow?.id, visible]);

  useEffect(() => {
    if (visible) {
      prevCowId.current = null;
      contentFade.setValue(1);
    }
  }, [contentFade, visible]);

  const indicatorTranslateY = useMemo(
    () =>
      slideAnim.interpolate({
        inputRange: allCows.map((_, index) => index),
        outputRange: allCows.map((_, index) => index * (THUMB_HEIGHT + THUMB_GAP)),
        extrapolate: 'clamp',
      }),
    [slideAnim, allCows]
  );

  const quickPills = useMemo(() => {
    if (!displayCow) return [];

    return [
      {
        icon: 'water-outline',
        label: `${displayCow.milkToday ?? 18} kg`,
        lib: 'mc',
      },
      {
        icon: 'thermometer',
        label: `${displayCow.temp ?? 38.6}°C`,
        lib: 'mc',
      },
      {
        icon: 'location-outline',
        label: displayCow.status === 'danger' ? 'Hors zone' : 'Zone OK',
        lib: 'io',
      },
    ];
  }, [displayCow]);

  const stats = useMemo(() => {
    if (!displayCow) return [];

    const { color } = getStatusStyle(displayCow.status);

    const rosetteStressLive = cow?.id === '12' && liveRosette?.apiReachable;
    let stressStatCol = COLORS.sage;
    if (rosetteStressLive && liveRosette?.stress && liveRosette.stress.ok !== false) {
      const k = Number(liveRosette.stress.pred_stress);
      if (k === 2) stressStatCol = '#C84C4C';
      else if (k === 1) stressStatCol = COLORS.warning;
      else if (k === 0) stressStatCol = COLORS.success;
    }

    const rosetteIllnessOk = cow?.id === '12' && liveRosette?.illness?.ok === true;
    let santeValue = displayCow.status === 'ok' ? 'Stable' : 'À suivre';
    let santeCol = color;
    if (rosetteIllnessOk && liveRosette.illness) {
      const v = illnessSanteValueFr(liveRosette.illness);
      if (v) santeValue = v;
      const hc = illnessSanteStatColor(liveRosette.illness.health_score);
      if (hc) santeCol = hc;
    }

    const stressOrActivityStat = rosetteStressLive
      ? {
          icon: 'pulse-outline',
          value: stressRiskLabelFr(liveRosette.stress),
          label: 'Niveau de stress',
          col: stressStatCol,
          lib: 'io',
        }
      : {
          icon: 'pulse-outline',
          value: displayCow.status === 'ok' ? 'Normale' : 'Faible',
          label: 'Activité',
          col: COLORS.warning,
          lib: 'io',
        };

    return [
      {
        icon: 'thermometer',
        value: `${displayCow.temp ?? 38.6}°C`,
        label: 'Température',
        col: color,
        lib: 'mc',
      },
      {
        icon: 'water-outline',
        value: `${displayCow.milkToday ?? 18} kg`,
        label: "Lait aujourd'hui",
        col: COLORS.primary,
        lib: 'mc',
      },
      stressOrActivityStat,
      {
        icon: 'shield-checkmark-outline',
        value: santeValue,
        label: 'Santé',
        col: santeCol,
        lib: 'io',
      },
    ];
  }, [displayCow, cow?.id, liveRosette]);

  if (!cow) return null;

  const { color, bg, label } = getStatusStyle(cow.status);

  return (
    <Modal visible={visible} animationType="slide" transparent={false} onRequestClose={onClose}>
      <SafeAreaView style={[detail.screen, { backgroundColor: color }]}>
        <View style={detail.container}>
          <View style={detail.rail}>
            <TouchableOpacity style={detail.railBackBtn} onPress={onClose}>
              <Ionicons name="chevron-back" size={22} color={COLORS.primary} />
            </TouchableOpacity>

            <ScrollView
              style={detail.railScroll}
              contentContainerStyle={detail.railThumbs}
              showsVerticalScrollIndicator={false}
            >
              <Animated.View
                style={[
                  detail.slidingPill,
                  {
                    backgroundColor: `${color}22`,
                    transform: [{ translateY: indicatorTranslateY }],
                  },
                ]}
              />

              {allCows.map((item) => {
                const isActive = item.id === cow.id;
                const { color: iconColor } = getStatusStyle(item.status);

                return (
                  <TouchableOpacity
                    key={item.id}
                    style={detail.railThumb}
                    onPress={() => onSelectCow(item)}
                    activeOpacity={0.82}
                  >
                    <View style={[detail.thumbDot, { backgroundColor: iconColor }]} />

                    <CowPhoto
                      status={item.status}
                      size={isActive ? 52 : 44}
                      cowId={item.id}
                    />

                    {isActive && (
                      <Animated.View
                        style={[detail.thumbActiveBar, { backgroundColor: iconColor }]}
                      />
                    )}
                  </TouchableOpacity>
                );
              })}
            </ScrollView>

            <TouchableOpacity style={detail.railArrow}>
              <Ionicons name="chevron-down" size={24} color={COLORS.sage} />
            </TouchableOpacity>
          </View>

          <Animated.View
            style={[
              detail.rightZone,
              {
                opacity: contentFade,
                backgroundColor: `${color}EE`,
              },
            ]}
          >
            <View style={detail.circleA} />
            <View style={detail.circleB} />

            <View style={detail.topBar}>
              <Text style={detail.headerTitle}>Détail</Text>

              <View style={detail.topActions}>
                <TouchableOpacity style={detail.iconBtn}>
                  <Ionicons name="search-outline" size={20} color={COLORS.cream} />
                </TouchableOpacity>

                <TouchableOpacity style={detail.iconBtnCircle}>
                  <Ionicons name="notifications-outline" size={19} color={COLORS.cream} />
                </TouchableOpacity>
              </View>
            </View>

            <View style={detail.identityBlock}>
              <AnimatedInfo key={`name-${cow.id}`} delay={60}>
                <Text style={detail.cowName}>{cow.name}</Text>
              </AnimatedInfo>

              <AnimatedInfo key={`sub-${cow.id}`} delay={130}>
                <Text style={detail.cowSub}>
                  #{cow.id} · {cow.breed} · {cow.age} ans
                </Text>
              </AnimatedInfo>

              <AnimatedInfo key={`badge-${cow.id}`} delay={200}>
                <View style={detail.statusBadge}>
                  <View style={detail.statusDot} />
                  <Text style={detail.statusBadgeText}>{cow.statusLabel || label}</Text>
                </View>
              </AnimatedInfo>

              <AnimatedInfo key={`pills-${cow.id}`} delay={270}>
                <View style={detail.heroPills}>
                  {quickPills.map((item) => (
                    <View key={`${item.icon}-${item.label}`} style={detail.heroPill}>
                      {item.lib === 'io' ? (
                        <Ionicons name={item.icon} size={15} color={COLORS.cream} />
                      ) : (
                        <MaterialCommunityIcons name={item.icon} size={15} color={COLORS.cream} />
                      )}

                      <Text style={detail.pillLabel}>{item.label}</Text>
                    </View>
                  ))}
                </View>
              </AnimatedInfo>

              <AnimatedInfo key={`desc-${cow.id}`} delay={340}>
                <Text style={detail.heroDesc}>
                  Suivi en temps réel de la santé, de la production laitière, de l’activité et de la position GPS.
                </Text>
              </AnimatedInfo>
            </View>

            <ScrollView
              style={detail.scrollPanel}
              contentContainerStyle={detail.panelContent}
              showsVerticalScrollIndicator={false}
            >
              <AnimatedInfo key={`summary-${cow.id}`} delay={420}>
                <Text style={detail.sectionTitle}>Résumé</Text>
              </AnimatedInfo>

              <View style={detail.grid}>
                {stats.map((item, index) => (
                  <AnimatedInfo
                    key={`${cow.id}-${item.label}-${item.value}`}
                    delay={500 + index * 80}
                    style={detail.statWrap}
                  >
                    <View style={detail.statCard}>
                      <View style={[detail.statIconBubble, { backgroundColor: `${item.col}12` }]}>
                        {item.lib === 'mc' ? (
                          <MaterialCommunityIcons name={item.icon} size={24} color={item.col} />
                        ) : (
                          <Ionicons name={item.icon} size={24} color={item.col} />
                        )}
                      </View>

                      <Text style={detail.statValue}>{item.value}</Text>
                      <Text style={detail.statLabel}>{item.label}</Text>
                    </View>
                  </AnimatedInfo>
                ))}
              </View>

              <TouchableOpacity
                activeOpacity={0.86}
                style={detail.talkBtn}
                onPress={() => setVoiceOpen(true)}
              >
                <View style={detail.talkIcon}>
                  <Ionicons name="mic" size={19} color={COLORS.primary} />
                </View>

                <View style={{ flex: 1 }}>
                  <Text style={detail.talkTitle}>Parle-moi</Text>
                  <Text style={detail.talkText}>
                    Vocal + classification IA du son (toux, normal, ovulation, nourriture)
                  </Text>
                </View>

                <Ionicons name="chevron-forward" size={18} color={COLORS.primary} />
              </TouchableOpacity>

              <View style={[detail.alertCard, { borderLeftColor: color }]}>
                <View style={detail.alertHeader}>
                  <View style={[detail.alertIcon, { backgroundColor: bg }]}>
                    <Ionicons name="location-outline" size={18} color={color} />
                  </View>

                  <View style={detail.alertTextWrap}>
                    <Text style={detail.alertTitle}>Suivi GPS</Text>
                    <Text style={detail.alertText}>
                      {cow.status === 'danger'
                        ? 'Dernière position détectée hors zone sécurisée.'
                        : 'Position stable dans la zone de pâturage.'}
                    </Text>
                  </View>
                </View>

                <TouchableOpacity style={[detail.btn, { backgroundColor: color }]}>
                  <Text style={detail.btnText}>Voir l’itinéraire</Text>
                  <Ionicons name="arrow-forward" size={16} color={COLORS.white} />
                </TouchableOpacity>
              </View>

              <View style={detail.activityHeader}>
                <View>
                  <Text style={detail.sectionTitleSmall}>Activité</Text>
                  <Text style={detail.activityDate}>{selectedDate}</Text>
                </View>

                <TouchableOpacity
                  activeOpacity={0.84}
                  style={detail.calendarBtn}
                  onPress={() => setCalendarOpen(true)}
                >
                  <Ionicons name="calendar-outline" size={18} color={COLORS.primary} />
                </TouchableOpacity>
              </View>

              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={detail.periodRow}
              >
                {PERIODS.map((item) => (
                  <TouchableOpacity
                    key={item.key}
                    activeOpacity={0.82}
                    onPress={() => setActivityPeriod(item.key)}
                    style={[
                      detail.periodBtn,
                      activityPeriod === item.key && detail.periodBtnActive,
                    ]}
                  >
                    <Text
                      style={[
                        detail.periodText,
                        activityPeriod === item.key && detail.periodTextActive,
                      ]}
                    >
                      {item.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>

              <ActivityChart period={activityPeriod} overrideSeries={activityChartRows} />

              <TouchableOpacity style={[detail.btn, detail.vetBtn]}>
                <MaterialCommunityIcons name="stethoscope" size={17} color={COLORS.white} />
                <Text style={detail.btnText}>Contacter vétérinaire</Text>
              </TouchableOpacity>
            </ScrollView>
          </Animated.View>
        </View>

        <BottomSheet
          visible={calendarOpen}
          title="Calendrier"
          onClose={() => setCalendarOpen(false)}
          height={500}
        >
          <CalendarPickerContent
            selectedDate={selectedDate}
            onSelect={(date) => {
              setSelectedDate(date);
              setCalendarOpen(false);
            }}
          />
        </BottomSheet>

        <BottomSheet
          visible={voiceOpen}
          title="Parle-moi"
          onClose={() => setVoiceOpen(false)}
          height={540}
        >
          <VoiceSheetContent cow={cow} />
        </BottomSheet>
      </SafeAreaView>
    </Modal>
  );
});

export default function HerdScreen() {
  const farmLoc = useFarmLocation('fr');

  const [cows, setCows] = useState(initialCows);
  const [liveRosette, setLiveRosette] = useState(null);
  const [query, setQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');
  const [selectedCow, setSelectedCow] = useState(null);
  const [addSheetOpen, setAddSheetOpen] = useState(false);
  const [snackbarVisible, setSnackbarVisible] = useState(false);

  const normalizedQuery = useMemo(() => query.trim().toLowerCase(), [query]);

  const filteredCows = useMemo(() => {
    return cows.filter((cow) => {
      const matchFilter = activeFilter === 'all' || cow.status === activeFilter;

      if (!matchFilter) return false;
      if (!normalizedQuery) return true;

      return (
        cow.name?.toLowerCase().includes(normalizedQuery) ||
        cow.id?.toLowerCase().includes(normalizedQuery) ||
        cow.breed?.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [activeFilter, normalizedQuery, cows]);

  const showSuccessSnackbar = useCallback(() => {
    setSnackbarVisible(true);

    setTimeout(() => {
      setSnackbarVisible(false);
    }, 2400);
  }, []);

  const handleAddCow = useCallback((newCow) => {
    setCows((prev) => [newCow, ...prev]);
  }, []);

  const refreshRosette = useCallback(async () => {
    try {
      const b = await fetchRosetteLiveBundle('12');
      setLiveRosette(b);
      setCows((prev) =>
        prev.map((c) =>
          c.id === '12'
            ? { ...c, milkToday: b.milkKg, temp: b.tempC, activity: b.stressLabelFr }
            : c
        )
      );
    } catch {
      /* garde les dernières valeurs liveRosette / liste */
    }
  }, []);

  useEffect(() => {
    refreshRosette();
    const id = setInterval(refreshRosette, 35000);
    return () => clearInterval(id);
  }, [refreshRosette]);

  const renderCow = useCallback(
    ({ item, index }) => (
      <CowCard cow={item} index={index} onPress={setSelectedCow} />
    ),
    []
  );

  const listHeader = useMemo(
    () => (
      <View>
        <View style={s.header}>
          <TouchableOpacity style={s.menuBtn}>
            <Ionicons name="menu-outline" size={28} color={COLORS.primary} />
          </TouchableOpacity>

          <View style={s.locBlock}>
            <Text style={s.locLabel}>Location</Text>

            <View style={s.locRow}>
              <Ionicons name="location" size={13} color={COLORS.primary} />
              <Text style={s.locText} numberOfLines={2}>
                {farmLoc.loading ? 'Localisation…' : farmLoc.label || '—'}
              </Text>
            </View>
          </View>

          <TouchableOpacity style={s.avatarBtn}>
            <MaterialCommunityIcons name="cow" size={20} color={COLORS.primary} />
          </TouchableOpacity>
        </View>

        <Text style={s.title}>My Herd</Text>
        <Text style={s.subtitle}>{cows.length} vaches suivies</Text>

        <View style={s.searchRow}>
          <View style={s.searchBox}>
            <Ionicons name="search-outline" size={18} color={COLORS.sage} />

            <TextInput
              value={query}
              onChangeText={setQuery}
              placeholder="Rechercher une vache…"
              placeholderTextColor={COLORS.sage}
              style={s.searchInput}
              autoCorrect={false}
              autoCapitalize="none"
            />
          </View>

          <TouchableOpacity style={s.filterBtn}>
            <Ionicons name="options-outline" size={20} color={COLORS.primary} />
          </TouchableOpacity>
        </View>

        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={s.pillsRow}
        >
          {FILTERS.map((filter) => (
            <FilterPill
              key={filter.key}
              item={filter}
              active={activeFilter === filter.key}
              onPress={setActiveFilter}
            />
          ))}
        </ScrollView>

        <View style={s.listHeader}>
          <Text style={s.listTitle}>Toutes les vaches</Text>
          <Text style={s.listCount}>{filteredCows.length}</Text>
        </View>
      </View>
    ),
    [activeFilter, filteredCows.length, query, cows.length, farmLoc.loading, farmLoc.label]
  );

  return (
    <SafeAreaView style={s.safe}>
      <FlatList
        data={filteredCows}
        renderItem={renderCow}
        keyExtractor={(item) => item.id}
        ListHeaderComponent={listHeader}
        ListEmptyComponent={
          <View style={s.empty}>
            <Ionicons name="search-outline" size={28} color={COLORS.sage} />
            <Text style={s.emptyTitle}>Aucune vache trouvée</Text>
            <Text style={s.emptyText}>Essayez un autre filtre ou une autre recherche.</Text>
          </View>
        }
        ItemSeparatorComponent={() => <View style={s.separator} />}
        contentContainerStyle={s.content}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        initialNumToRender={12}
        maxToRenderPerBatch={12}
        windowSize={9}
        removeClippedSubviews={Platform.OS === 'android'}
      />

      <TouchableOpacity
        style={s.fab}
        activeOpacity={0.85}
        onPress={() => setAddSheetOpen(true)}
      >
        <Ionicons name="add" size={28} color={COLORS.white} />
      </TouchableOpacity>

      <CowDetailModal
        visible={!!selectedCow}
        cow={selectedCow}
        allCows={cows}
        onClose={() => setSelectedCow(null)}
        onSelectCow={setSelectedCow}
        liveRosette={liveRosette}
      />

      <BottomSheet
        visible={addSheetOpen}
        title="Ajouter une vache"
        onClose={() => setAddSheetOpen(false)}
        height={680}
      >
        <AddCowSheetContent
          onAddCow={handleAddCow}
          onClose={() => setAddSheetOpen(false)}
          onSuccess={showSuccessSnackbar}
        />
      </BottomSheet>

      {snackbarVisible && (
        <View style={s.snackbar}>
          <Ionicons name="checkmark-circle" size={20} color={COLORS.white} />
          <Text style={s.snackbarText}>Vache ajoutée</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const sheet = StyleSheet.create({
  overlay: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.34)',
  },
  container: {
    backgroundColor: COLORS.cream,
    borderTopLeftRadius: 30,
    borderTopRightRadius: 30,
    paddingHorizontal: 18,
    paddingTop: 10,
    paddingBottom: Platform.OS === 'ios' ? 28 : 18,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -10 },
    shadowOpacity: 0.16,
    shadowRadius: 22,
    elevation: 16,
  },
  handle: {
    alignSelf: 'center',
    width: 44,
    height: 5,
    borderRadius: 99,
    backgroundColor: COLORS.border,
    marginBottom: 14,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 14,
  },
  title: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 22,
    color: COLORS.black,
    letterSpacing: -0.6,
  },
  closeBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
});

const photoStyles = StyleSheet.create({
  wrap: {
    borderRadius: 36,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.78)',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 18 },
    shadowOpacity: 0.18,
    shadowRadius: 26,
    elevation: 9,
  },
  glow: {
    position: 'absolute',
    width: 130,
    height: 130,
    borderRadius: 65,
    backgroundColor: 'rgba(255,255,255,0.26)',
    top: -30,
    right: -30,
  },
  cowImage: {
    width: '105%',
    height: '105%',
  },
});

const pill = StyleSheet.create({
  wrap: {
    flexDirection: 'column',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 14,
    paddingVertical: 14,
    borderRadius: 20,
    backgroundColor: COLORS.white,
    borderWidth: 1.2,
    borderColor: COLORS.border,
    minWidth: 68,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.07,
    shadowRadius: 10,
    elevation: 3,
  },
  active: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  label: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 11,
    color: COLORS.dark,
  },
  labelActive: {
    color: COLORS.white,
  },
});

const card = StyleSheet.create({
  outer: {
    minHeight: 192,
    flexDirection: 'row',
    alignItems: 'center',
    position: 'relative',
    paddingLeft: 0,
  },
  photoBlock: {
    width: 160,
    height: 186,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 2,
  },
  infoCard: {
    flex: 1,
    minHeight: 112,
    marginLeft: -4,
    backgroundColor: COLORS.white,
    borderRadius: 26,
    paddingVertical: 12,
    paddingLeft: 16,
    paddingRight: 14,
    borderWidth: 1,
    borderColor: COLORS.border,
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 18,
    elevation: 4,
    zIndex: 1,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 8,
  },
  name: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 20,
    color: COLORS.black,
    letterSpacing: -0.6,
  },
  breed: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.dark,
    marginTop: 2,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginTop: 6,
  },
  meta: {
    fontFamily: 'Inter_500Medium',
    fontSize: 11.5,
    color: COLORS.dark,
    flex: 1,
  },
  bottomRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 9,
    gap: 8,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    flex: 1,
  },
  dot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  statusText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 10.5,
  },
  milkPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: COLORS.cream,
  },
  milkText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 10.5,
    color: COLORS.primary,
  },
});

const detail = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: COLORS.white,
  },
  container: {
    flex: 1,
    flexDirection: 'row',
  },
  rail: {
    width: 88,
    backgroundColor: COLORS.white,
    borderTopRightRadius: 28,
    borderBottomRightRadius: 28,
    paddingVertical: 18,
    paddingHorizontal: 6,
    alignItems: 'center',
    justifyContent: 'space-between',
    zIndex: 20,
    shadowColor: '#000',
    shadowOffset: { width: 8, height: 0 },
    shadowOpacity: 0.12,
    shadowRadius: 20,
    elevation: 12,
  },
  railBackBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: COLORS.cream,
    alignItems: 'center',
    justifyContent: 'center',
  },
  railScroll: {
    flex: 1,
    width: '100%',
  },
  railThumbs: {
    alignItems: 'center',
    gap: THUMB_GAP,
    paddingVertical: 10,
    position: 'relative',
  },
  slidingPill: {
    position: 'absolute',
    top: 10,
    width: THUMB_HEIGHT,
    height: THUMB_HEIGHT,
    borderRadius: 22,
    zIndex: 0,
  },
  railThumb: {
    width: THUMB_HEIGHT,
    height: THUMB_HEIGHT,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'transparent',
    position: 'relative',
    zIndex: 1,
  },
  thumbDot: {
    position: 'absolute',
    top: 10,
    right: 10,
    width: 7,
    height: 7,
    borderRadius: 4,
    zIndex: 3,
  },
  thumbActiveBar: {
    position: 'absolute',
    right: -6,
    top: '25%',
    bottom: '25%',
    width: 4,
    borderRadius: 2,
  },
  railArrow: {
    width: 42,
    height: 42,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rightZone: {
    flex: 1,
    minWidth: 0,
    position: 'relative',
    overflow: 'hidden',
  },
  circleA: {
    position: 'absolute',
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: 'rgba(255,255,255,0.07)',
    top: -70,
    right: -70,
    zIndex: 0,
  },
  circleB: {
    position: 'absolute',
    width: 130,
    height: 130,
    borderRadius: 65,
    backgroundColor: 'rgba(0,0,0,0.10)',
    left: -50,
    top: 100,
    zIndex: 0,
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingTop: 16,
    zIndex: 10,
  },
  headerTitle: {
    fontFamily: 'PlusJakartaSans_700Bold',
    fontSize: 17,
    color: 'rgba(246,246,236,0.90)',
  },
  topActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  iconBtn: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconBtnCircle: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: 'rgba(246,246,236,0.16)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  identityBlock: {
    paddingHorizontal: 16,
    paddingTop: 10,
    zIndex: 10,
  },
  cowName: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 34,
    lineHeight: 38,
    color: '#E9E2B8',
    letterSpacing: -1,
  },
  cowSub: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 13,
    color: 'rgba(246,246,236,0.80)',
    marginTop: 5,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    alignSelf: 'flex-start',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    marginTop: 8,
    backgroundColor: 'rgba(255,255,255,0.18)',
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: COLORS.white,
  },
  statusBadgeText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 12,
    color: COLORS.white,
  },
  heroPills: {
    flexDirection: 'row',
    gap: 8,
    marginTop: 12,
    flexWrap: 'wrap',
  },
  heroPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(246,246,236,0.35)',
    backgroundColor: 'rgba(255,255,255,0.08)',
  },
  pillLabel: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.cream,
  },
  heroDesc: {
    fontFamily: 'Inter_500Medium',
    fontSize: 13,
    lineHeight: 19,
    color: 'rgba(246,246,236,0.72)',
    marginTop: 10,
  },
  scrollPanel: {
    flex: 1,
    backgroundColor: COLORS.cream,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    marginTop: 14,
  },
  panelContent: {
    paddingHorizontal: 16,
    paddingTop: 18,
    paddingBottom: 52,
  },
  sectionTitle: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 22,
    color: COLORS.black,
    marginBottom: 14,
  },
  sectionTitleSmall: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 21,
    color: COLORS.black,
    letterSpacing: -0.5,
  },
  activityDate: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.sage,
    marginTop: 3,
  },
  activityHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  calendarBtn: {
    width: 42,
    height: 42,
    borderRadius: 16,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  periodRow: {
    gap: 8,
    paddingBottom: 12,
  },
  periodBtn: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 999,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  periodBtnActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  periodText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 12,
    color: COLORS.primary,
  },
  periodTextActive: {
    color: COLORS.white,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 18,
  },
  statWrap: {
    width: '48%',
    marginBottom: 12,
  },
  statCard: {
    width: '100%',
    minHeight: 124,
    backgroundColor: COLORS.white,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingHorizontal: 14,
    paddingVertical: 14,
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.08,
    shadowRadius: 14,
    elevation: 4,
  },
  statIconBubble: {
    width: 42,
    height: 42,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  statValue: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 20,
    color: COLORS.black,
    marginTop: 12,
  },
  statLabel: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.sage,
    marginTop: 4,
    lineHeight: 16,
  },
  talkBtn: {
    backgroundColor: COLORS.white,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 12,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.07,
    shadowRadius: 12,
    elevation: 3,
  },
  talkIcon: {
    width: 42,
    height: 42,
    borderRadius: 16,
    backgroundColor: COLORS.cream,
    alignItems: 'center',
    justifyContent: 'center',
  },
  talkTitle: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 15,
    color: COLORS.black,
  },
  talkText: {
    fontFamily: 'Inter_500Medium',
    fontSize: 12,
    color: COLORS.sage,
    marginTop: 3,
  },
  alertCard: {
    backgroundColor: COLORS.white,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderLeftWidth: 5,
    padding: 13,
    marginBottom: 16,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.07,
    shadowRadius: 12,
    elevation: 3,
  },
  alertHeader: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 10,
  },
  alertIcon: {
    width: 38,
    height: 38,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  alertTextWrap: {
    flex: 1,
  },
  alertTitle: {
    fontFamily: 'PlusJakartaSans_700Bold',
    fontSize: 14,
    color: COLORS.black,
    marginBottom: 3,
  },
  alertText: {
    fontFamily: 'Inter_500Medium',
    fontSize: 12,
    color: COLORS.dark,
    lineHeight: 17,
  },
  btn: {
    height: 46,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
  },
  btnText: {
    fontFamily: 'Inter_700Bold',
    color: COLORS.white,
    fontSize: 13,
  },
  vetBtn: {
    backgroundColor: COLORS.earth,
    marginTop: 12,
  },
});

const activity = StyleSheet.create({
  card: {
    backgroundColor: COLORS.white,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 14,
    marginBottom: 10,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 14,
    elevation: 4,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
    alignItems: 'flex-start',
  },
  kicker: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 10,
    letterSpacing: 1.4,
    color: COLORS.earth,
  },
  title: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 17,
    color: COLORS.black,
    marginTop: 4,
    letterSpacing: -0.4,
  },
  totalBadge: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: COLORS.cream,
  },
  totalText: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 11,
    color: COLORS.primary,
  },
  lineLegendWrap: {
    gap: 8,
    paddingTop: 14,
    paddingBottom: 12,
  },
  lineLegendBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 999,
    backgroundColor: COLORS.cream,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  legendDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  lineLegendText: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 10,
    color: COLORS.dark,
  },
  curveBox: {
    height: 198,
    borderRadius: 22,
    backgroundColor: COLORS.cream,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    marginTop: 4,
  },
  curveGridLineTop: {
    position: 'absolute',
    left: 16,
    right: 16,
    top: 44,
    height: 1,
    backgroundColor: 'rgba(46,65,45,0.08)',
  },
  curveGridLineMid: {
    position: 'absolute',
    left: 16,
    right: 16,
    top: 96,
    height: 1,
    backgroundColor: 'rgba(46,65,45,0.08)',
  },
  curveGridLineBottom: {
    position: 'absolute',
    left: 16,
    right: 16,
    bottom: 44,
    height: 1,
    backgroundColor: 'rgba(46,65,45,0.08)',
  },
  curveLine: {
    position: 'absolute',
    height: 2.8,
    borderRadius: 999,
  },
  curvePointTouch: {
    position: 'absolute',
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  curvePoint: {
    width: 9,
    height: 9,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: COLORS.white,
  },
  curveXAxisLabel: {
    position: 'absolute',
    fontFamily: 'Inter_700Bold',
    fontSize: 9,
    color: COLORS.sage,
    maxWidth: 56,
    textAlign: 'center',
  },
  selectedPanel: {
    marginTop: 12,
    backgroundColor: COLORS.cream,
    borderRadius: 18,
    padding: 12,
  },
  selectedTitle: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 14,
    color: COLORS.black,
    marginBottom: 6,
  },
  curveDesc: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.dark,
    lineHeight: 18,
  },
});

const addSheet = StyleSheet.create({
  content: {
    paddingBottom: 32,
  },
  photoBox: {
    height: 94,
    borderRadius: 24,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    marginBottom: 16,
  },
  photoText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 13,
    color: COLORS.primary,
  },
  sectionLabel: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 11,
    letterSpacing: 1.2,
    color: COLORS.earth,
    marginBottom: 10,
    marginTop: 6,
  },
  inputGroup: {
    marginBottom: 12,
  },
  label: {
    fontFamily: 'Inter_700Bold',
    fontSize: 12,
    color: COLORS.dark,
    marginBottom: 7,
  },
  input: {
    height: 48,
    borderRadius: 16,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingHorizontal: 14,
    fontFamily: 'Inter_600SemiBold',
    fontSize: 13,
    color: COLORS.black,
  },
  selectInput: {
    height: 48,
    borderRadius: 16,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingHorizontal: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
  },
  selectText: {
    flex: 1,
    fontFamily: 'Inter_700Bold',
    fontSize: 13,
    color: COLORS.black,
  },
  sexRow: {
    flexDirection: 'row',
    gap: 10,
  },
  sexBtn: {
    flex: 1,
    height: 46,
    borderRadius: 16,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sexBtnActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  sexText: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 12,
    color: COLORS.primary,
  },
  sexTextActive: {
    color: COLORS.white,
  },
  generatedBox: {
    borderRadius: 22,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 14,
    gap: 10,
    marginBottom: 18,
  },
  generatedRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  generatedLabel: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.sage,
  },
  generatedValue: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 12,
    color: COLORS.black,
  },
  submitBtn: {
    height: 52,
    borderRadius: 18,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    gap: 8,
  },
  submitText: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 14,
    color: COLORS.white,
  },
  optionRow: {
    minHeight: 50,
    borderRadius: 16,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingHorizontal: 14,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  optionRowActive: {
    backgroundColor: 'rgba(46,65,45,0.08)',
    borderColor: COLORS.primary,
  },
  optionText: {
    flex: 1,
    fontFamily: 'Inter_700Bold',
    fontSize: 13,
    color: COLORS.dark,
  },
  optionTextActive: {
    color: COLORS.primary,
  },
});

const calendar = StyleSheet.create({
  content: {
    paddingTop: 2,
  },
  monthHeader: {
    height: 46,
    borderRadius: 18,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 8,
    marginBottom: 14,
  },
  monthBtn: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.cream,
  },
  monthTitle: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 15,
    color: COLORS.black,
    textTransform: 'capitalize',
  },
  weekRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  weekText: {
    width: `${100 / 7}%`,
    textAlign: 'center',
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 11,
    color: COLORS.sage,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    backgroundColor: COLORS.white,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 8,
  },
  dayCell: {
    width: `${100 / 7}%`,
    height: 42,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 14,
  },
  dayCellActive: {
    backgroundColor: COLORS.primary,
  },
  dayCellToday: {
    backgroundColor: 'rgba(46,65,45,0.08)',
  },
  dayText: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 12,
    color: COLORS.black,
  },
  dayTextActive: {
    color: COLORS.white,
  },
  infoBox: {
    marginTop: 14,
    backgroundColor: COLORS.white,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  infoText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 12,
    color: COLORS.dark,
  },
});

const voice = StyleSheet.create({
  content: {
    alignItems: 'center',
    paddingTop: 8,
  },
  desc: {
    fontFamily: 'Inter_500Medium',
    fontSize: 13,
    color: COLORS.dark,
    textAlign: 'center',
    lineHeight: 19,
    marginBottom: 30,
  },
  micOuter: {
    width: 104,
    height: 104,
    borderRadius: 52,
    backgroundColor: 'rgba(46,65,45,0.10)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  micBtn: {
    width: 78,
    height: 78,
    borderRadius: 39,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.24,
    shadowRadius: 16,
    elevation: 8,
  },
  micBtnRecording: {
    backgroundColor: COLORS.warning,
    shadowColor: COLORS.warning,
  },
  hint: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.sage,
    marginTop: 24,
    marginBottom: 18,
    textAlign: 'center',
  },
  subHint: {
    fontFamily: 'Inter_500Medium',
    fontSize: 11,
    color: COLORS.sage,
    opacity: 0.9,
    textAlign: 'center',
    alignSelf: 'stretch',
    paddingHorizontal: 8,
    lineHeight: 16,
    marginBottom: 6,
  },
  errMsg: {
    width: '100%',
    marginTop: 8,
    marginBottom: 4,
    fontFamily: 'Inter_600SemiBold',
    fontSize: 11,
    color: COLORS.danger,
    textAlign: 'center',
    lineHeight: 16,
  },
  answerBox: {
    width: '100%',
    backgroundColor: COLORS.white,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: 14,
    flexDirection: 'row',
    gap: 10,
  },
  answerIcon: {
    width: 38,
    height: 38,
    borderRadius: 14,
    backgroundColor: COLORS.cream,
    alignItems: 'center',
    justifyContent: 'center',
  },
  answerText: {
    flex: 1,
    fontFamily: 'Inter_600SemiBold',
    fontSize: 13,
    color: COLORS.dark,
    lineHeight: 19,
  },
});

const s = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.cream,
  },
  content: {
    paddingHorizontal: 18,
    paddingTop: 10,
    paddingBottom: 120,
  },
  header: {
    height: 60,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  menuBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  locBlock: {
    alignItems: 'center',
  },
  locLabel: {
    fontFamily: 'Inter_500Medium',
    fontSize: 11,
    color: COLORS.sage,
    marginBottom: 3,
  },
  locRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  locText: {
    fontFamily: 'PlusJakartaSans_700Bold',
    fontSize: 14,
    color: COLORS.black,
  },
  avatarBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 10,
    elevation: 3,
  },
  title: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 32,
    color: COLORS.black,
    letterSpacing: -1,
    marginTop: 6,
  },
  subtitle: {
    fontFamily: 'Inter_500Medium',
    fontSize: 13,
    color: COLORS.sage,
    marginTop: 3,
    marginBottom: 18,
  },
  searchRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 20,
  },
  searchBox: {
    flex: 1,
    height: 52,
    backgroundColor: COLORS.white,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: COLORS.border,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 18,
    gap: 10,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.06,
    shadowRadius: 12,
    elevation: 2,
  },
  searchInput: {
    flex: 1,
    fontFamily: 'Inter_500Medium',
    fontSize: 14,
    color: COLORS.black,
  },
  filterBtn: {
    width: 52,
    height: 52,
    borderRadius: 20,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.06,
    shadowRadius: 12,
    elevation: 2,
  },
  pillsRow: {
    gap: 12,
    paddingRight: 8,
    marginBottom: 24,
  },
  listHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  listTitle: {
    fontFamily: 'PlusJakartaSans_700Bold',
    fontSize: 19,
    color: COLORS.black,
  },
  listCount: {
    fontFamily: 'Inter_700Bold',
    fontSize: 13,
    color: COLORS.primary,
  },
  separator: {
    height: 20,
  },
  empty: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 44,
    paddingHorizontal: 24,
  },
  emptyTitle: {
    fontFamily: 'PlusJakartaSans_700Bold',
    fontSize: 17,
    color: COLORS.black,
    marginTop: 12,
  },
  emptyText: {
    fontFamily: 'Inter_500Medium',
    fontSize: 13,
    color: COLORS.sage,
    marginTop: 4,
    textAlign: 'center',
  },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: Platform.OS === 'ios' ? 106 : 96,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.28,
    shadowRadius: 18,
    elevation: 10,
  },
  snackbar: {
    position: 'absolute',
    left: 18,
    right: 18,
    bottom: Platform.OS === 'ios' ? 38 : 28,
    minHeight: 52,
    borderRadius: 18,
    backgroundColor: COLORS.primary,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.26,
    shadowRadius: 16,
    elevation: 12,
  },
  snackbarText: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 14,
    color: COLORS.white,
  },
});