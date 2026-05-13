import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  SafeAreaView,
  Animated,
  Dimensions,
  Pressable,
  ActivityIndicator,
} from 'react-native';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import {
  buildStressAlertItems,
  fetchHomeDashboardModelPayload,
  isStressAlertActive,
  stressHealthPctDisplay,
} from '../services/liveFarmModels';
import {
  fetchOpenMeteoCurrent,
  weatherCodeToIonName,
} from '../services/openMeteo';
import { useFarmLocation } from '../hooks/useFarmLocation';

const { width } = Dimensions.get('window');

const COLORS = {
  primary: '#2E412D',
  primaryDeep: '#1D2B1D',
  primarySoft: 'rgba(46,65,45,0.14)',
  green2: '#5A8B64',
  green3: '#89A98F',
  emerald: '#2c5140',
  earth: '#7C6144',
  gold: '#B98A32',
  olive: '#A19F78',
  cream: '#f6f6ec',
  cream2: '#EEF2E8',
  sage: '#AFB2A1',
  darkOlive: '#3E3E32',
  black: '#10130F',
  white: '#FFFFFF',
  success: '#4F8F63',
  warning: '#D9962F',
  danger: '#C84C4C',
  info: '#4E78A8',
  border: '#E5E3D6',
};

const HERO_CARD_WIDTH = width * 0.64;
const HERO_CARD_SPACING = -14;
const HERO_SIDE_PADDING = (width - HERO_CARD_WIDTH) / 2 - 18;
const HERO_SNAP = HERO_CARD_WIDTH + HERO_CARD_SPACING;

const ALERT_CARD_WIDTH = width * 0.76;
const ALERT_CARD_SPACING = 18;
const ALERT_SIDE_PADDING = (width - ALERT_CARD_WIDTH) / 2;
const ALERT_SNAP = ALERT_CARD_WIDTH + ALERT_CARD_SPACING;

const FALLBACK_MILK_SERIES = [176, 184, 181, 190, 186, 180, 186];
const FALLBACK_MILK_LABELS = ['L', 'M', 'M', 'J', 'V', 'S', 'D'];

const STATIC_HERD_COUNT = 24;

function buildHeroSlides({
  herdCount,
  milkKgRounded,
  healthPct,
  alertCount,
}) {
  return [
    {
      id: 'overview',
      title: 'Tableau',
      value: String(herdCount),
      label: 'Vaches',
      badge: 'Troupeau',
      icon: 'cow',
      type: 'mc',
      color: COLORS.primary,
    },
    {
      id: 'milk',
      title: 'Lait',
      value: `${milkKgRounded} kg`,
      label: 'Lait / jour',
      badge: 'Production',
      icon: 'water-outline',
      type: 'ion',
      color: COLORS.info,
    },
    {
      id: 'health',
      title: 'Santé',
      value: healthPct,
      label: 'Indice de santé',
      badge: 'Troupeau',
      icon: 'heart-pulse',
      type: 'mc',
      color: COLORS.success,
    },
    {
      id: 'alerts',
      title: 'Alertes',
      value: String(Math.min(Math.max(alertCount, 0), 99)),
      label: 'À traiter',
      badge: 'Urgent',
      icon: 'warning-outline',
      type: 'ion',
      color: COLORS.warning,
    },
  ];
}

/** Alertes non liées au stress (Rosette utilise les données IoT / modèle pour le stress). */
const alertCarouselFallback = [
  {
    id: '1',
    cow: 'Marguerite #07',
    title: 'Hors zone sécurisée',
    description: 'Dernière position détectée hors de la zone de pâturage.',
    time: 'Il y a 12 min',
    status: 'Urgent',
    color: COLORS.danger,
    icon: 'location-outline',
    type: 'ion',
  },
  {
    id: '3',
    cow: 'Collier #03',
    title: 'Batterie faible',
    description: 'Le collier GPS présente un niveau de batterie faible.',
    time: 'Il y a 1h',
    status: 'À charger',
    color: COLORS.warning,
    icon: 'battery-dead-outline',
    type: 'ion',
  },
];

function AnimatedCard({ children, delay = 0, style }) {
  const fade = useRef(new Animated.Value(0)).current;
  const slide = useRef(new Animated.Value(18)).current;
  const scale = useRef(new Animated.Value(0.98)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fade, {
        toValue: 1,
        duration: 480,
        delay,
        useNativeDriver: true,
      }),
      Animated.spring(slide, {
        toValue: 0,
        delay,
        friction: 8,
        tension: 80,
        useNativeDriver: true,
      }),
      Animated.spring(scale, {
        toValue: 1,
        delay,
        friction: 8,
        tension: 80,
        useNativeDriver: true,
      }),
    ]).start();
  }, [fade, slide, scale, delay]);

  return (
    <Animated.View
      style={[
        style,
        {
          opacity: fade,
          transform: [{ translateY: slide }, { scale }],
        },
      ]}
    >
      {children}
    </Animated.View>
  );
}

function RevealOnScroll({
  children,
  scrollY,
  viewportHeight,
  offset = 90,
  style,
  onEnter,
}) {
  const [layout, setLayout] = useState({ y: null, height: 0 });
  const [visible, setVisible] = useState(false);

  const opacity = useRef(new Animated.Value(0)).current;
  const translateY = useRef(new Animated.Value(42)).current;
  const scale = useRef(new Animated.Value(0.965)).current;

  const animateIn = () => {
    Animated.parallel([
      Animated.timing(opacity, {
        toValue: 1,
        duration: 430,
        useNativeDriver: true,
      }),
      Animated.spring(translateY, {
        toValue: 0,
        friction: 8,
        tension: 70,
        useNativeDriver: true,
      }),
      Animated.spring(scale, {
        toValue: 1,
        friction: 8,
        tension: 70,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const animateOut = () => {
    Animated.parallel([
      Animated.timing(opacity, {
        toValue: 0,
        duration: 170,
        useNativeDriver: true,
      }),
      Animated.timing(translateY, {
        toValue: 42,
        duration: 170,
        useNativeDriver: true,
      }),
      Animated.timing(scale, {
        toValue: 0.965,
        duration: 170,
        useNativeDriver: true,
      }),
    ]).start();
  };

  useEffect(() => {
    if (layout.y === null || viewportHeight === 0) return;

    const top = layout.y;
    const bottom = layout.y + layout.height;
    const screenTop = scrollY;
    const screenBottom = scrollY + viewportHeight;

    const isInside = screenBottom - offset >= top && screenTop + offset <= bottom;

    if (isInside && !visible) {
      setVisible(true);
      onEnter?.();
      animateIn();
    }

    if (!isInside && visible) {
      setVisible(false);
      animateOut();
    }
  }, [scrollY, viewportHeight, layout, visible]);

  return (
    <Animated.View
      onLayout={(event) => {
        setLayout({
          y: event.nativeEvent.layout.y,
          height: event.nativeEvent.layout.height,
        });
      }}
      style={[
        style,
        {
          opacity,
          transform: [{ translateY }, { scale }],
        },
      ]}
    >
      {children}
    </Animated.View>
  );
}

function HeroCarousel({ slides, locationSubtitle }) {
  const scrollX = useRef(new Animated.Value(0)).current;

  return (
    <AnimatedCard delay={60} style={styles.heroSection}>
      <View style={styles.compactHero}>
        <View style={styles.heroCircleOne} />
        <View style={styles.heroCircleTwo} />
        <View style={styles.heroPremiumGlow} />

        <View style={styles.heroTop}>
          <View style={{ flex: 1 }}>
            <Text style={styles.heroKicker}>FERME CONNECTÉE</Text>
            <Text style={styles.heroTitle}>Tableau de bord</Text>
            <Text style={styles.heroSubtitle}>
              {locationSubtitle || 'Localisation…'}
            </Text>
          </View>

          <View style={styles.onlineBadge}>
            <View style={styles.onlineDot} />
            <Text style={styles.onlineText}>En ligne</Text>
          </View>
        </View>
      </View>

      <Animated.FlatList
        style={styles.heroCardsList}
        data={slides}
        keyExtractor={(item) => item.id}
        horizontal
        showsHorizontalScrollIndicator={false}
        snapToInterval={HERO_SNAP}
        decelerationRate="fast"
        bounces={false}
        contentContainerStyle={[
          styles.heroCardsContent,
          {
            paddingLeft: HERO_SIDE_PADDING,
            paddingRight: HERO_SIDE_PADDING,
          },
        ]}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { x: scrollX } } }],
          { useNativeDriver: true }
        )}
        scrollEventThrottle={16}
        renderItem={({ item, index }) => {
          const inputRange = [
            (index - 1) * HERO_SNAP,
            index * HERO_SNAP,
            (index + 1) * HERO_SNAP,
          ];

          const scale = scrollX.interpolate({
            inputRange,
            outputRange: [0.82, 1, 0.82],
            extrapolate: 'clamp',
          });

          const translateY = scrollX.interpolate({
            inputRange,
            outputRange: [26, 0, 26],
            extrapolate: 'clamp',
          });

          const rotate = scrollX.interpolate({
            inputRange,
            outputRange: ['-8deg', '0deg', '8deg'],
            extrapolate: 'clamp',
          });

          const opacity = scrollX.interpolate({
            inputRange,
            outputRange: [0.58, 1, 0.58],
            extrapolate: 'clamp',
          });

          return (
            <Animated.View
              style={[
                styles.heroTravelSlide,
                {
                  width: HERO_CARD_WIDTH,
                  marginRight: HERO_CARD_SPACING,
                  opacity,
                  transform: [{ scale }, { translateY }, { rotate }],
                },
              ]}
            >
              <TouchableOpacity activeOpacity={0.9} style={styles.heroTravelCard}>
                <View
                  style={[
                    styles.heroTravelImage,
                    { backgroundColor: `${item.color}16` },
                  ]}
                >
                  <View
                    style={[
                      styles.heroTravelCircle,
                      { backgroundColor: `${item.color}22` },
                    ]}
                  />

                  <View style={styles.glassIconBox}>
                    {item.type === 'mc' ? (
                      <MaterialCommunityIcons
                        name={item.icon}
                        size={72}
                        color={item.color}
                      />
                    ) : (
                      <Ionicons name={item.icon} size={72} color={item.color} />
                    )}
                  </View>
                </View>

                <View style={styles.heroTravelFooter}>
                  <View>
                    <Text style={styles.heroTravelValue}>{item.value}</Text>
                    <Text style={styles.heroTravelLabel}>{item.label}</Text>
                  </View>

                  <View
                    style={[
                      styles.heroTravelBadge,
                      { backgroundColor: `${item.color}18` },
                    ]}
                  >
                    <Text
                      style={[
                        styles.heroTravelBadgeText,
                        { color: item.color },
                      ]}
                    >
                      {item.badge}
                    </Text>
                  </View>
                </View>
              </TouchableOpacity>
            </Animated.View>
          );
        }}
      />
    </AnimatedCard>
  );
}

function SmartWidget({ icon, title, value, subtitle, color, delay }) {
  return (
    <AnimatedCard delay={delay} style={styles.widgetCard}>
      <View style={styles.widgetHighlight} />
      <View style={[styles.widgetIcon, { backgroundColor: `${color}18` }]}>
        {icon}
      </View>

      <Text style={styles.widgetValue}>{value}</Text>
      <Text style={styles.widgetTitle}>{title}</Text>
      <Text style={styles.widgetSubtitle}>{subtitle}</Text>
    </AnimatedCard>
  );
}

function MilkChart({ playKey = 0, values, labels }) {
  const series = values?.length === 7 ? values : FALLBACK_MILK_SERIES;
  const dayLbl = labels?.length === 7 ? labels : FALLBACK_MILK_LABELS;
  const max = Math.max(...series, 1);
  const [selectedIndex, setSelectedIndex] = useState(series.length - 1);

  const animatedValues = useRef(series.map(() => new Animated.Value(0))).current;
  const pressValues = useRef(series.map(() => new Animated.Value(1))).current;
  const bubbleScale = useRef(new Animated.Value(1)).current;

  const seriesAnimKey = series.join(',');

  useEffect(() => {
    animatedValues.forEach((anim) => anim.setValue(0));

    Animated.stagger(
      80,
      animatedValues.map((anim) =>
        Animated.spring(anim, {
          toValue: 1,
          friction: 8,
          tension: 72,
          useNativeDriver: false,
        })
      )
    ).start();
  }, [playKey, seriesAnimKey]);

  useEffect(() => {
    bubbleScale.setValue(0.86);
    Animated.spring(bubbleScale, {
      toValue: 1,
      friction: 6,
      tension: 130,
      useNativeDriver: true,
    }).start();
  }, [selectedIndex]);

  const handlePress = (index) => {
    setSelectedIndex(index);

    Animated.sequence([
      Animated.spring(pressValues[index], {
        toValue: 1.12,
        friction: 5,
        tension: 180,
        useNativeDriver: true,
      }),
      Animated.spring(pressValues[index], {
        toValue: 1,
        friction: 5,
        tension: 160,
        useNativeDriver: true,
      }),
    ]).start();
  };

  return (
    <View style={styles.chartWrap}>
      <View style={styles.chartGridLineOne} />
      <View style={styles.chartGridLineTwo} />
      <View style={styles.chartGridLineThree} />

      {series.map((value, index) => {
        const isSelected = index === selectedIndex;
        const barHeight = 48 + (value / max) * 72;

        const animatedHeight = animatedValues[index].interpolate({
          inputRange: [0, 1],
          outputRange: [8, barHeight],
        });

        const animatedOpacity = animatedValues[index].interpolate({
          inputRange: [0, 1],
          outputRange: [0.35, 1],
        });

        return (
          <Pressable
            key={`${value}-${index}`}
            onPress={() => handlePress(index)}
            style={styles.chartItem}
          >
            {isSelected && (
              <Animated.View
                style={[
                  styles.chartValueBubble,
                  {
                    transform: [{ scale: bubbleScale }],
                  },
                ]}
              >
                <Text style={styles.chartValueText}>
                  {Math.round(value * 10) / 10}
                </Text>
                <Text style={styles.chartValueUnit}>kg</Text>
              </Animated.View>
            )}

            <Animated.View
              style={[
                styles.chartTrack,
                isSelected && styles.chartTrackSelected,
                {
                  transform: [{ scale: pressValues[index] }],
                },
              ]}
            >
              <Animated.View
                style={[
                  styles.chartBar,
                  {
                    height: animatedHeight,
                    opacity: animatedOpacity,
                    backgroundColor: isSelected ? COLORS.emerald : COLORS.green3,
                  },
                ]}
              />

              {isSelected && <View style={styles.chartGlow} />}
            </Animated.View>

            <Text style={[styles.chartDay, isSelected && styles.chartDayActive]}>
              {dayLbl[index]}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

function AlertCarousel({ items }) {
  const scrollX = useRef(new Animated.Value(0)).current;

  return (
    <AnimatedCard delay={0} style={styles.stackedCarouselSection}>
      <View style={styles.cardHeader}>
        <View>
          <Text style={styles.sectionKicker}>SUIVI TEMPS RÉEL</Text>
          <Text style={styles.cardTitle}>Alertes récentes</Text>
        </View>

        <TouchableOpacity style={styles.smallArrowBtn}>
          <Ionicons name="chevron-forward" size={18} color={COLORS.primary} />
        </TouchableOpacity>
      </View>

      <Animated.FlatList
        data={items}
        keyExtractor={(item) => item.id}
        horizontal
        showsHorizontalScrollIndicator={false}
        snapToInterval={ALERT_SNAP}
        decelerationRate="fast"
        bounces={false}
        contentContainerStyle={styles.stackedCarouselContent}
        onScroll={Animated.event(
          [{ nativeEvent: { contentOffset: { x: scrollX } } }],
          { useNativeDriver: true }
        )}
        scrollEventThrottle={16}
        renderItem={({ item, index }) => {
          const inputRange = [
            (index - 1) * ALERT_SNAP,
            index * ALERT_SNAP,
            (index + 1) * ALERT_SNAP,
          ];

          const scale = scrollX.interpolate({
            inputRange,
            outputRange: [0.82, 1, 0.82],
            extrapolate: 'clamp',
          });

          const translateY = scrollX.interpolate({
            inputRange,
            outputRange: [34, 0, 34],
            extrapolate: 'clamp',
          });

          const opacity = scrollX.interpolate({
            inputRange,
            outputRange: [0.55, 1, 0.55],
            extrapolate: 'clamp',
          });

          const rotate = scrollX.interpolate({
            inputRange,
            outputRange: ['-4deg', '0deg', '4deg'],
            extrapolate: 'clamp',
          });

          return (
            <Animated.View
              style={[
                styles.stackedSlide,
                {
                  opacity,
                  transform: [{ scale }, { translateY }, { rotate }],
                },
              ]}
            >
              <TouchableOpacity activeOpacity={0.92} style={styles.stackedCard}>
                <View
                  style={[
                    styles.stackedHero,
                    { backgroundColor: `${item.color}14` },
                  ]}
                >
                  <View
                    style={[
                      styles.stackedCircle,
                      { backgroundColor: `${item.color}22` },
                    ]}
                  />

                  <View style={styles.stackedMainIcon}>
                    {item.type === 'mc' ? (
                      <MaterialCommunityIcons
                        name={item.icon}
                        size={64}
                        color={item.color}
                      />
                    ) : (
                      <Ionicons name={item.icon} size={64} color={item.color} />
                    )}
                  </View>
                </View>

                <View style={styles.stackedTextBlock}>
                  <View style={styles.stackedTopLine}>
                    <Text style={styles.stackedCow}>{item.cow}</Text>

                    <View
                      style={[
                        styles.stackedBadge,
                        { backgroundColor: `${item.color}18` },
                      ]}
                    >
                      <Text
                        style={[
                          styles.stackedBadgeText,
                          { color: item.color },
                        ]}
                      >
                        {item.status}
                      </Text>
                    </View>
                  </View>

                  <Text style={styles.stackedTitle}>{item.title}</Text>
                  <Text style={styles.stackedDescription}>{item.description}</Text>

                  <View style={styles.stackedFooter}>
                    <Text style={styles.stackedTime}>{item.time}</Text>

                    <View style={styles.stackedArrow}>
                      <Ionicons name="arrow-forward" size={18} color={COLORS.white} />
                    </View>
                  </View>
                </View>
              </TouchableOpacity>
            </Animated.View>
          );
        }}
      />
    </AnimatedCard>
  );
}

function QuickAction({ icon, label, dark }) {
  const scale = useRef(new Animated.Value(1)).current;

  const pressIn = () => {
    Animated.spring(scale, {
      toValue: 0.95,
      friction: 6,
      tension: 180,
      useNativeDriver: true,
    }).start();
  };

  const pressOut = () => {
    Animated.spring(scale, {
      toValue: 1,
      friction: 6,
      tension: 180,
      useNativeDriver: true,
    }).start();
  };

  return (
    <Animated.View style={{ flex: 1, transform: [{ scale }] }}>
      <Pressable
        onPressIn={pressIn}
        onPressOut={pressOut}
        style={[styles.quickAction, dark && styles.quickActionDark]}
      >
        {icon}
        <Text style={[styles.quickActionText, dark && styles.quickActionTextDark]}>
          {label}
        </Text>
      </Pressable>
    </Animated.View>
  );
}

export default function HomeScreen() {
  const farmLoc = useFarmLocation('fr');

  const [scrollY, setScrollY] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);
  const [milkChartPlayKey, setMilkChartPlayKey] = useState(0);

  const [dashLoading, setDashLoading] = useState(true);
  const [milkSeries, setMilkSeries] = useState(FALLBACK_MILK_SERIES);
  const [milkLabels, setMilkLabels] = useState(FALLBACK_MILK_LABELS);
  const [milkAvg, setMilkAvg] = useState(
    FALLBACK_MILK_SERIES.reduce((a, b) => a + b, 0) / FALLBACK_MILK_SERIES.length
  );
  const [thiStr, setThiStr] = useState('--');
  const [thiSub, setThiSub] = useState('Chargement…');
  const [stressSnapshot, setStressSnapshot] = useState(null);
  const [heroSlides, setHeroSlides] = useState(() =>
    buildHeroSlides({
      herdCount: STATIC_HERD_COUNT,
      milkKgRounded: Math.round(FALLBACK_MILK_SERIES[FALLBACK_MILK_SERIES.length - 1]),
      healthPct: '91%',
      alertCount: alertCarouselFallback.length,
    })
  );
  const [alertItems, setAlertItems] = useState(alertCarouselFallback);

  const [weatherSnap, setWeatherSnap] = useState(null);
  const [weatherLoading, setWeatherLoading] = useState(true);
  const [weatherError, setWeatherError] = useState(null);

  const loadWeather = useCallback(async ({ silent } = {}, coords) => {
    const c = coords;
    if (!c || typeof c.latitude !== 'number' || typeof c.longitude !== 'number') return;

    if (!silent) {
      setWeatherLoading(true);
      setWeatherError(null);
    }
    try {
      const w = await fetchOpenMeteoCurrent(c);
      setWeatherSnap(w);
      setWeatherError(null);
    } catch {
      if (!silent) {
        setWeatherSnap(null);
        setWeatherError('Météo temporairement indisponible.');
      }
    } finally {
      if (!silent) setWeatherLoading(false);
    }
  }, []);

  const loadDashboard = useCallback(async () => {
    setDashLoading(true);
    try {
      const d = await fetchHomeDashboardModelPayload('12');
      setMilkSeries(d.milkKgByDay);
      setMilkLabels(d.milkDayLabels);
      setMilkAvg(d.milkAvgKg);
      setThiStr(d.thiStr);
      setThiSub(d.stressSubtitle);
      setStressSnapshot(d.stress);

      const stressAlerts = buildStressAlertItems(d.stress, {
        cowName: 'Rosette',
        cowNum: '12',
      }).map((a) => ({
        ...a,
        color: a.color === '#C84C4C' ? COLORS.danger : COLORS.warning,
      }));

      setAlertItems([...stressAlerts, ...alertCarouselFallback]);

      const milkRounded = Math.round(d.milkTodayKg ?? d.milkKgByDay[6] ?? 0);
      setHeroSlides(
        buildHeroSlides({
          herdCount: STATIC_HERD_COUNT,
          milkKgRounded: milkRounded,
          healthPct: stressHealthPctDisplay(d.stress),
          alertCount: stressAlerts.length + alertCarouselFallback.length,
        })
      );
    } catch {
      setThiSub('API modèle hors ligne (port 8008)');
      setAlertItems(alertCarouselFallback);
      setHeroSlides(
        buildHeroSlides({
          herdCount: STATIC_HERD_COUNT,
          milkKgRounded: Math.round(FALLBACK_MILK_SERIES[6]),
          healthPct: '91%',
          alertCount: alertCarouselFallback.length,
        })
      );
    } finally {
      setDashLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
    const t = setInterval(loadDashboard, 45000);
    return () => clearInterval(t);
  }, [loadDashboard]);

  useEffect(() => {
    if (farmLoc.loading) return undefined;
    loadWeather({ silent: false }, farmLoc.coords);
    const t = setInterval(
      () => loadWeather({ silent: true }, farmLoc.coords),
      15 * 60 * 1000
    );
    return () => clearInterval(t);
  }, [farmLoc.loading, farmLoc.coords.latitude, farmLoc.coords.longitude, loadWeather]);

  const milkAvgRounded = Math.round(milkAvg * 10) / 10;

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView
        style={styles.screen}
        contentContainerStyle={styles.content}
        showsVerticalScrollIndicator={false}
        scrollEventThrottle={16}
        onLayout={(event) => {
          setViewportHeight(event.nativeEvent.layout.height);
        }}
        onScroll={(event) => {
          setScrollY(event.nativeEvent.contentOffset.y);
        }}
      >
        <View style={styles.topHeader}>
          <View>
            <Text style={styles.title}>BoviTech Farm</Text>
          </View>

          <View style={styles.headerActions}>
            <TouchableOpacity activeOpacity={0.85} style={styles.headerIcon}>
              <Ionicons name="notifications-outline" size={21} color={COLORS.primary} />
              {isStressAlertActive(stressSnapshot) && (
                <View style={styles.notificationDot} />
              )}
            </TouchableOpacity>

            <TouchableOpacity activeOpacity={0.85} style={styles.avatar}>
              <MaterialCommunityIcons name="cow" size={22} color={COLORS.primary} />
            </TouchableOpacity>
          </View>
        </View>

        <HeroCarousel
          slides={heroSlides}
          locationSubtitle={
            farmLoc.loading ? 'Localisation…' : farmLoc.label || '—'
          }
        />

        <RevealOnScroll scrollY={scrollY} viewportHeight={viewportHeight}>
          <AnimatedCard delay={0} style={styles.weatherCard}>
            <View style={styles.weatherPremiumOrb} />

            <View style={styles.weatherTop}>
              <View>
                <Text style={styles.sectionKicker}>MÉTÉO À LA FERME</Text>
                <Text style={styles.weatherLocation}>
                  {farmLoc.loading ? 'Localisation…' : farmLoc.label || '—'}
                </Text>
                {weatherSnap?.updated ? (
                  <Text style={styles.weatherUpdated}>
                    Mis à jour {weatherSnap.updated} · Open-Meteo
                  </Text>
                ) : null}
              </View>

              <View style={styles.weatherIcon}>
                {weatherLoading && !weatherSnap ? (
                  <ActivityIndicator size="small" color={COLORS.primary} />
                ) : (
                  <Ionicons
                    name={weatherCodeToIonName(weatherSnap?.weatherCode)}
                    size={28}
                    color={COLORS.primary}
                  />
                )}
              </View>
            </View>

            <View style={styles.weatherMain}>
              <Text style={styles.weatherTemp}>
                {weatherSnap ? `${weatherSnap.temp}°` : weatherError ? '—' : '…'}
              </Text>
              <View style={{ flex: 1 }}>
                <Text style={styles.weatherState}>
                  {weatherError
                    ? weatherError
                    : weatherSnap
                      ? weatherSnap.summaryFr
                      : 'Chargement…'}
                </Text>
                <Text style={styles.weatherDetails}>
                  {weatherSnap
                    ? `Ressenti ${weatherSnap.feelsLike}° · Vent ${weatherSnap.windDirFr} ${weatherSnap.windKmh} km/h`
                    : weatherLoading
                      ? 'Données temps réel pour ce quartier.'
                      : ''}
                </Text>
              </View>
            </View>

            <View style={styles.weatherAdvice}>
              <Ionicons name="leaf-outline" size={18} color={COLORS.success} />
              <Text style={styles.weatherAdviceText}>
                {weatherSnap?.pastureFr ||
                  'Les prévisions Open-Meteo suivent votre position ou la zone par défaut.'}
              </Text>
            </View>
          </AnimatedCard>
        </RevealOnScroll>

        <RevealOnScroll scrollY={scrollY} viewportHeight={viewportHeight}>
          <View style={styles.widgetsGrid}>
            <SmartWidget
              delay={0}
              color={COLORS.primary}
              icon={<MaterialCommunityIcons name="cow" size={21} color={COLORS.primary} />}
              value={String(STATIC_HERD_COUNT)}
              title="Troupeau"
              subtitle="Colliers actifs"
            />

            <SmartWidget
              delay={80}
              color={COLORS.warning}
              icon={<Ionicons name="warning-outline" size={21} color={COLORS.warning} />}
              value={String(alertItems.length)}
              title="Alertes"
              subtitle="À surveiller"
            />

            <SmartWidget
              delay={160}
              color={COLORS.info}
              icon={<Ionicons name="analytics-outline" size={21} color={COLORS.info} />}
              value={thiStr}
              title="THI"
              subtitle={thiSub}
            />

            <SmartWidget
              delay={240}
              color={COLORS.success}
              icon={<Ionicons name="heart-outline" size={21} color={COLORS.success} />}
              value={stressHealthPctDisplay(stressSnapshot)}
              title="Santé"
              subtitle="Indice global"
            />
          </View>
        </RevealOnScroll>

        <RevealOnScroll
          scrollY={scrollY}
          viewportHeight={viewportHeight}
          onEnter={() => setMilkChartPlayKey((prev) => prev + 1)}
        >
          <AnimatedCard delay={0} style={styles.chartCard}>
            <View style={styles.chartCardGlow} />

            <View style={styles.cardHeader}>
              <View style={{ flex: 1 }}>
                <Text style={styles.sectionKicker}>PRODUCTION LAITIÈRE</Text>
                <Text style={styles.cardTitle}>7 derniers jours</Text>
              </View>

              <View style={styles.cardHeaderBadge}>
                {dashLoading ? (
                  <ActivityIndicator size="small" color={COLORS.emerald} />
                ) : (
                  <Text style={styles.cardHeaderBadgeText}>Moy. {milkAvgRounded} kg</Text>
                )}
              </View>
            </View>

            <MilkChart
              key={milkSeries.join('-')}
              playKey={milkChartPlayKey}
              values={milkSeries}
              labels={milkLabels}
            />
          </AnimatedCard>
        </RevealOnScroll>

        <RevealOnScroll scrollY={scrollY} viewportHeight={viewportHeight}>
          <AlertCarousel items={alertItems} />
        </RevealOnScroll>

        <RevealOnScroll scrollY={scrollY} viewportHeight={viewportHeight}>
          <AnimatedCard delay={0} style={styles.actionsCard}>
            <Text style={styles.cardTitle}>Actions rapides</Text>

            <View style={styles.quickActionsRow}>
              <QuickAction
                dark
                icon={<Ionicons name="add" size={19} color={COLORS.white} />}
                label="Ajouter"
              />

              <QuickAction
                icon={<Ionicons name="map-outline" size={19} color={COLORS.primary} />}
                label="GPS"
              />

              <QuickAction
                icon={
                  <Ionicons
                    name="chatbubble-ellipses-outline"
                    size={19}
                    color={COLORS.primary}
                  />
                }
                label="Assistant"
              />
            </View>
          </AnimatedCard>
        </RevealOnScroll>
      </ScrollView>
    </SafeAreaView>
  );
}

const shadow = {
  shadowColor: COLORS.primary,
  shadowOffset: { width: 0, height: 14 },
  shadowOpacity: 0.09,
  shadowRadius: 24,
  elevation: 5,
};

const premiumShadow = {
  shadowColor: COLORS.primaryDeep,
  shadowOffset: { width: 0, height: 22 },
  shadowOpacity: 0.16,
  shadowRadius: 34,
  elevation: 10,
};

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.cream,
  },
  screen: {
    flex: 1,
    backgroundColor: COLORS.cream,
  },
  content: {
    paddingHorizontal: 18,
    paddingTop: 14,
    paddingBottom: 124,
  },

  topHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    minHeight: 62,
    marginBottom: 12,
  },
  greeting: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 14,
    color: COLORS.sage,
    marginBottom: 3,
  },
  title: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 30,
    color: COLORS.black,
    letterSpacing: -1,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  headerIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: COLORS.white,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
    position: 'relative',
    ...shadow,
  },
  notificationDot: {
    position: 'absolute',
    top: 11,
    right: 12,
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: COLORS.warning,
  },
  avatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: COLORS.white,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: COLORS.border,
    ...shadow,
  },

  heroSection: {
    marginBottom: 8,
    position: 'relative',
    overflow: 'visible',
  },
  compactHero: {
    height: 165,
    borderRadius: 36,
    backgroundColor: COLORS.primary,
    padding: 20,
    overflow: 'hidden',
    zIndex: 1,
    elevation: 1,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 18 },
    shadowOpacity: 0.18,
    shadowRadius: 28,
  },
  heroCircleOne: {
    position: 'absolute',
    width: 190,
    height: 190,
    borderRadius: 95,
    backgroundColor: 'rgba(255,255,255,0.075)',
    right: -58,
    top: -68,
  },
  heroCircleTwo: {
    position: 'absolute',
    width: 135,
    height: 135,
    borderRadius: 68,
    backgroundColor: 'rgba(161,159,120,0.22)',
    left: -42,
    bottom: -58,
  },
  heroPremiumGlow: {
    position: 'absolute',
    width: 220,
    height: 60,
    borderRadius: 60,
    backgroundColor: 'rgba(255,255,255,0.08)',
    right: -30,
    bottom: 22,
    transform: [{ rotate: '-18deg' }],
  },
  heroTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  heroKicker: {
    fontFamily: 'Inter_700Bold',
    fontSize: 11,
    color: 'rgba(246,246,236,0.68)',
    letterSpacing: 1.3,
    marginBottom: 6,
  },
  heroTitle: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 28,
    lineHeight: 32,
    color: COLORS.white,
    letterSpacing: -0.9,
  },
  heroSubtitle: {
    fontFamily: 'Inter_500Medium',
    fontSize: 13,
    color: 'rgba(246,246,236,0.76)',
    marginTop: 5,
  },
  onlineBadge: {
    height: 34,
    paddingHorizontal: 12,
    borderRadius: 17,
    backgroundColor: 'rgba(255,255,255,0.14)',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.14)',
  },
  onlineDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: COLORS.emerald,
  },
  onlineText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 12,
    color: COLORS.white,
  },

  heroCardsList: {
    marginTop: -54,
    zIndex: 10,
    elevation: 10,
    overflow: 'visible',
  },
  heroCardsContent: {
    paddingTop: 0,
    paddingBottom: 22,
  },
  heroTravelSlide: {
    height: 255,
    zIndex: 20,
    elevation: 20,
  },
  heroTravelCard: {
    height: 238,
    borderRadius: 34,
    backgroundColor: COLORS.white,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(229,227,214,0.86)',
    zIndex: 30,
    elevation: 30,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 20 },
    shadowOpacity: 0.22,
    shadowRadius: 30,
  },
  heroTravelImage: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  heroTravelCircle: {
    position: 'absolute',
    width: 170,
    height: 170,
    borderRadius: 85,
    right: -48,
    top: -55,
  },
  glassIconBox: {
    width: 112,
    height: 112,
    borderRadius: 38,
    backgroundColor: 'rgba(255,255,255,0.82)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.72)',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 14 },
    shadowOpacity: 0.12,
    shadowRadius: 24,
    elevation: 8,
  },
  heroTravelFooter: {
    minHeight: 76,
    paddingHorizontal: 16,
    paddingVertical: 13,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 10,
    backgroundColor: COLORS.white,
  },
  heroTravelValue: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 26,
    color: COLORS.black,
    letterSpacing: -0.8,
  },
  heroTravelLabel: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.sage,
    marginTop: 2,
  },
  heroTravelBadge: {
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 999,
    maxWidth: 110,
  },
  heroTravelBadgeText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 10.5,
  },

  weatherCard: {
    backgroundColor: COLORS.white,
    borderRadius: 32,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(229,227,214,0.9)',
    overflow: 'hidden',
    ...premiumShadow,
  },
  weatherPremiumOrb: {
    position: 'absolute',
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: 'rgba(46,65,45,0.055)',
    right: -55,
    top: -70,
  },
  weatherTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 18,
  },
  sectionKicker: {
    fontFamily: 'Inter_700Bold',
    fontSize: 10.5,
    letterSpacing: 2,
    color: COLORS.earth,
    marginBottom: 5,
  },
  weatherLocation: {
    fontFamily: 'Inter_500Medium',
    fontSize: 13,
    color: COLORS.sage,
  },
  weatherUpdated: {
    fontFamily: 'Inter_500Medium',
    fontSize: 10,
    color: COLORS.olive,
    marginTop: 4,
    opacity: 0.85,
  },
  weatherIcon: {
    width: 50,
    height: 50,
    borderRadius: 20,
    backgroundColor: COLORS.primarySoft,
    alignItems: 'center',
    justifyContent: 'center',
  },
  weatherMain: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 18,
    marginBottom: 16,
  },
  weatherTemp: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 54,
    color: COLORS.primary,
    letterSpacing: -2,
  },
  weatherState: {
    fontFamily: 'PlusJakartaSans_700Bold',
    fontSize: 19,
    color: COLORS.black,
  },
  weatherDetails: {
    fontFamily: 'Inter_500Medium',
    fontSize: 13,
    color: COLORS.sage,
    marginTop: 4,
  },
  weatherAdvice: {
    flexDirection: 'row',
    gap: 8,
    backgroundColor: COLORS.cream,
    borderRadius: 20,
    padding: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(229,227,214,0.72)',
  },
  weatherAdviceText: {
    flex: 1,
    fontFamily: 'Inter_500Medium',
    fontSize: 12.5,
    color: COLORS.darkOlive,
    lineHeight: 18,
  },

  widgetsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginBottom: 16,
  },
  widgetCard: {
    width: '48.2%',
    minHeight: 148,
    backgroundColor: COLORS.white,
    borderRadius: 30,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(229,227,214,0.88)',
    overflow: 'hidden',
    ...shadow,
  },
  widgetHighlight: {
    position: 'absolute',
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: 'rgba(46,65,45,0.045)',
    right: -38,
    top: -38,
  },
  widgetIcon: {
    width: 42,
    height: 42,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 18,
  },
  widgetValue: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 26,
    color: COLORS.black,
    letterSpacing: -0.8,
  },
  widgetTitle: {
    fontFamily: 'Inter_700Bold',
    fontSize: 13,
    color: COLORS.darkOlive,
    marginTop: 3,
  },
  widgetSubtitle: {
    fontFamily: 'Inter_500Medium',
    fontSize: 12,
    color: COLORS.sage,
    marginTop: 3,
  },

  chartCard: {
    backgroundColor: COLORS.cream2,
    borderRadius: 34,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(46,65,45,0.10)',
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 22 },
    shadowOpacity: 0.15,
    shadowRadius: 34,
    elevation: 8,
  },
  chartCardGlow: {
    position: 'absolute',
    width: 210,
    height: 210,
    borderRadius: 105,
    backgroundColor: 'rgba(0,166,106,0.08)',
    right: -98,
    top: -112,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
    alignItems: 'flex-start',
    marginBottom: 16,
  },
  cardTitle: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 22,
    color: COLORS.black,
    letterSpacing: -0.75,
  },
  cardHeaderBadge: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: 'rgba(0,166,106,0.12)',
    borderRadius: 999,
    borderWidth: 1,
    borderColor: 'rgba(0,166,106,0.15)',
  },
  cardHeaderBadgeText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 11,
    color: COLORS.emerald,
  },

  chartWrap: {
    height: 190,
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    paddingTop: 32,
    paddingHorizontal: 2,
    position: 'relative',
  },
  chartGridLineOne: {
    position: 'absolute',
    left: 8,
    right: 8,
    top: 58,
    height: 1,
    backgroundColor: 'rgba(46,65,45,0.06)',
  },
  chartGridLineTwo: {
    position: 'absolute',
    left: 8,
    right: 8,
    top: 96,
    height: 1,
    backgroundColor: 'rgba(46,65,45,0.07)',
  },
  chartGridLineThree: {
    position: 'absolute',
    left: 8,
    right: 8,
    top: 134,
    height: 1,
    backgroundColor: 'rgba(46,65,45,0.05)',
  },
  chartItem: {
    alignItems: 'center',
    flex: 1,
    position: 'relative',
  },
  chartTrack: {
    height: 132,
    width: 30,
    borderRadius: 999,
    backgroundColor: 'rgba(255,255,255,0.58)',
    justifyContent: 'flex-end',
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(229,227,214,0.86)',
  },
  chartTrackSelected: {
    backgroundColor: 'rgba(255,255,255,0.78)',
    borderColor: 'rgba(0,166,106,0.32)',
    shadowColor: COLORS.emerald,
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.22,
    shadowRadius: 18,
    elevation: 7,
  },
  chartBar: {
    width: '100%',
    borderRadius: 999,
  },
  chartGlow: {
    position: 'absolute',
    bottom: 9,
    alignSelf: 'center',
    width: 15,
    height: 15,
    borderRadius: 8,
    backgroundColor: 'rgba(255,255,255,0.36)',
  },
  chartDay: {
    fontFamily: 'Inter_700Bold',
    fontSize: 11,
    color: COLORS.sage,
    marginTop: 10,
  },
  chartDayActive: {
    color: COLORS.emerald,
  },
  chartValueBubble: {
    position: 'absolute',
    top: -32,
    minWidth: 43,
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: COLORS.emerald,
    zIndex: 5,
    alignItems: 'center',
    shadowColor: COLORS.emerald,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 14,
    elevation: 8,
  },
  chartValueText: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 11,
    color: COLORS.black,
    lineHeight: 12,
  },
  chartValueUnit: {
    fontFamily: 'Inter_800ExtraBold',
    fontSize: 8,
    color: COLORS.black,
    lineHeight: 9,
  },

  stackedCarouselSection: {
    marginBottom: 18,
  },
  stackedCarouselContent: {
    paddingLeft: ALERT_SIDE_PADDING,
    paddingRight: ALERT_SIDE_PADDING,
    paddingVertical: 22,
  },
  stackedSlide: {
    width: ALERT_CARD_WIDTH,
    marginRight: ALERT_CARD_SPACING,
  },
  stackedCard: {
    minHeight: 330,
    borderRadius: 38,
    backgroundColor: COLORS.white,
    borderWidth: 1,
    borderColor: 'rgba(229,227,214,0.92)',
    overflow: 'hidden',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 22 },
    shadowOpacity: 0.18,
    shadowRadius: 34,
    elevation: 10,
  },
  stackedHero: {
    height: 142,
    borderBottomLeftRadius: 34,
    borderBottomRightRadius: 34,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  stackedCircle: {
    position: 'absolute',
    width: 180,
    height: 180,
    borderRadius: 90,
    right: -46,
    top: -62,
  },
  stackedMainIcon: {
    width: 96,
    height: 96,
    borderRadius: 36,
    backgroundColor: COLORS.white,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.14,
    shadowRadius: 20,
    elevation: 7,
  },
  stackedTextBlock: {
    padding: 18,
  },
  stackedTopLine: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 10,
    marginBottom: 12,
  },
  stackedCow: {
    fontFamily: 'Inter_700Bold',
    fontSize: 13,
    color: COLORS.sage,
    flex: 1,
  },
  stackedBadge: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
  },
  stackedBadgeText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 10.5,
  },
  stackedTitle: {
    fontFamily: 'PlusJakartaSans_800ExtraBold',
    fontSize: 24,
    lineHeight: 29,
    color: COLORS.black,
    letterSpacing: -0.8,
  },
  stackedDescription: {
    fontFamily: 'Inter_500Medium',
    fontSize: 13,
    color: COLORS.darkOlive,
    lineHeight: 19,
    marginTop: 9,
  },
  stackedFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: 22,
  },
  stackedTime: {
    fontFamily: 'Inter_600SemiBold',
    fontSize: 12,
    color: COLORS.sage,
  },
  stackedArrow: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: COLORS.primary,
    alignItems: 'center',
    justifyContent: 'center',
  },

  smallArrowBtn: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: COLORS.cream,
    alignItems: 'center',
    justifyContent: 'center',
  },

  actionsCard: {
    backgroundColor: COLORS.white,
    borderRadius: 32,
    padding: 18,
    borderWidth: 1,
    borderColor: 'rgba(229,227,214,0.92)',
    ...shadow,
  },
  quickActionsRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 14,
  },
  quickAction: {
    height: 72,
    borderRadius: 24,
    backgroundColor: COLORS.cream,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    borderWidth: 1,
    borderColor: 'rgba(229,227,214,0.8)',
  },
  quickActionDark: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  quickActionText: {
    fontFamily: 'Inter_700Bold',
    fontSize: 12,
    color: COLORS.primary,
  },
  quickActionTextDark: {
    color: COLORS.white,
  },
});