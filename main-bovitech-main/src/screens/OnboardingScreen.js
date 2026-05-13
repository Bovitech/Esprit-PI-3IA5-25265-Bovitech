import React, { useEffect, useRef } from 'react';
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  StyleSheet,
  StatusBar,
  Animated,
} from 'react-native';

import { COLORS, RADIUS, SPACING, SHADOWS } from '../constants/theme';

export default function OnboardingScreen({ navigation }) {
  const logoScale = useRef(new Animated.Value(0.85)).current;
  const logoOpacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(logoScale, {
        toValue: 1,
        friction: 5,
        tension: 80,
        useNativeDriver: true,
      }),
      Animated.timing(logoOpacity, {
        toValue: 1,
        duration: 700,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  return (
    <View style={styles.root}>
      <StatusBar translucent backgroundColor="transparent" barStyle="light-content" />

      <Image
        source={require('../../assets/bovitechbg.png')}
        style={styles.bgImage}
        resizeMode="cover"
      />

      <View style={styles.darkOverlay} />

      <View style={styles.content}>
        <View style={styles.top}>
          <Animated.Image
            source={require('../../assets/logobt.png')}
            style={[
              styles.logoImage,
              {
                opacity: logoOpacity,
                transform: [{ scale: logoScale }],
              },
            ]}
            resizeMode="contain"
          />
        </View>

        <View style={styles.card}>
          <Text style={styles.title}>
            Surveille.{'\n'}
            Analyse.{'\n'}
            <Text style={styles.titleAccent}>Optimise.</Text>
          </Text>

          <Text style={styles.subtitle}>
            Suivi intelligent de votre troupeau en temps réel.
          </Text>

          <TouchableOpacity
            style={styles.primaryButton}
            activeOpacity={0.85}
            onPress={() => navigation.navigate('Register')}
          >
            <Text style={styles.primaryButtonText}>Commencer</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryButton}
            activeOpacity={0.85}
            onPress={() => navigation.navigate('Login')}
          >
            <Text style={styles.secondaryButtonText}>Se connecter</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#07110D',
  },

  bgImage: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    top: 0,
    left: 0,
  },

  darkOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(27, 67, 50, 0.28)',
  },

  content: {
    flex: 1,
    paddingHorizontal: SPACING.lg,
    paddingTop: 76,
    paddingBottom: 34,
    justifyContent: 'space-between',
  },

  top: {
    alignItems: 'center',
  },

  logoImage: {
    width: 190,
    height: 70,
  },

  card: {
    backgroundColor: 'rgba(27, 67, 50, 0.78)',
    borderRadius: RADIUS.xl,
    padding: 24,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.18)',
  },

  title: {
    color: COLORS.white,
    fontSize: 36,
    fontWeight: '900',
    lineHeight: 44,
  },

  titleAccent: {
    color: '#F2C94C',
  },

  subtitle: {
    color: 'rgba(255,255,255,0.9)',
    fontSize: 15,
    lineHeight: 22,
    marginTop: 14,
    marginBottom: 24,
  },

  primaryButton: {
    height: 56,
    borderRadius: RADIUS.full,
    backgroundColor: '#F2C94C',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 12,
    ...SHADOWS.strong,
  },

  primaryButtonText: {
    color: '#1B4332',
    fontSize: 16,
    fontWeight: '800',
  },

  secondaryButton: {
    height: 54,
    borderRadius: RADIUS.full,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.45)',
    backgroundColor: 'rgba(255,255,255,0.08)',
    justifyContent: 'center',
    alignItems: 'center',
  },

  secondaryButtonText: {
    color: COLORS.white,
    fontSize: 15,
    fontWeight: '700',
  },
});