import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Image,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StatusBar,
  Animated,
} from 'react-native';

import { COLORS, RADIUS, SPACING, SHADOWS } from '../constants/theme';

export default function RegisterScreen({ navigation }) {
  const [step, setStep] = useState(1);

  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    farmName: '',
    cowCount: '',
    region: '',
  });

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

  const update = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleNext = () => {
    setStep(2);
  };

  const handleRegister = () => {
    navigation.replace('Home');
  };

  return (
    <View style={styles.root}>
      <StatusBar translucent backgroundColor="transparent" barStyle="light-content" />

      <Image
        source={require('../../assets/bovitechbg.png')}
        style={styles.bgImage}
        resizeMode="cover"
      />

      <View style={styles.darkOverlay} />

      <KeyboardAvoidingView
        style={styles.wrapper}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          <TouchableOpacity
            style={styles.backBtn}
            onPress={() => (step === 2 ? setStep(1) : navigation.goBack())}
          >
            <Text style={styles.backText}>←</Text>
          </TouchableOpacity>

          <View style={styles.logoWrap}>
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
              {step === 1 ? 'Créer un compte' : 'Votre exploitation'}
            </Text>

            <Text style={styles.subtitle}>
              {step === 1
                ? 'Créez vos identifiants de connexion.'
                : 'Ajoutez les informations de votre ferme.'}
            </Text>

            <View style={styles.stepRow}>
              <View style={[styles.stepDot, step >= 1 && styles.stepDotActive]}>
                <Text style={[styles.stepNumber, step >= 1 && styles.stepNumberActive]}>
                  1
                </Text>
              </View>

              <View style={[styles.stepLine, step === 2 && styles.stepLineActive]} />

              <View style={[styles.stepDot, step === 2 && styles.stepDotActive]}>
                <Text style={[styles.stepNumber, step === 2 && styles.stepNumberActive]}>
                  2
                </Text>
              </View>
            </View>

            {step === 1 && (
              <>
                <TextInput
                  style={styles.input}
                  placeholder="Nom d’utilisateur"
                  placeholderTextColor="rgba(255,255,255,0.65)"
                  value={form.username}
                  onChangeText={(v) => update('username', v)}
                />

                <TextInput
                  style={styles.input}
                  placeholder="Adresse e-mail"
                  placeholderTextColor="rgba(255,255,255,0.65)"
                  value={form.email}
                  onChangeText={(v) => update('email', v)}
                  keyboardType="email-address"
                  autoCapitalize="none"
                  autoCorrect={false}
                />

                <TextInput
                  style={styles.input}
                  placeholder="Mot de passe"
                  placeholderTextColor="rgba(255,255,255,0.65)"
                  value={form.password}
                  onChangeText={(v) => update('password', v)}
                  secureTextEntry
                />

                <TextInput
                  style={styles.input}
                  placeholder="Confirmer le mot de passe"
                  placeholderTextColor="rgba(255,255,255,0.65)"
                  value={form.confirmPassword}
                  onChangeText={(v) => update('confirmPassword', v)}
                  secureTextEntry
                />

                <TouchableOpacity style={styles.primaryButton} onPress={handleNext}>
                  <Text style={styles.primaryButtonText}>Continuer</Text>
                </TouchableOpacity>
              </>
            )}

            {step === 2 && (
              <>
                <TextInput
                  style={styles.input}
                  placeholder="Nom de la ferme"
                  placeholderTextColor="rgba(255,255,255,0.65)"
                  value={form.farmName}
                  onChangeText={(v) => update('farmName', v)}
                />

                <TextInput
                  style={styles.input}
                  placeholder="Nombre de vaches"
                  placeholderTextColor="rgba(255,255,255,0.65)"
                  value={form.cowCount}
                  onChangeText={(v) => update('cowCount', v)}
                  keyboardType="numeric"
                />

                <TextInput
                  style={styles.input}
                  placeholder="Région"
                  placeholderTextColor="rgba(255,255,255,0.65)"
                  value={form.region}
                  onChangeText={(v) => update('region', v)}
                />

                <TouchableOpacity style={styles.primaryButton} onPress={handleRegister}>
                  <Text style={styles.primaryButtonText}>Créer mon compte</Text>
                </TouchableOpacity>
              </>
            )}

            <View style={styles.bottomRow}>
              <Text style={styles.bottomText}>Déjà un compte ? </Text>
              <TouchableOpacity onPress={() => navigation.navigate('Login')}>
                <Text style={styles.bottomLink}>Se connecter</Text>
              </TouchableOpacity>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
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
  },

  darkOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(27, 67, 50, 0.36)',
  },

  wrapper: {
    flex: 1,
  },

  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: SPACING.lg,
    paddingTop: 58,
    paddingBottom: 34,
  },

  backBtn: {
    alignSelf: 'flex-start',
    marginBottom: 10,
  },

  backText: {
    color: COLORS.white,
    fontSize: 32,
    fontWeight: '700',
  },

  logoWrap: {
    alignItems: 'center',
    marginBottom: 22,
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
    fontSize: 30,
    fontWeight: '900',
    marginBottom: 8,
  },

  subtitle: {
    color: 'rgba(255,255,255,0.84)',
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 22,
  },

  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },

  stepDot: {
    width: 30,
    height: 30,
    borderRadius: 15,
    backgroundColor: 'rgba(255,255,255,0.18)',
    alignItems: 'center',
    justifyContent: 'center',
  },

  stepDotActive: {
    backgroundColor: '#F2C94C',
  },

  stepNumber: {
    color: 'rgba(255,255,255,0.7)',
    fontWeight: '900',
  },

  stepNumberActive: {
    color: '#1B4332',
  },

  stepLine: {
    width: 70,
    height: 2,
    backgroundColor: 'rgba(255,255,255,0.24)',
  },

  stepLineActive: {
    backgroundColor: '#F2C94C',
  },

  input: {
    height: 54,
    borderRadius: RADIUS.md,
    backgroundColor: 'rgba(255,255,255,0.15)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.22)',
    paddingHorizontal: 16,
    color: COLORS.white,
    fontSize: 15,
    marginBottom: 14,
  },

  primaryButton: {
    height: 56,
    borderRadius: RADIUS.full,
    backgroundColor: '#F2C94C',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 4,
    ...SHADOWS.strong,
  },

  primaryButtonText: {
    color: '#1B4332',
    fontSize: 16,
    fontWeight: '900',
  },

  bottomRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 22,
  },

  bottomText: {
    color: 'rgba(255,255,255,0.72)',
    fontSize: 14,
  },

  bottomLink: {
    color: '#F2C94C',
    fontSize: 14,
    fontWeight: '800',
  },
});