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
  StatusBar,
  Animated,
} from 'react-native';

import { COLORS, RADIUS, SPACING, SHADOWS } from '../constants/theme';

export default function LoginScreen({ navigation }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

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

  const handleLogin = () => {
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
        style={styles.content}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Text style={styles.backText}>←</Text>
        </TouchableOpacity>

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
          <Text style={styles.title}>Bon retour</Text>
          <Text style={styles.subtitle}>
            Connectez-vous pour accéder au suivi de votre troupeau.
          </Text>

          <TextInput
            style={styles.input}
            placeholder="Adresse e-mail"
            placeholderTextColor="rgba(255,255,255,0.65)"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
            autoCapitalize="none"
            autoCorrect={false}
          />

          <TextInput
            style={styles.input}
            placeholder="Mot de passe"
            placeholderTextColor="rgba(255,255,255,0.65)"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />

          <TouchableOpacity style={styles.forgotBtn}>
            <Text style={styles.forgotText}>Mot de passe oublié ?</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.primaryButton} onPress={handleLogin}>
            <Text style={styles.primaryButtonText}>Se connecter</Text>
          </TouchableOpacity>

          <View style={styles.bottomRow}>
            <Text style={styles.bottomText}>Pas encore de compte ? </Text>
            <TouchableOpacity onPress={() => navigation.navigate('Register')}>
              <Text style={styles.bottomLink}>S’inscrire</Text>
            </TouchableOpacity>
          </View>
        </View>
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
    backgroundColor: 'rgba(27, 67, 50, 0.34)',
  },

  content: {
    flex: 1,
    paddingHorizontal: SPACING.lg,
    paddingTop: 58,
    paddingBottom: 34,
    justifyContent: 'space-between',
  },

  backBtn: {
    position: 'absolute',
    top: 58,
    left: 24,
    zIndex: 5,
  },

  backText: {
    color: COLORS.white,
    fontSize: 32,
    fontWeight: '700',
  },

  top: {
    alignItems: 'center',
    marginTop: 20,
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
    fontSize: 32,
    fontWeight: '900',
    marginBottom: 8,
  },

  subtitle: {
    color: 'rgba(255,255,255,0.84)',
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 24,
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

  forgotBtn: {
    alignSelf: 'flex-end',
    marginBottom: 22,
  },

  forgotText: {
    color: '#F2C94C',
    fontSize: 13,
    fontWeight: '700',
  },

  primaryButton: {
    height: 56,
    borderRadius: RADIUS.full,
    backgroundColor: '#F2C94C',
    justifyContent: 'center',
    alignItems: 'center',
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