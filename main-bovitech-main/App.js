import React from 'react';
import {
  ActivityIndicator,
  Platform,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createStackNavigator } from '@react-navigation/stack';

import {
  useFonts,
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  Inter_700Bold,
} from '@expo-google-fonts/inter';

import {
  PlusJakartaSans_600SemiBold,
  PlusJakartaSans_700Bold,
  PlusJakartaSans_800ExtraBold,
} from '@expo-google-fonts/plus-jakarta-sans';

import OnboardingScreen from './src/screens/OnboardingScreen';
import LoginScreen from './src/screens/LoginScreen';
import RegisterScreen from './src/screens/RegisterScreen';
import AppNavigator from './src/navigation/AppNavigator';

import { colors } from './src/theme/colors';
import { initI18n } from './src/i18n';
import { AuthProvider, useAuth } from './src/context/AuthContext';

const Stack =
  Platform.OS === 'web' ? createStackNavigator() : createNativeStackNavigator();

function AppNavigation() {
  const { isAuthenticated, isLoading: authLoading } = useAuth();

  if (authLoading) {
    return (
      <View style={styles.boot}>
        <ActivityIndicator size="large" color={colors.green} />
        <Text style={styles.bootText}>Chargement BoviTech…</Text>
      </View>
    );
  }

  const stackScreenOptions =
    Platform.OS === 'web'
      ? { headerShown: false, cardStyle: { flex: 1 } }
      : { headerShown: false };

  return (
    <Stack.Navigator
      initialRouteName={isAuthenticated ? 'Home' : 'Onboarding'}
      screenOptions={stackScreenOptions}
    >
      <Stack.Screen name="Onboarding" component={OnboardingScreen} />
      <Stack.Screen name="Login" component={LoginScreen} />
      <Stack.Screen name="Register" component={RegisterScreen} />
      <Stack.Screen name="Home" component={AppNavigator} />
    </Stack.Navigator>
  );
}

export default function App() {
  const [ready, setReady] = React.useState(false);

  const [fontsLoaded] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Inter_700Bold,
    PlusJakartaSans_600SemiBold,
    PlusJakartaSans_700Bold,
    PlusJakartaSans_800ExtraBold,
  });

  React.useEffect(() => {
    let alive = true;

    (async () => {
      try {
        await initI18n();
      } finally {
        if (alive) setReady(true);
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  if (!ready || !fontsLoaded) {
    return (
      <View style={styles.boot}>
        <ActivityIndicator size="large" color={colors.green} />
        <Text style={styles.bootText}>Chargement BoviTech…</Text>
      </View>
    );
  }

  return (
    <AuthProvider>
      <View style={styles.root}>
        <NavigationContainer>
          <StatusBar backgroundColor={colors.green} barStyle="light-content" />
          <AppNavigation />
        </NavigationContainer>
      </View>
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
  boot: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f4f6f9',
  },
  bootText: {
    marginTop: 16,
    fontSize: 15,
    color: '#1A3C2E',
  },
});
