import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { View, Text, StyleSheet } from 'react-native';
import { colors } from '../theme/colors';
import { t } from '../i18n';

import HomeScreen from '../screens/HomeScreen';
import HerdScreen from '../screens/HerdScreen';
import GPSScreen from '../screens/GPSScreen';
import HealthScreen from '../screens/HealthScreen';
import PredictionsScreen from '../screens/PredictionsScreen';
import ChatbotScreen from '../screens/ChatbotScreen';

const Tab = createBottomTabNavigator();

// ── Custom Tab Icon (SVG via inline paths) ────────────────────────────────────
function TabIcon({ name, focused }) {
  const icons = {
    Home: (
      <View style={{ alignItems: 'center' }}>
        <Text style={{ fontSize: 20, lineHeight: 24 }}>🏠</Text>
      </View>
    ),
    Herd: (
      <View style={{ alignItems: 'center' }}>
        <Text style={{ fontSize: 20, lineHeight: 24 }}>🐄</Text>
      </View>
    ),
    GPS: (
      <View style={{ alignItems: 'center' }}>
        <Text style={{ fontSize: 20, lineHeight: 24 }}>🗺️</Text>
      </View>
    ),
    Health: (
      <View style={{ alignItems: 'center' }}>
        <Text style={{ fontSize: 20, lineHeight: 24 }}>❤️</Text>
      </View>
    ),
    Assistant: (
      <View style={{ alignItems: 'center' }}>
        <Text style={{ fontSize: 20, lineHeight: 24 }}>🤖</Text>
      </View>
    ),
    Predictions: (
      <View style={{ alignItems: 'center' }}>
        <Text style={{ fontSize: 20, lineHeight: 24 }}>📈</Text>
      </View>
    ),
  };
  return (
    <View
      style={[
        styles.iconWrapper,
        focused && {
          backgroundColor: colors.greenLight,
          borderRadius: 14,
        },
      ]}
    >
      {icons[name]}
    </View>
  );
}

export default function AppNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ focused }) => (
          <TabIcon name={route.name} focused={focused} />
        ),
        tabBarActiveTintColor: colors.green,
        tabBarInactiveTintColor: colors.grayMid,
        tabBarStyle: styles.tabBar,
        tabBarLabelStyle: styles.tabLabel,
        tabBarItemStyle: styles.tabItem,
      })}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{ tabBarLabel: t('tabs.home') }}
      />
      <Tab.Screen
        name="Herd"
        component={HerdScreen}
        options={{ tabBarLabel: t('tabs.herd') }}
      />
      <Tab.Screen
        name="GPS"
        component={GPSScreen}
        options={{ tabBarLabel: t('tabs.gps') }}
      />
      <Tab.Screen
        name="Health"
        component={HealthScreen}
        options={{ tabBarLabel: t('tabs.health') }}
      />
      <Tab.Screen
        name="Predictions"
        component={PredictionsScreen}
        options={{ tabBarLabel: t('tabs.predictions') }}
      />
      <Tab.Screen
        name="Assistant"
        component={ChatbotScreen}
        options={{ tabBarLabel: t('tabs.assistant') }}
      />
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: colors.tabBarBg,
    borderTopWidth: 0,
    paddingBottom: 10,
    paddingTop: 8,
    minHeight: 64,
    borderTopLeftRadius: 22,
    borderTopRightRadius: 22,
    shadowColor: '#1B4332',
    shadowOffset: { width: 0, height: -6 },
    shadowOpacity: 0.08,
    shadowRadius: 18,
    elevation: 12,
  },
  tabLabel: {
    fontSize: 10,
    fontWeight: '600',
    marginTop: 2,
    letterSpacing: 0.2,
  },
  tabItem: {
    paddingTop: 4,
  },
  iconWrapper: {
    padding: 6,
    paddingHorizontal: 10,
  },
});
