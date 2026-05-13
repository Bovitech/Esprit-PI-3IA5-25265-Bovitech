import React, { useEffect, useRef } from 'react';
import { StyleSheet, Animated, Platform } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

import HomeScreen from '../screens/HomeScreen';
import HerdScreen from '../screens/HerdScreen';
import GPSScreen from '../screens/GPSScreen';
import PredictionsScreen from '../screens/PredictionsScreen';
import ChatbotScreen from '../screens/ChatbotScreen';

const Tab = createBottomTabNavigator();

function AnimatedTabIcon({ routeName, focused }) {
  const scale = useRef(new Animated.Value(focused ? 1 : 0)).current;

  useEffect(() => {
    Animated.spring(scale, {
      toValue: focused ? 1 : 0,
      friction: 7,
      tension: 95,
      useNativeDriver: true,
    }).start();
  }, [focused, scale]);

  const color = focused ? '#F2C94C' : 'rgba(255,255,255,0.72)';

  const iconScale = scale.interpolate({
    inputRange: [0, 1],
    outputRange: [1, 1.12],
  });

  const translateY = scale.interpolate({
    inputRange: [0, 1],
    outputRange: [0, -2],
  });

  const size = focused ? 23 : 21;

  let icon = null;

  if (routeName === 'Home') {
    icon = <Ionicons name={focused ? 'home' : 'home-outline'} size={size} color={color} />;
  } else if (routeName === 'Herd') {
    icon = <MaterialCommunityIcons name="cow" size={size + 2} color={color} />;
  } else if (routeName === 'GPS') {
    icon = <Ionicons name={focused ? 'map' : 'map-outline'} size={size} color={color} />;
  } else if (routeName === 'Predictions') {
    icon = <Ionicons name={focused ? 'analytics' : 'analytics-outline'} size={size} color={color} />;
  } else {
    icon = (
      <Ionicons
        name={focused ? 'chatbubble-ellipses' : 'chatbubble-ellipses-outline'}
        size={size}
        color={color}
      />
    );
  }

  return (
    <Animated.View
      style={[
        styles.iconWrapper,
        focused && styles.iconWrapperActive,
        {
          transform: [{ scale: iconScale }, { translateY }],
        },
      ]}
    >
      {icon}
    </Animated.View>
  );
}

export default function AppNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarIcon: ({ focused }) => (
          <AnimatedTabIcon routeName={route.name} focused={focused} />
        ),
        tabBarActiveTintColor: '#F2C94C',
        tabBarInactiveTintColor: 'rgba(255,255,255,0.72)',
        tabBarStyle: styles.tabBar,
        tabBarLabelStyle: styles.tabLabel,
        tabBarItemStyle: styles.tabItem,
        tabBarHideOnKeyboard: true,
      })}
    >
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{ tabBarLabel: 'Accueil' }}
      />

      <Tab.Screen
        name="Herd"
        component={HerdScreen}
        options={{ tabBarLabel: 'Troupeau' }}
      />

      <Tab.Screen
        name="GPS"
        component={GPSScreen}
        options={{ tabBarLabel: 'GPS' }}
      />

      <Tab.Screen
        name="Predictions"
        component={PredictionsScreen}
        options={{ tabBarLabel: 'Prédictions' }}
      />

      <Tab.Screen
        name="Assistant"
        component={ChatbotScreen}
        options={{ tabBarLabel: 'Assistant' }}
      />
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    position: 'absolute',
    left: 14,
    right: 14,
    bottom: Platform.OS === 'ios' ? 18 : 12,
    height: 74,
    backgroundColor: 'rgba(27, 67, 50, 0.92)',
    borderTopWidth: 0,
    borderRadius: 28,
    paddingTop: 10,
    paddingBottom: Platform.OS === 'ios' ? 14 : 10,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.16)',
    shadowColor: '#1B4332',
    shadowOffset: { width: 0, height: 14 },
    shadowOpacity: 0.22,
    shadowRadius: 28,
    elevation: 18,
  },

  tabLabel: {
    fontSize: 10.5,
    fontWeight: '800',
    marginTop: 2,
  },

  tabItem: {
    paddingTop: 2,
  },

  iconWrapper: {
    minWidth: 44,
    minHeight: 34,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 18,
  },

  iconWrapperActive: {
    backgroundColor: 'rgba(242,201,76,0.18)',
  },
});