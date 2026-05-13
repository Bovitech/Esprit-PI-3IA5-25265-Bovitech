import React, { memo, useRef } from 'react';
import { Animated, StyleSheet, Text, TouchableOpacity } from 'react-native';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

const COLORS = {
  primary: '#2E412D',
  cream: '#F6F6EC',
  sage: '#AFB2A1',
  dark: '#3E3E32',
  white: '#FFFFFF',
  border: '#E5E3D6',
};

const QuickAction = memo(function QuickAction({ icon, label, onPress, lib = 'ion' }) {
  const scale = useRef(new Animated.Value(1)).current;

  const pressIn = () => {
    Animated.spring(scale, {
      toValue: 0.96,
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
    <Animated.View style={[styles.wrap, { transform: [{ scale }] }]}>
      <TouchableOpacity
        style={styles.card}
        activeOpacity={0.88}
        onPress={onPress}
        onPressIn={pressIn}
        onPressOut={pressOut}
      >
        {lib === 'mc' ? (
          <MaterialCommunityIcons name={icon} size={18} color={COLORS.primary} />
        ) : (
          <Ionicons name={icon} size={18} color={COLORS.primary} />
        )}

        <Text style={styles.label} numberOfLines={1}>
          {label}
        </Text>
      </TouchableOpacity>
    </Animated.View>
  );
});

const styles = StyleSheet.create({
  wrap: {
    width: '48%',
  },

  card: {
    height: 44,
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.86)',
    borderWidth: 1,
    borderColor: COLORS.border,
    paddingHorizontal: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.07,
    shadowRadius: 16,
    elevation: 3,
  },

  label: {
    flex: 1,
    fontFamily: 'Inter_700Bold',
    fontSize: 12,
    color: COLORS.dark,
  },
});

export default QuickAction;