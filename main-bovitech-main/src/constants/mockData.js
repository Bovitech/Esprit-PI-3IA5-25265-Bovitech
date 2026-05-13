// src/constants/mockData.js

export const COWS = [
  {
    id: 'COW001',
    name: 'Lina',
    tag: '#001',
    breed: 'Holstein',
    age: 4,
    weight: 580,
    status: 'healthy',
    collarId: 'CLR-001',
    lastUpdate: '2025-03-24T08:30:00',
    location: { latitude: 36.8065, longitude: 10.1815 },
    health: {
      temperature: 38.4,
      activityLevel: 85,
      rumination: 420,
      heartRate: 72,
    },
    milk: {
      today: 28.5,
      yesterday: 27.8,
      weekly: [26, 27, 28, 27.5, 28, 28.5, 28.5],
    },
    reproduction: {
      lastEstrus: '2025-02-10',
      nextEstrus: '2025-03-24',
      status: 'in_estrus',
    },
    alerts: [],
  },
  {
    id: 'COW002',
    name: 'Farah',
    tag: '#002',
    breed: 'Brune',
    age: 5,
    weight: 610,
    status: 'alert',
    collarId: 'CLR-002',
    lastUpdate: '2025-03-24T08:25:00',
    location: { latitude: 36.8072, longitude: 10.1822 },
    health: {
      temperature: 39.8,
      activityLevel: 42,
      rumination: 180,
      heartRate: 88,
    },
    milk: {
      today: 18.2,
      yesterday: 24.1,
      weekly: [24, 23, 22, 24, 21, 20, 18],
    },
    reproduction: {
      lastEstrus: '2025-01-15',
      nextEstrus: '2025-04-05',
      status: 'normal',
    },
    alerts: [
      { id: 'A1', type: 'health', severity: 'high', message: 'Température élevée détectée (39.8°C)', time: '08:20' },
      { id: 'A2', type: 'milk', severity: 'medium', message: 'Baisse de production de 25%', time: '07:00' },
    ],
  },
  {
    id: 'COW003',
    name: 'Samira',
    tag: '#003',
    breed: 'Holstein',
    age: 3,
    weight: 520,
    status: 'healthy',
    collarId: 'CLR-003',
    lastUpdate: '2025-03-24T08:28:00',
    location: { latitude: 36.8058, longitude: 10.1808 },
    health: {
      temperature: 38.2,
      activityLevel: 90,
      rumination: 450,
      heartRate: 68,
    },
    milk: {
      today: 22.1,
      yesterday: 21.8,
      weekly: [20, 21, 21.5, 22, 21.8, 22, 22.1],
    },
    reproduction: {
      lastEstrus: '2025-03-03',
      nextEstrus: '2025-04-14',
      status: 'pregnant',
    },
    alerts: [],
  },
  {
    id: 'COW004',
    name: 'Nour',
    tag: '#004',
    breed: 'Montbéliarde',
    age: 6,
    weight: 650,
    status: 'out_of_zone',
    collarId: 'CLR-004',
    lastUpdate: '2025-03-24T08:15:00',
    location: { latitude: 36.8120, longitude: 10.1890 },
    health: {
      temperature: 38.6,
      activityLevel: 78,
      rumination: 390,
      heartRate: 74,
    },
    milk: {
      today: 30.2,
      yesterday: 30.8,
      weekly: [31, 30.5, 30, 30.8, 30.5, 30.8, 30.2],
    },
    reproduction: {
      lastEstrus: '2025-02-20',
      nextEstrus: '2025-04-01',
      status: 'normal',
    },
    alerts: [
      { id: 'A3', type: 'security', severity: 'high', message: 'Vache hors zone de pâturage', time: '08:10' },
    ],
  },
];

export const FARM_STATS = {
  totalCows: 4,
  healthyCows: 2,
  alertCows: 1,
  outOfZone: 1,
  totalMilkToday: 99.0,
  totalMilkYesterday: 104.5,
  avgTemperature: 38.75,
};

export const SAFE_ZONE = {
  center: { latitude: 36.8065, longitude: 10.1815 },
  radius: 500, // meters
};
