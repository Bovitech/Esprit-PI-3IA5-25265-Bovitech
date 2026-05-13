const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// npm on Windows can leave a transient `.react-native-webview-*` folder; if it is removed
// while Metro is crawling, the watcher throws ENOENT and Expo exits. Ignore that path.
config.resolver.blockList = [
  ...(config.resolver.blockList ?? []),
  /[/\\]node_modules[/\\]\.react-native-webview-[^/\\]+[/\\]/,
];

module.exports = config;
