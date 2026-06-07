let rawData = [];
let groupedData = {};
let tagIds = [];
let index = 0;
let playing = false;
let interval = null;
let selectedTag = null;

let map;
let cowMarkers = {};
let predMarkers = {};

let drawing = false;
let pickingReference = false;
let currentZonePoints = [];
let zones = JSON.parse(localStorage.getItem("herdtrack_zones") || "{}");

let latRef = Number(localStorage.getItem("herdtrack_lat_ref")) || 36.8065;
let lonRef = Number(localStorage.getItem("herdtrack_lon_ref")) || 10.1815;

const R = 6371000;

const cowSelect = document.getElementById("cowSelect");
const slider = document.getElementById("slider");
const playBtn = document.getElementById("playBtn");

const useLocationBtn = document.getElementById("useLocationBtn");
const pickRefBtn = document.getElementById("pickRefBtn");
const refDisplay = document.getElementById("refDisplay");

const zoneTypeSelect = document.getElementById("zoneType");
const startDrawBtn = document.getElementById("startDrawBtn");
const saveZoneBtn = document.getElementById("saveZoneBtn");
const clearZonesBtn = document.getElementById("clearZonesBtn");

const ISOLATION_DISTANCE_CM = 300;
const ISOLATION_WARNING_CM = 450;
const ISOLATION_COUNT_MIN = 0;

function updateRefDisplay() {
  refDisplay.textContent = `${latRef.toFixed(6)}, ${lonRef.toFixed(6)}`;
}

function xyToGps(x_cm, y_cm) {
  const x_m = x_cm / 100;
  const y_m = y_cm / 100;

  const dLat = y_m / R;
  const dLon = x_m / (R * Math.cos(latRef * Math.PI / 180));

  const lat = latRef + dLat * 180 / Math.PI;
  const lon = lonRef + dLon * 180 / Math.PI;

  return [lon, lat];
}

function groupByTag(data) {
  const grouped = {};

  data.forEach(d => {
    if (!grouped[d.tag_id]) grouped[d.tag_id] = [];
    grouped[d.tag_id].push(d);
  });

  return grouped;
}

function computeMetrics() {
  const errors = rawData.map(d => d.error_cm).filter(e => Number.isFinite(e));

  const mae = errors.reduce((sum, e) => sum + e, 0) / errors.length;

  const rmse = Math.sqrt(
    errors.reduce((sum, e) => sum + e * e, 0) / errors.length
  );

  document.getElementById("mae").textContent = `${mae.toFixed(2)} cm`;
  document.getElementById("rmse").textContent = `${rmse.toFixed(2)} cm`;
}

function computeIsolationStatus(currentIndex) {
  const status = {};

  tagIds.forEach(tag => {
    status[tag] = { neighbors: 0 };
  });

  for (let i = 0; i < tagIds.length; i++) {
    const tagA = tagIds[i];
    const dataA = groupedData[tagA];
    if (!dataA || dataA.length <= currentIndex) continue;

    const pA = dataA[currentIndex];

    for (let j = i + 1; j < tagIds.length; j++) {
      const tagB = tagIds[j];
      const dataB = groupedData[tagB];
      if (!dataB || dataB.length <= currentIndex) continue;

      const pB = dataB[currentIndex];

      const dx = pA.real_x_cm - pB.real_x_cm;
      const dy = pA.real_y_cm - pB.real_y_cm;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist <= ISOLATION_DISTANCE_CM) {
        status[tagA].neighbors += 1;
        status[tagB].neighbors += 1;
      }
    }
  }

  return status;
}

function computePredictedIsolationRisk(tag) {
  const cowData = groupedData[tag];

  if (!cowData || cowData.length <= index) {
    return {
      level: "unknown",
      text: "Unknown",
      color: "#6c757d"
    };
  }

  const selectedCow = cowData[index];

  let minPredictedDistance = Infinity;

  tagIds.forEach(otherTag => {
    if (otherTag === tag) return;

    const otherData = groupedData[otherTag];
    if (!otherData || otherData.length <= index) return;

    const otherCow = otherData[index];

    const dx = selectedCow.pred_x_cm - otherCow.pred_x_cm;
    const dy = selectedCow.pred_y_cm - otherCow.pred_y_cm;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < minPredictedDistance) {
      minPredictedDistance = dist;
    }
  });

  if (!Number.isFinite(minPredictedDistance)) {
    return {
      level: "unknown",
      text: "Unknown",
      color: "#6c757d"
    };
  }

  if (minPredictedDistance > ISOLATION_WARNING_CM) {
    return {
      level: "high",
      text: `High risk (${minPredictedDistance.toFixed(1)} cm from nearest predicted cow)`,
      color: "#e63757"
    };
  }

  if (minPredictedDistance > ISOLATION_DISTANCE_CM) {
    return {
      level: "medium",
      text: `Medium risk (${minPredictedDistance.toFixed(1)} cm from nearest predicted cow)`,
      color: "#f39c12"
    };
  }

  return {
    level: "low",
    text: `Low risk (${minPredictedDistance.toFixed(1)} cm from nearest predicted cow)`,
    color: "#00b894"
  };
}

function pointInPolygon(point, polygon) {
  const x = point[0];
  const y = point[1];

  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0];
    const yi = polygon[i][1];
    const xj = polygon[j][0];
    const yj = polygon[j][1];

    const intersect =
      ((yi > y) !== (yj > y)) &&
      (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi);

    if (intersect) inside = !inside;
  }

  return inside;
}

function isInsideAuthorizedZone(lngLat) {
  if (!zones.authorized || zones.authorized.length < 3) {
    return true;
  }

  return pointInPolygon(lngLat, zones.authorized);
}

function getFutureRisk(p) {
  const predLngLat = xyToGps(p.pred_x_cm, p.pred_y_cm);

  if (!zones.authorized || zones.authorized.length < 3) {
    return {
      text: "No authorized zone defined",
      color: "#6c757d"
    };
  }

  const predictedInside = isInsideAuthorizedZone(predLngLat);

  if (!predictedInside) {
    return {
      text: "Risk of leaving authorized zone",
      color: "#e63757"
    };
  }

  return {
    text: "Low risk",
    color: "#00b894"
  };
}

function zoneColor(type) {
  if (type === "authorized") return "#00b894";
  if (type === "feeding") return "#27ae60";
  if (type === "resting") return "#3d523b";
  if (type === "restricted") return "#e63757";
  return "#3d523b";
}

function zoneFill(type) {
  if (type === "authorized") return "rgba(0,184,148,0.12)";
  if (type === "feeding") return "rgba(39,174,96,0.18)";
  if (type === "resting") return "rgba(61,82,59,0.16)";
  if (type === "restricted") return "rgba(230,55,87,0.18)";
  return "rgba(61,82,59,0.14)";
}

function zoneLabel(type) {
  if (type === "authorized") return "Authorized zone";
  if (type === "feeding") return "Feeding area";
  if (type === "resting") return "Resting area";
  if (type === "restricted") return "Restricted area";
  return type;
}

function refreshZonesLayer() {
  if (!map || !map.isStyleLoaded()) return;

  const features = [];

  Object.keys(zones).forEach(type => {
    const points = zones[type];

    if (points && points.length >= 3) {
      features.push({
        type: "Feature",
        properties: {
          type,
          name: zoneLabel(type),
          stroke: zoneColor(type),
          fill: zoneFill(type)
        },
        geometry: {
          type: "Polygon",
          coordinates: [[...points, points[0]]]
        }
      });
    }
  });

  if (currentZonePoints.length >= 2) {
    features.push({
      type: "Feature",
      properties: {
        type: "drawing",
        name: "Drawing zone",
        stroke: "#12263f",
        fill: "rgba(18,38,63,0.08)"
      },
      geometry: {
        type: "LineString",
        coordinates: currentZonePoints
      }
    });
  }

  const geojson = {
    type: "FeatureCollection",
    features
  };

  if (map.getSource("zones")) {
    map.getSource("zones").setData(geojson);
  } else {
    map.addSource("zones", {
      type: "geojson",
      data: geojson
    });

    map.addLayer({
      id: "zones-fill",
      type: "fill",
      source: "zones",
      filter: ["==", ["geometry-type"], "Polygon"],
      paint: {
        "fill-color": ["get", "fill"],
        "fill-opacity": 1
      }
    });

    map.addLayer({
      id: "zones-line",
      type: "line",
      source: "zones",
      paint: {
        "line-color": ["get", "stroke"],
        "line-width": 3
      }
    });
  }
}

function updatePanel(p, outside, isolated, neighbors) {
  const currentLngLat = xyToGps(p.real_x_cm, p.real_y_cm);
  const zoneRisk = getFutureRisk(p);
  const isolationRisk = computePredictedIsolationRisk(p.tag_id);

  document.getElementById("tagId").textContent = p.tag_id;
  document.getElementById("time").textContent = p.datetime;

  document.getElementById("gpsPos").textContent =
    `${currentLngLat[1].toFixed(6)}, ${currentLngLat[0].toFixed(6)}`;

  document.getElementById("realPos").textContent =
    `${p.real_x_cm.toFixed(1)}, ${p.real_y_cm.toFixed(1)} cm`;

  document.getElementById("predPos").textContent =
    `${p.pred_x_cm.toFixed(1)}, ${p.pred_y_cm.toFixed(1)} cm`;

  document.getElementById("error").textContent =
    `${p.error_cm.toFixed(1)} cm`;

  let statusText = "Inside authorized zone";
  let statusColor = "#00b894";

  if (outside) {
    statusText = "Outside authorized zone";
    statusColor = "#e63757";
  } else if (isolated) {
    statusText = `Inside zone / Isolated (${neighbors} neighbors)`;
    statusColor = "#f39c12";
  } else {
    statusText = `Inside zone / Connected (${neighbors} neighbors)`;
    statusColor = "#00b894";
  }

  document.getElementById("status").textContent = statusText;
  document.getElementById("status").style.color = statusColor;

  document.getElementById("futureRisk").textContent = zoneRisk.text;
  document.getElementById("futureRisk").style.color = zoneRisk.color;

  document.getElementById("isolationRisk").textContent = isolationRisk.text;
  document.getElementById("isolationRisk").style.color = isolationRisk.color;
}

function updateHerdSummary(iso) {
  let total = 0;
  let inside = 0;
  let outside = 0;
  let isolated = 0;

  tagIds.forEach(tag => {
    const cowData = groupedData[tag];
    if (!cowData || cowData.length <= index) return;

    total += 1;

    const p = cowData[index];
    const lngLat = xyToGps(p.real_x_cm, p.real_y_cm);
    const isOutside = !isInsideAuthorizedZone(lngLat);

    const neighbors = iso[tag] ? iso[tag].neighbors : 0;
    const isIsolated = neighbors <= ISOLATION_COUNT_MIN;

    if (isOutside) outside += 1;
    else inside += 1;

    if (!isOutside && isIsolated) isolated += 1;
  });

  const alert = outside > 0 || isolated > 0;

  document.getElementById("totalCows").textContent = total;
  document.getElementById("insideCount").textContent = inside;
  document.getElementById("outsideCount").textContent = outside;
  document.getElementById("isolatedCount").textContent = isolated;

  const herdStatus = document.getElementById("herdStatus");
  herdStatus.textContent = alert ? "ALERT" : "NORMAL";
  herdStatus.style.color = alert ? "#e63757" : "#00b894";
  herdStatus.style.fontWeight = "bold";
}

function updateTrajectoryLayers() {
  const cowData = groupedData[selectedTag];
  if (!cowData || cowData.length <= index) return;

  const start = Math.max(0, index - 30);
  const coords = [];

  for (let i = start; i <= index; i++) {
    const p = cowData[i];
    coords.push(xyToGps(p.real_x_cm, p.real_y_cm));
  }

  const current = cowData[index];

  const predictionLine = [
    xyToGps(current.real_x_cm, current.real_y_cm),
    xyToGps(current.pred_x_cm, current.pred_y_cm)
  ];

  const trajectoryData = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: coords
        },
        properties: {}
      }
    ]
  };

  const predLineData = {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: predictionLine
        },
        properties: {}
      }
    ]
  };

  if (map.getSource("selectedTrajectory")) {
    map.getSource("selectedTrajectory").setData(trajectoryData);
  } else {
    map.addSource("selectedTrajectory", {
      type: "geojson",
      data: trajectoryData
    });

    map.addLayer({
      id: "selectedTrajectoryLayer",
      type: "line",
      source: "selectedTrajectory",
      paint: {
        "line-color": "#3d523b",
        "line-width": 4
      }
    });
  }

  if (map.getSource("predictionLine")) {
    map.getSource("predictionLine").setData(predLineData);
  } else {
    map.addSource("predictionLine", {
      type: "geojson",
      data: predLineData
    });

    map.addLayer({
      id: "predictionLineLayer",
      type: "line",
      source: "predictionLine",
      paint: {
        "line-color": "#3d523b",
        "line-width": 2,
        "line-dasharray": [2, 2]
      }
    });
  }
}

function updateMap() {
  if (!map || tagIds.length === 0) return;

  const iso = computeIsolationStatus(index);
  updateHerdSummary(iso);
  refreshZonesLayer();

  tagIds.forEach(tag => {
    const cowData = groupedData[tag];
    if (!cowData || cowData.length <= index) return;

    const p = cowData[index];

    const currentLngLat = xyToGps(p.real_x_cm, p.real_y_cm);
    const predLngLat = xyToGps(p.pred_x_cm, p.pred_y_cm);

    const outside = !isInsideAuthorizedZone(currentLngLat);
    const neighbors = iso[tag] ? iso[tag].neighbors : 0;
    const isolated = neighbors <= ISOLATION_COUNT_MIN;

    let color = "#00b894";
    if (outside) color = "#e63757";
    else if (isolated) color = "#f39c12";

    if (!cowMarkers[tag]) {
      const el = document.createElement("div");
      el.className = "cow-marker";
      el.innerHTML = "🐄";
      el.style.background = color;

      el.addEventListener("click", () => {
        selectedTag = tag;
        cowSelect.value = tag;
        updateMap();
      });

      cowMarkers[tag] = new maplibregl.Marker(el)
        .setLngLat(currentLngLat)
        .addTo(map);
    } else {
      cowMarkers[tag].setLngLat(currentLngLat);
      cowMarkers[tag].getElement().style.background = color;
    }

    cowMarkers[tag].getElement().classList.toggle("selected", tag === selectedTag);

    if (!predMarkers[tag]) {
      const predEl = document.createElement("div");
      predEl.className = "pred-marker";

      predMarkers[tag] = new maplibregl.Marker(predEl)
        .setLngLat(predLngLat)
        .addTo(map);
    } else {
      predMarkers[tag].setLngLat(predLngLat);
    }

    if (tag === selectedTag) {
      updatePanel(p, outside, isolated, neighbors);
    }
  });

  updateTrajectoryLayers();
}

function initMap() {
  map = new maplibregl.Map({
    container: "map",
    style: "https://tiles.openfreemap.org/styles/liberty",
    center: [lonRef, latRef],
    zoom: 19
  });

  // Liberty style references POI icons not always present in the sprite sheet.
  map.on("styleimagemissing", e => {
    if (map.hasImage(e.id)) return;
    const size = 1;
    map.addImage(e.id, {
      width: size,
      height: size,
      data: new Uint8Array(size * size * 4)
    });
  });

  map.addControl(new maplibregl.NavigationControl());

  map.on("load", () => {
    refreshZonesLayer();
    updateMap();
    updateRefDisplay();
  });

  map.on("click", e => {
    if (pickingReference) {
      latRef = e.lngLat.lat;
      lonRef = e.lngLat.lng;

      localStorage.setItem("herdtrack_lat_ref", latRef);
      localStorage.setItem("herdtrack_lon_ref", lonRef);

      pickingReference = false;
      pickRefBtn.textContent = "Set reference by clicking map";

      updateRefDisplay();
      reloadMarkersAfterRefChange();
      return;
    }

    if (!drawing) return;

    currentZonePoints.push([e.lngLat.lng, e.lngLat.lat]);
    refreshZonesLayer();
  });
}

function reloadMarkersAfterRefChange() {
  Object.values(cowMarkers).forEach(marker => marker.remove());
  Object.values(predMarkers).forEach(marker => marker.remove());

  cowMarkers = {};
  predMarkers = {};

  if (map) {
    map.setCenter([lonRef, latRef]);
    map.setZoom(19);
  }

  updateMap();
}

fetch("predictions_all.json")
  .then(res => res.json())
  .then(data => {
    rawData = data;
    groupedData = groupByTag(data);
    tagIds = Object.keys(groupedData);

    tagIds.forEach(tag => {
      const option = document.createElement("option");
      option.value = tag;
      option.textContent = tag;
      cowSelect.appendChild(option);
    });

    selectedTag = tagIds[0];
    cowSelect.value = selectedTag;
    slider.max = groupedData[selectedTag].length - 1;

    computeMetrics();
    initMap();
    updateRefDisplay();
  });

useLocationBtn.addEventListener("click", () => {
  if (!navigator.geolocation) {
    alert("Geolocation is not supported by this browser.");
    return;
  }

  navigator.geolocation.getCurrentPosition(
    position => {
      latRef = position.coords.latitude;
      lonRef = position.coords.longitude;

      localStorage.setItem("herdtrack_lat_ref", latRef);
      localStorage.setItem("herdtrack_lon_ref", lonRef);

      updateRefDisplay();
      reloadMarkersAfterRefChange();
    },
    () => {
      alert("Unable to get your location. You can use 'Set reference by clicking map' instead.");
    }
  );
});

pickRefBtn.addEventListener("click", () => {
  pickingReference = true;
  drawing = false;

  pickRefBtn.textContent = "Click on map...";
  startDrawBtn.textContent = "Start drawing zone";
});

startDrawBtn.addEventListener("click", () => {
  drawing = true;
  pickingReference = false;

  currentZonePoints = [];
  startDrawBtn.textContent = "Drawing...";
  pickRefBtn.textContent = "Set reference by clicking map";
});

saveZoneBtn.addEventListener("click", () => {
  const type = zoneTypeSelect.value;

  if (currentZonePoints.length < 3) {
    alert("Draw at least 3 points.");
    return;
  }

  zones[type] = currentZonePoints;
  localStorage.setItem("herdtrack_zones", JSON.stringify(zones));

  drawing = false;
  currentZonePoints = [];
  startDrawBtn.textContent = "Start drawing zone";

  refreshZonesLayer();
  updateMap();
});

clearZonesBtn.addEventListener("click", () => {
  zones = {};
  currentZonePoints = [];
  localStorage.removeItem("herdtrack_zones");

  refreshZonesLayer();
  updateMap();
});

cowSelect.addEventListener("change", () => {
  selectedTag = cowSelect.value;
  slider.max = groupedData[selectedTag].length - 1;

  if (index > slider.max) {
    index = 0;
    slider.value = 0;
  }

  updateMap();
});

slider.addEventListener("input", () => {
  index = Number(slider.value);
  updateMap();
});

playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";

  if (playing) {
    interval = setInterval(() => {
      index++;

      if (index >= groupedData[selectedTag].length) {
        index = 0;
      }

      slider.value = index;
      updateMap();
    }, 120);
  } else {
    clearInterval(interval);
  }
});

/* Slide-in drawers: full map by default; farm setup & cow/herd toggles */
(function setupDrawers() {
  const farmDrawer = document.getElementById("farmDrawer");
  const detailsDrawer = document.getElementById("detailsDrawer");
  const backdrop = document.getElementById("drawerBackdrop");
  const btnFarm = document.getElementById("btnFarmSetup");
  const btnDetails = document.getElementById("btnDetails");
  const closeFarmBtn = document.getElementById("closeFarmDrawer");
  const closeDetailsBtn = document.getElementById("closeDetailsDrawer");

  if (!farmDrawer || !detailsDrawer || !backdrop || !btnFarm || !btnDetails) return;

  function triggerMapResize() {
    if (typeof map !== "undefined" && map && typeof map.resize === "function") {
      setTimeout(() => map.resize(), 280);
    }
  }

  function updateBackdrop() {
    const farmOpen = farmDrawer.classList.contains("is-open");
    const detOpen = detailsDrawer.classList.contains("is-open");
    backdrop.classList.toggle("is-visible", farmOpen || detOpen);
    btnFarm.classList.toggle("is-active", farmOpen);
    btnDetails.classList.toggle("is-active", detOpen);
    farmDrawer.setAttribute("aria-hidden", farmOpen ? "false" : "true");
    detailsDrawer.setAttribute("aria-hidden", detOpen ? "false" : "true");
    backdrop.setAttribute("aria-hidden", farmOpen || detOpen ? "false" : "true");
    triggerMapResize();
  }

  function closeFarmDrawer() {
    farmDrawer.classList.remove("is-open");
    updateBackdrop();
  }

  function closeDetailsDrawer() {
    detailsDrawer.classList.remove("is-open");
    updateBackdrop();
  }

  function toggleFarmDrawer() {
    const opening = !farmDrawer.classList.contains("is-open");
    farmDrawer.classList.toggle("is-open", opening);
    if (opening) detailsDrawer.classList.remove("is-open");
    updateBackdrop();
  }

  function toggleDetailsDrawer() {
    const opening = !detailsDrawer.classList.contains("is-open");
    detailsDrawer.classList.toggle("is-open", opening);
    if (opening) farmDrawer.classList.remove("is-open");
    updateBackdrop();
  }

  btnFarm.addEventListener("click", toggleFarmDrawer);
  btnDetails.addEventListener("click", toggleDetailsDrawer);
  closeFarmBtn.addEventListener("click", closeFarmDrawer);
  closeDetailsBtn.addEventListener("click", closeDetailsDrawer);
  backdrop.addEventListener("click", () => {
    closeFarmDrawer();
    closeDetailsDrawer();
  });

  updateBackdrop();
})();