import streamlit as st
import osmnx as ox
import folium
from streamlit_folium import st_folium
import numpy as np
import networkx as nx
import geopandas as gpd
import pandas as pd
import json
import math
from shapely import wkt
from shapely.ops import nearest_points
from shapely.geometry import Polygon, MultiPolygon
try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    _make_valid = None
from sklearn.cluster import KMeans
from shapely.geometry import Point, MultiPoint

# Настройки страницы
st.set_page_config(page_title="Центры доступности районов", layout="wide")
st.markdown(
    """
<style>
.leaflet-control-attribution { 
  font-size: 9px !important; 
  opacity: 0.5 !important; 
  padding-right: 8px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("### Оценка нового общественного центра в административном районе")
st.sidebar.header("Параметры")
if st.sidebar.button("Сбросить кэш"):
    st.cache_data.clear()
    st.session_state["map_obj"] = None
    st.session_state["last_error"] = None
    st.sidebar.success("Кэш очищен")

# Параметры интерфейса
st.sidebar.subheader("Район")
district_options = {
    "Адмиралтейский": "Admiralteysky District, Saint Petersburg, Russia",
    "Василеостровский": "Vasileostrovsky District, Saint Petersburg, Russia",
    "Выборгский": "Vyborgsky District, Saint Petersburg, Russia",
    "Калининский": "Kalininsky District, Saint Petersburg, Russia",
    "Кировский": "Kirovsky District, Saint Petersburg, Russia",
    "Колпинский": "Kolpinsky District, Saint Petersburg, Russia",
    "Красногвардейский": "Krasnogvardeysky District, Saint Petersburg, Russia",
    "Красносельский": "Krasnoselsky District, Saint Petersburg, Russia",
    "Кронштадтский": "Kronshtadtsky District, Saint Petersburg, Russia",
    "Курортный": "Kurortny District, Saint Petersburg, Russia",
    "Московский": "Moskovsky District, Saint Petersburg, Russia",
    "Невский": "Nevsky District, Saint Petersburg, Russia",
    "Петроградский": "Petrogradsky District, Saint Petersburg, Russia",
    "Петродворцовый": "Petrodvortsovy District, Saint Petersburg, Russia",
    "Приморский": "Primorsky District, Saint Petersburg, Russia",
    "Пушкинский": "Pushkinsky District, Saint Petersburg, Russia",
    "Фрунзенский": "Frunzensky District, Saint Petersburg, Russia",
    "Центральный": "Tsentralny District, Saint Petersburg, Russia",
}
district_population = {
    "Адмиралтейский": 157000,
    "Василеостровский": 215000,
    "Выборгский": 538000,
    "Калининский": 540000,
    "Кировский": 340000,
    "Колпинский": 193000,
    "Красногвардейский": 356000,
    "Красносельский": 408000,
    "Кронштадтский": 44000,
    "Курортный": 79000,
    "Московский": 356000,
    "Невский": 515000,
    "Петроградский": 130000,
    "Петродворцовый": 149000,
    "Приморский": 660000,
    "Пушкинский": 247000,
    "Фрунзенский": 395000,
    "Центральный": 226000,
}
selected_district_ru = st.sidebar.selectbox("Выберите район", list(district_options.keys()), index=0)
selected_district = district_options[selected_district_ru]
pop = district_population.get(selected_district_ru, 0)
st.sidebar.markdown(
    f"<div style='line-height:1.05; font-size:12px;'>Население района: {pop:,}</div>".replace(",", " "),
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='line-height:1.05; font-size:12px;'>Расчет: 1 центр на 200 000 человек.</div>",
    unsafe_allow_html=True,
)
if pop > 0:
    rec_centers = max(1, int(np.ceil(pop / 200000)))
else:
    rec_centers = None
coverage = st.session_state.get("last_coverage_pct")
if rec_centers is not None or coverage is not None:
    rec_text = f"{rec_centers}" if rec_centers is not None else "—"
    cov_text = f"{coverage:.0f}%" if coverage is not None else "—"
    st.sidebar.markdown(
        f"<div style='line-height:1.1; font-size:13px;'>Рекомендуемое число центров: "
        f"<b style='font-size:13px;'>{rec_text}</b></div>"
        f"<div style='line-height:1.1; font-size:13px;'>Покрытие жилых домов: "
        f"<b style='font-size:13px;'>{cov_text}</b></div>",
        unsafe_allow_html=True,
    )

st.sidebar.subheader("Центральные места")
walking_time = st.sidebar.slider("Макс. время пешей доступности (мин)", 10, 60, 30)
base_centers = st.sidebar.slider("Число центров", 1, 6, 2)
if st.sidebar.button("Рассчитать"):
    st.session_state["run_calc"] = True
fast_mode = st.sidebar.checkbox("Быстрый режим (прямые расстояния)", value=True)
show_isochrones = st.sidebar.checkbox("Показывать зону пешей доступности", value=True)
show_clusters = st.sidebar.checkbox("Показывать кластеризацию (Weighted K-means)", value=True)
show_candidate_zones = st.sidebar.checkbox("Показывать кандидатские зоны (зелёные)", value=True)

activity_minutes = st.session_state.get("activity_minutes", 10)
transport_minutes = st.session_state.get("transport_minutes", 15)

# Параметры расстояний
speed_kmh = 4.5
meters_per_min = (speed_kmh * 1000) / 60
max_dist = walking_time * meters_per_min
activity_radius = activity_minutes * meters_per_min
transport_radius = transport_minutes * meters_per_min

# Радиусы зон центричности зависят от выбранного времени пешей доступности.
small_minutes = max(2.5, walking_time / 6)
large_minutes = max(5.0, walking_time / 3)
base_r = small_minutes * meters_per_min

# Загрузка данных OpenStreetMap
@st.cache_data(show_spinner=False)
def load_area(dist_name: str) -> gpd.GeoDataFrame:
    return ox.geocode_to_gdf(dist_name).to_crs(epsg=3857)


@st.cache_data(show_spinner=False)
def load_buildings(dist_name: str) -> gpd.GeoDataFrame:
    buildings = ox.features_from_place(dist_name, tags={"building": True})
    return buildings.to_crs(epsg=3857)


@st.cache_data(show_spinner=False)
def load_graph(dist_name: str):
    # graph_from_place загружает пешеходный граф сразу по границе района.
    G = ox.graph_from_place(dist_name, network_type="walk", simplify=True)
    return ox.project_graph(G, to_crs=3857)


# Геообработка
def _nearest_node(G, x, y):
    return ox.distance.nearest_nodes(G, x, y)

def _fix_geom(geom):
    g = geom
    try:
        if _make_valid is not None:
            g = _make_valid(g)
    except Exception:
        pass
    try:
        if hasattr(g, "is_valid") and not g.is_valid:
            g = g.buffer(0)
    except Exception:
        pass
    try:
        if hasattr(g, "is_empty") and g.is_empty:
            g = g.convex_hull
    except Exception:
        pass
    return g


def _isochrone_for_center(G, center_xy, travel_dist, clip_poly):
    node = _nearest_node(G, center_xy[0], center_xy[1])
    subgraph = nx.ego_graph(G, node, radius=travel_dist, distance="length")
    if subgraph.number_of_nodes() == 0:
        return None

    try:
        nodes_gdf = ox.graph_to_gdfs(
            subgraph,
            nodes=True,
            edges=False,
            node_geometry=True,
            fill_edge_geometry=False,
        )
        edges_gdf = ox.graph_to_gdfs(
            subgraph,
            nodes=False,
            edges=True,
            node_geometry=False,
            fill_edge_geometry=True,
        )
    except Exception:
        return None

    all_geoms = []
    if not nodes_gdf.empty:
        all_geoms.extend(list(nodes_gdf.buffer(40).geometry))
    if not edges_gdf.empty:
        all_geoms.extend(list(edges_gdf.buffer(25).geometry))
    if not all_geoms:
        return None

    try:
        union = gpd.GeoSeries(all_geoms, crs=3857).unary_union
    except Exception:
        return None
    if union.is_empty:
        return None

    # Изохрона приводится к цельной области без внутренних дыр.
    try:
        if union.geom_type == "Polygon":
            union = Polygon(union.exterior)
        elif union.geom_type == "MultiPolygon":
            union = MultiPolygon([Polygon(p.exterior) for p in union.geoms if not p.is_empty])
    except Exception:
        pass

    union = _fix_geom(union)

    if clip_poly is not None:
        try:
            clipped = union.intersection(clip_poly)
        except Exception:
            clipped = union
    else:
        clipped = union

    if clipped.is_empty:
        return None
    return clipped


def _build_coords_weights(buildings_gdf):
    residential_buildings = {
        "residential",
        "apartments",
        "house",
        "detached",
        "semidetached_house",
        "terrace",
        "dormitory",
        "bungalow",
        "static_caravan",
    }
    non_residential_buildings = {
        "industrial",
        "warehouse",
        "commercial",
        "retail",
        "garage",
        "garages",
        "storage",
        "shed",
        "hangar",
        "farm",
        "greenhouse",
        "roof",
        "service",
        "construction",
        "school",
        "hospital",
        "kindergarten",
        "university",
        "office",
        "transportation",
        "train_station",
        "carport",
        "cowshed",
        "stable",
    }
    non_res_landuse = {
        "industrial",
        "commercial",
        "railway",
        "garages",
        "depot",
        "construction",
    }

    def _parse_level(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().split(";")[0]
        return pd.to_numeric(value, errors="coerce")

    def _parse_height(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip().split(";")[0].replace("m", "")
        return pd.to_numeric(value, errors="coerce")

    valid_rows = []
    for _, row in buildings_gdf.iterrows():
        geom = row.geometry
        if geom is None or not geom.is_valid:
            continue

        b = str(row.get("building", "")).lower()
        landuse = str(row.get("landuse", "")).lower()
        b_use = str(row.get("building:use", "")).lower()
        b_purpose = str(row.get("building:purpose", "")).lower()

        if b in non_residential_buildings:
            continue
        if landuse in non_res_landuse:
            continue
        if any(x in b_use for x in ["industrial", "warehouse", "garage", "storage", "commercial"]):
            continue
        if any(x in b_purpose for x in ["industrial", "warehouse", "garage", "storage", "commercial"]):
            continue

        if b in residential_buildings:
            valid_rows.append(row)
            continue

        # Нераспознанный тип сохраняется только при явных жилых тегах.
        if any(x in b_use for x in ["residential", "apartments", "house", "dormitory", "dwelling"]):
            valid_rows.append(row)
            continue
        if any(x in b_purpose for x in ["residential", "apartments", "house", "dormitory", "dwelling"]):
            valid_rows.append(row)
            continue

    if not valid_rows:
        return np.empty((0, 2)), None, []

    geoms = [r.geometry for r in valid_rows]
    coords = np.array([[g.centroid.x, g.centroid.y] for g in geoms])

    # Вес здания - приближение численности жителей: площадь * этажность.
    areas = []
    for row in valid_rows:
        g = row.geometry
        try:
            a = float(g.area)
        except Exception:
            a = 0.0
        levels = _parse_level(row.get("building:levels"))
        if pd.isna(levels):
            levels = _parse_level(row.get("levels"))
        height = _parse_height(row.get("height"))
        if pd.notna(height):
            levels_from_height = max(height / 3.0, 1.0)
            levels = max(levels if pd.notna(levels) else 1.0, levels_from_height)
        if pd.isna(levels) or levels is None:
            levels = 1.0
        areas.append(max(a, 1.0) * float(levels))
    weights = np.array(areas)

    building_rows = []
    for row in valid_rows:
        building_rows.append(
            {
                "geometry": row.geometry,
                "name": row.get("name") or "",
                "address": _format_address(row),
                "description": row.get("description") or row.get("note") or "",
                "osm_id": row.get("osmid") or row.get("osm_id") or "",
            }
        )

    return coords, weights, building_rows


def _build_all_buildings(buildings_gdf):
    rows = []
    for _, row in buildings_gdf.iterrows():
        geom = row.geometry
        if geom is None or not geom.is_valid:
            continue
        rows.append(
            {
                "geometry": geom,
                "name": row.get("name") or "",
                "address": _format_address(row),
                "description": row.get("description") or row.get("note") or "",
                "osm_id": row.get("osmid") or row.get("osm_id") or "",
                "category": row.get("building") or "",
            }
        )
    return rows


def _kmeans_centers_labels(coords, weights, k):
    k = max(1, min(int(k), coords.shape[0]))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    if weights is not None:
        kmeans.fit(coords, sample_weight=weights)
    else:
        kmeans.fit(coords)
    return kmeans.cluster_centers_, kmeans.labels_


def _coverage_ratio(G, coords, centers, travel_dist, use_fast=False):
    if coords.size == 0:
        return 0.0
    if use_fast:
        # Быстрый режим считает покрытие по прямым расстояниям в метрах.
        centers_arr = np.array(centers)
        if centers_arr.size == 0:
            return 0.0
        diffs = coords[:, None, :] - centers_arr[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))
        covered = (dists.min(axis=1) <= travel_dist).sum()
        return covered / coords.shape[0] * 100.0

    nodes = [
        ox.distance.nearest_nodes(G, coords[i, 0], coords[i, 1])
        for i in range(coords.shape[0])
    ]
    center_nodes = [ox.distance.nearest_nodes(G, c[0], c[1]) for c in centers]

    covered = 0
    for n in nodes:
        best = None
        for cn in center_nodes:
            try:
                d = nx.shortest_path_length(G, n, cn, weight="length")
            except Exception:
                d = None
            if d is None:
                continue
            if best is None or d < best:
                best = d
        if best is not None and best <= travel_dist:
            covered += 1

    return covered / coords.shape[0] * 100.0


def compute_centers_and_isochrones(dist_name, base_k, extra_k, travel_dist, target_coverage, use_fast=False):
    area_gdf = load_area(dist_name)
    geom = area_gdf.geometry.iloc[0]
    poly = _fix_geom(geom)
    if hasattr(poly, "is_empty") and poly.is_empty:
        poly = None

    buildings = load_buildings(dist_name)
    coords, weights, building_rows = _build_coords_weights(buildings)
    all_building_rows = _build_all_buildings(buildings)
    if coords.shape[0] == 0:
        return area_gdf, [], [], 0.0, [], None, all_building_rows, np.array([])

    G = load_graph(dist_name)

    centers, labels = _kmeans_centers_labels(coords, weights, base_k)
    coverage = _coverage_ratio(G, coords, centers, travel_dist, use_fast=use_fast)

    isochrones = []
    for c in centers:
        iso = _isochrone_for_center(G, c, travel_dist, poly)
        if iso is not None:
            isochrones.append(iso)

    candidate_zone = None
    if isochrones:
        try:
            union = gpd.GeoSeries(isochrones, crs=3857).unary_union
            union = _fix_geom(union)

            # Кандидатская зона ограничивает ядро вокруг центров областью пешей доступности.
            center_buffers = []
            core_radius = max(travel_dist * 0.4, 300)
            for c in centers:
                center_buffers.append(Point(c[0], c[1]).buffer(core_radius))
            core = gpd.GeoSeries(center_buffers, crs=3857).unary_union
            core = _fix_geom(core)

            try:
                inner = core.intersection(union)
            except Exception:
                inner = core

            if poly is not None:
                try:
                    candidate_zone = inner.intersection(poly)
                except Exception:
                    candidate_zone = inner
            else:
                candidate_zone = inner
            if candidate_zone is not None and candidate_zone.is_empty:
                candidate_zone = None
        except Exception:
            candidate_zone = None

    return area_gdf, isochrones, centers, coverage, building_rows, candidate_zone, all_building_rows, labels


# Состояние приложения и визуализация
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None
if "selected_feature" not in st.session_state:
    st.session_state["selected_feature"] = None
if "selected_feature_id" not in st.session_state:
    st.session_state["selected_feature_id"] = None
if "show_addresses" not in st.session_state:
    st.session_state["show_addresses"] = False
if "addresses_df" not in st.session_state:
    st.session_state["addresses_df"] = None
if "identity_score" not in st.session_state:
    st.session_state["identity_score"] = None
if "last_coverage_pct" not in st.session_state:
    st.session_state["last_coverage_pct"] = None
if "run_calc" not in st.session_state:
    st.session_state["run_calc"] = False
if "centricity_base_r" not in st.session_state:
    st.session_state["centricity_base_r"] = None
if "calc_id" not in st.session_state:
    st.session_state["calc_id"] = 0
if "last_calc_district" not in st.session_state:
    st.session_state["last_calc_district"] = None
if "zone_center" not in st.session_state:
    st.session_state["zone_center"] = None
if "zone_base_r" not in st.session_state:
    st.session_state["zone_base_r"] = None
if "activity_score" not in st.session_state:
    st.session_state["activity_score"] = None
if "activity_count" not in st.session_state:
    st.session_state["activity_count"] = None
if "activity_counts" not in st.session_state:
    st.session_state["activity_counts"] = None
if "activity_pois" not in st.session_state:
    st.session_state["activity_pois"] = None
if "centricity_score" not in st.session_state:
    st.session_state["centricity_score"] = None
if "transport_score" not in st.session_state:
    st.session_state["transport_score"] = None
if "transport_count" not in st.session_state:
    st.session_state["transport_count"] = None
if "transport_counts" not in st.session_state:
    st.session_state["transport_counts"] = None
if "transport_pois" not in st.session_state:
    st.session_state["transport_pois"] = None
if "selected_osm_id" not in st.session_state:
    st.session_state["selected_osm_id"] = None
if "tagged_buildings" not in st.session_state:
    st.session_state["tagged_buildings"] = None
if "tagged_context_id" not in st.session_state:
    st.session_state["tagged_context_id"] = None
if "hide_small_buildings" not in st.session_state:
    st.session_state["hide_small_buildings"] = False
if "hidden_tagged_ids" not in st.session_state:
    st.session_state["hidden_tagged_ids"] = set()

def _base_map():
    m = folium.Map(location=[59.94, 30.28], zoom_start=12, tiles=None)
    folium.TileLayer(
        tiles="cartodbpositron",
        name="Basemap",
        attr="OpenStreetMap",
    ).add_to(m)
    return m


@st.cache_data(show_spinner=False)
def load_features_in_zone(zone_geojson: str) -> gpd.GeoDataFrame:
    try:
        data = json.loads(zone_geojson)
        zone_gdf = gpd.GeoDataFrame.from_features(data["features"], crs=4326)
    except Exception:
        return gpd.GeoDataFrame()
    if zone_gdf.empty:
        return gpd.GeoDataFrame()
    try:
        zone_geom = zone_gdf.geometry.unary_union
    except Exception:
        return gpd.GeoDataFrame()

    tags = {
        "building": True,
        "amenity": True,
        "shop": True,
        "office": True,
        "leisure": True,
        "tourism": True,
        "healthcare": True,
        "public_transport": True,
    }
    try:
        gdf = ox.features_from_polygon(zone_geom, tags=tags)
        return gdf
    except Exception:
        return gpd.GeoDataFrame()


def _format_address(row) -> str:
    parts = []
    street = row.get("addr:street")
    number = row.get("addr:housenumber")
    city = row.get("addr:city")
    postcode = row.get("addr:postcode")
    if street:
        parts.append(str(street))
    if number:
        parts.append(str(number))
    if city:
        parts.append(str(city))
    if postcode:
        parts.append(str(postcode))
    return ", ".join(parts) if parts else ""


@st.cache_data(show_spinner=False)
def load_activity_pois(lat: float, lon: float, radius_m: int) -> gpd.GeoDataFrame:
    try:
        ox.settings.timeout = 60
        ox.settings.requests_timeout = 30
        ox.settings.overpass_settings = "[out:json][timeout:25]"
    except Exception:
        pass
    tags = {
        "amenity": ["cafe", "restaurant", "fast_food", "theatre", "cinema", "arts_centre", "community_centre"],
        "tourism": ["museum", "gallery", "attraction"],
        "leisure": ["sports_centre", "fitness_centre", "stadium"],
        "shop": ["mall", "marketplace"],
        "office": ["coworking"],
    }
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
        return gdf
    except Exception:
        return gpd.GeoDataFrame()


@st.cache_data(show_spinner=False)
def load_transport_pois(lat: float, lon: float, radius_m: int) -> gpd.GeoDataFrame:
    tags = {
        "highway": ["bus_stop"],
        "public_transport": ["stop_position", "platform", "station"],
        "railway": ["station", "halt", "subway_entrance"],
        "station": ["subway", "metro"],
    }
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
        return gdf
    except Exception:
        return gpd.GeoDataFrame()


def _score_display(value) -> str:
    if value in {"0", "1", "2"}:
        return f"{value}/2"
    return "-/2"

def _area_m2(geom_3857):
    try:
        return float(gpd.GeoSeries([geom_3857], crs=3857).to_crs(epsg=32635).area.iloc[0])
    except Exception:
        try:
            return float(geom_3857.area)
        except Exception:
            return 0.0


def _total_score_color(total: int, max_total: int = 8) -> str:
    if total is None:
        return "#999999"
    t = max(0, min(total / max_total, 1.0))
    r = int(220 * (1 - t) + 46 * t)
    g = int(70 * (1 - t) + 204 * t)
    b = int(70 * (1 - t) + 113 * t)
    return f"rgb({r},{g},{b})"


def _reset_scores():
    st.session_state["identity_score"] = None
    st.session_state["activity_score"] = None
    st.session_state["activity_count"] = None
    st.session_state["activity_counts"] = None
    st.session_state["activity_pois"] = None
    st.session_state["transport_score"] = None
    st.session_state["transport_count"] = None
    st.session_state["transport_counts"] = None
    st.session_state["transport_pois"] = None
    st.session_state["centricity_score"] = None
    st.session_state["activity_context_id"] = None
    st.session_state["transport_context_id"] = None


if st.session_state.get("last_calc_district") and st.session_state["last_calc_district"] != selected_district:
    st.session_state["selected_feature"] = None
    st.session_state["selected_osm_id"] = None
    _reset_scores()


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _select_nearest_building(click_lat, click_lon, results, max_dist_m=60):
    best = None
    best_d = None
    for item in results or []:
        candidates = item.get("all_buildings", []) or item.get("buildings", [])
        for b in candidates:
            d = _haversine_m(click_lat, click_lon, b["lat"], b["lon"])
            if best_d is None or d < best_d:
                best_d = d
                best = b
    if best is not None and best_d is not None and best_d <= max_dist_m:
        return best
    return None


def _match_tagged_for_selected(selected, tagged, max_dist_m=25):
    if not selected or not tagged:
        return None
    sel_osm = selected.get("osm_id")
    sel_lat = selected.get("lat")
    sel_lon = selected.get("lon")
    if sel_osm:
        for b in tagged:
            if b.get("osm_id") == sel_osm:
                return b.get("uid")
    if sel_lat is None or sel_lon is None:
        return None
    best_id = None
    best_d = None
    for b in tagged:
        d = _haversine_m(sel_lat, sel_lon, b.get("lat"), b.get("lon"))
        if best_d is None or d < best_d:
            best_d = d
            best_id = b.get("uid")
    if best_d is not None and best_d <= max_dist_m:
        return best_id
    return None


def _buildings_by_tags_in_radius(dist_name, centers, radius_m, forbidden_tags, forbidden_amenity):
    buildings = load_buildings(dist_name)
    if buildings is None or buildings.empty:
        return []
    b3857 = buildings.to_crs(epsg=3857)
    if not centers:
        return []

    selected = []
    for _, row in b3857.iterrows():
        geom = row.geometry
        if geom is None or not geom.is_valid:
            continue
        b = str(row.get("building", "")).lower()
        use = str(row.get("building:use", "")).lower()
        landuse = str(row.get("landuse", "")).lower()
        amenity = str(row.get("amenity", "")).lower()
        if amenity in forbidden_amenity:
            continue
        if b in forbidden_tags or use in forbidden_tags or landuse in forbidden_tags:
            continue
        area = _area_m2(geom)
        g4326 = gpd.GeoSeries([geom], crs=3857).to_crs(epsg=4326).iloc[0]
        c = g4326.centroid
        dmin = None
        for lat_c, lon_c in centers:
            d = _haversine_m(c.y, c.x, lat_c, lon_c)
            if dmin is None or d < dmin:
                dmin = d
        if dmin is None or dmin > radius_m:
            continue
        nearest_idx = None
        nearest_d = None
        for i_c, (lat_c, lon_c) in enumerate(centers):
            d = _haversine_m(c.y, c.x, lat_c, lon_c)
            if nearest_d is None or d < nearest_d:
                nearest_d = d
                nearest_idx = i_c

        ring = 2 if (nearest_d is not None and nearest_d <= radius_m) else None
        if nearest_d is not None:
            if nearest_d <= radius_m / 2:
                ring = 1
            elif nearest_d <= radius_m:
                ring = 2

        selected.append(
            {
                "lat": float(g4326.centroid.y),
                "lon": float(g4326.centroid.x),
                "area": area,
                "tag": b or use or landuse,
                "geom_wkt_4326": g4326.wkt,
                "name": row.get("name") or "",
                "address": _format_address(row),
                "description": row.get("description") or row.get("note") or "",
                "osm_id": row.get("osmid") or row.get("osm_id") or "",
                "image": row.get("image") or "",
                "center_idx": nearest_idx,
                "ring": ring,
                "dist_m": nearest_d,
                "uid": f"{row.get('osmid') or row.get('osm_id') or ''}:{float(g4326.centroid.y):.6f},{float(g4326.centroid.x):.6f}",
            }
        )
    return selected


def _scale_score_from_area(area_m2):
    if not isinstance(area_m2, (int, float)):
        return None
    if area_m2 < 1500:
        return "0"
    if area_m2 <= 5000:
        return "1"
    return "2"

if st.session_state.get("run_calc"):
    st.session_state["run_calc"] = False
    with st.spinner("Строим центры и зоны доступности..."):
        try:
            st.session_state["results"] = []
            st.session_state["selected_feature"] = None
            st.session_state["selected_osm_id"] = None
            _reset_scores()
            st.session_state["calc_id"] += 1
            st.session_state["last_calc_district"] = selected_district
            # Радиус центричности фиксируется для текущего расчета.
            st.session_state["centricity_base_r"] = base_r
            colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

            results = []
            for i, dist_name in enumerate([selected_district]):
                area_gdf, isochrones, centers, coverage, building_rows, candidate_zone, all_building_rows, labels = compute_centers_and_isochrones(
                    dist_name,
                    base_centers,
                    0,
                    max_dist,
                    0,
                    use_fast=fast_mode,
                )

                color = colors[i % len(colors)]

                area_json = area_gdf.to_crs(epsg=4326).to_json()
                iso_jsons = []
                for iso in isochrones:
                    try:
                        geo = gpd.GeoDataFrame(geometry=[iso], crs=3857).to_crs(epsg=4326)
                        iso_jsons.append(geo.to_json())
                    except Exception:
                        continue
                center_points = []
                for c in centers:
                    c_gps = gpd.GeoSeries([Point(c[0], c[1])], crs=3857).to_crs(epsg=4326).iloc[0]
                    center_points.append([c_gps.y, c_gps.x])
                building_points = []
                if building_rows:
                    b_geoms = [r["geometry"] for r in building_rows]
                    b_gps = gpd.GeoSeries(b_geoms, crs=3857).to_crs(epsg=4326)
                    for i_b, g in enumerate(b_gps):
                        r = building_rows[i_b]
                        cluster_id = int(labels[i_b]) if labels is not None and len(labels) > i_b else None
                        building_points.append(
                            {
                                "lat": float(g.centroid.y),
                                "lon": float(g.centroid.x),
                                "name": r.get("name") or "Жилой дом",
                                "address": r.get("address") or "",
                                "description": r.get("description") or "",
                                "osm_id": r.get("osm_id") or "",
                                "uid": f"{r.get('osm_id') or ''}:{float(g.centroid.y):.6f},{float(g.centroid.x):.6f}",
                                "area": _area_m2(building_rows[i_b]["geometry"]) if building_rows else 0.0,
                                "image": r.get("image") or "",
                                "category": "residential",
                                "district": dist_name,
                                "cluster_id": cluster_id,
                            }
                        )

                all_building_points = []
                if all_building_rows:
                    all_geoms = [r["geometry"] for r in all_building_rows]
                    all_gps = gpd.GeoSeries(all_geoms, crs=3857).to_crs(epsg=4326)
                    for i_b, g in enumerate(all_gps):
                        r = all_building_rows[i_b]
                        all_building_points.append(
                            {
                                "lat": float(g.centroid.y),
                                "lon": float(g.centroid.x),
                                "name": r.get("name") or "Здание",
                                "address": r.get("address") or "",
                                "description": r.get("description") or "",
                                "osm_id": r.get("osm_id") or "",
                                "uid": f"{r.get('osm_id') or ''}:{float(g.centroid.y):.6f},{float(g.centroid.x):.6f}",
                                "area": _area_m2(all_building_rows[i_b]["geometry"]) if all_building_rows else 0.0,
                                "image": r.get("image") or "",
                                "category": r.get("category") or "",
                                "district": dist_name,
                            }
                        )

                candidate_zone_json = None
                candidate_zone_wkt = None
                candidate_zone_wkt_3857 = None
                if candidate_zone is not None:
                    try:
                        cz_gdf = gpd.GeoDataFrame(geometry=[candidate_zone], crs=3857).to_crs(epsg=4326)
                        candidate_zone_json = cz_gdf.to_json()
                        candidate_zone_wkt = cz_gdf.geometry.iloc[0].wkt
                        candidate_zone_wkt_3857 = candidate_zone.wkt
                    except Exception:
                        candidate_zone_json = None

                st.session_state["zone_base_r"] = st.session_state.get("centricity_base_r")

                results.append(
                    {
                        "name": dist_name,
                        "color": color,
                        "area_json": area_json,
                        "iso_jsons": iso_jsons,
                        "centers": center_points,
                        "coverage": coverage,
                        "buildings": building_points,
                        "candidate_zone": candidate_zone_json,
                        "candidate_zone_wkt": candidate_zone_wkt,
                        "candidate_zone_wkt_3857": candidate_zone_wkt_3857,
                        "all_buildings": all_building_points,
                    }
                )

            st.session_state["results"] = results
            st.session_state["last_coverage_pct"] = coverage if results else None
            st.session_state["last_error"] = None
        except Exception as e:
            st.session_state["last_error"] = str(e)

m = _base_map()
if st.session_state["results"]:
    cluster_palette = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    for item in st.session_state["results"]:
        color = item["color"]

        folium.GeoJson(
            item["area_json"],
            style_function=lambda x, col=color: {
                "color": col,
                "fill": False,
                "weight": 2,
                "dashArray": "5, 5",
            },
        ).add_to(m)

        if show_isochrones:
            for iso_json in item["iso_jsons"]:
                folium.GeoJson(
                    iso_json,
                    style_function=lambda x: {
                        "fillColor": "#A8E6CF",
                        "color": "#7BC8A4",
                        "fillOpacity": 0.12,
                        "weight": 1,
                    },
                ).add_to(m)

        for lat, lon in item["centers"]:
            folium.Marker(
                [lat, lon],
                icon=folium.Icon(color="white", icon_color=color, icon="star", prefix="fa"),
            ).add_to(m)
        for idx, (lat, lon) in enumerate(item["centers"], start=1):
            folium.map.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f"<div style='font-size:12px;color:{color};font-weight:700;'>{idx}</div>"
                ),
            ).add_to(m)

        if show_candidate_zones:
            base_r = st.session_state.get("zone_base_r") or max(80, max_dist * 0.15)
            for lat_c, lon_c in item.get("centers", []):
                folium.Circle(
                    location=[lat_c, lon_c],
                    radius=base_r,
                    color="#2ECC71",
                    weight=2,
                    fill=True,
                    fill_opacity=0.05,
                ).add_to(m)
                folium.Circle(
                    location=[lat_c, lon_c],
                    radius=base_r * 2,
                    color="#A3D9A5",
                    weight=1,
                    fill=False,
                    dash_array="4, 6",
                ).add_to(m)

        for b in item.get("buildings", []):
            if show_clusters and b.get("cluster_id") is not None:
                cidx = int(b["cluster_id"])
                ccol = cluster_palette[cidx % len(cluster_palette)]
                radius = 2
                opacity = 0.8
            else:
                ccol = color
                radius = 1
                opacity = 0.6
            folium.CircleMarker(
                [b["lat"], b["lon"]],
                radius=radius,
                color=ccol,
                fill=True,
                fill_opacity=opacity,
                weight=0,
            ).add_to(m)

    tagged = st.session_state.get("tagged_buildings")
    if tagged and st.session_state.get("tagged_context_id") == st.session_state.get("calc_id"):
        for b in tagged:
            threshold = st.session_state.get("area_threshold", 500)
            if b.get("uid") in st.session_state.get("hidden_tagged_ids", set()):
                continue
            if b.get("area", 0) < threshold:
                continue
            try:
                geom = wkt.loads(b["geom_wkt_4326"])
                folium.GeoJson(
                    gpd.GeoDataFrame(geometry=[geom], crs=4326).to_json(),
                    style_function=lambda x: {
                        "color": "#D35400",
                        "weight": 1,
                        "fillColor": "#D35400",
                        "fillOpacity": 0.2,
                    },
                    name="Здание по тегу",
                ).add_to(m)
            except Exception:
                pass

# Подсветка выбранного объекта и его окружения
if st.session_state["selected_feature"]:
    sf = st.session_state["selected_feature"]
    folium.Marker(
        [sf["lat"], sf["lon"]],
        icon=folium.Icon(color="black", icon="info-sign"),
        tooltip=sf.get("name") or "Выбранный объект",
        popup=folium.Popup(
            html=(
                f"<b>{sf.get('name','')}</b><br>"
                f"Категория: {sf.get('category','')}<br>"
                f"Адрес: {sf.get('address','')}<br>"
                f"Описание: {sf.get('description','')}<br>"
                f"OSM ID: {sf.get('osm_id','')}"
            ),
            max_width=400,
        ),
    ).add_to(m)
    folium.CircleMarker(
        [sf["lat"], sf["lon"]],
        radius=6,
        color="#000000",
        fill=False,
        weight=2,
    ).add_to(m)
    pois = st.session_state.get("activity_pois")
    if st.session_state.get("activity_context_id") != st.session_state.get("selected_osm_id"):
        pois = None
    if pois is not None and not pois.empty:
        folium.Circle(
            location=[sf["lat"], sf["lon"]],
            radius=activity_radius,
            color="#6C5CE7",
            weight=1,
            fill=True,
            fill_opacity=0.03,
        ).add_to(m)
        try:
            pois_4326 = pois.to_crs(epsg=4326)
        except Exception:
            pois_4326 = pois
        for geom in pois_4326.geometry:
            try:
                g = geom.centroid
                folium.CircleMarker(
                    [g.y, g.x],
                    radius=2,
                    color="#6C5CE7",
                    fill=True,
                    fill_opacity=0.8,
                    weight=0,
                ).add_to(m)
            except Exception:
                continue
    tpois = st.session_state.get("transport_pois")
    if st.session_state.get("transport_context_id") != st.session_state.get("selected_osm_id"):
        tpois = None
    if tpois is not None and not tpois.empty:
        folium.Circle(
            location=[sf["lat"], sf["lon"]],
            radius=transport_radius,
            color="#2C7BE5",
            weight=1,
            fill=True,
            fill_opacity=0.03,
        ).add_to(m)
    tpois = st.session_state.get("transport_pois")
    if st.session_state.get("transport_context_id") != st.session_state.get("selected_osm_id"):
        tpois = None
    if tpois is not None and not tpois.empty:
        try:
            tpois_4326 = tpois.to_crs(epsg=4326)
        except Exception:
            tpois_4326 = tpois
        for geom in tpois_4326.geometry:
            try:
                g = geom.centroid
                folium.CircleMarker(
                    [g.y, g.x],
                    radius=2,
                    color="#2C7BE5",
                    fill=True,
                    fill_opacity=0.8,
                    weight=0,
                ).add_to(m)
            except Exception:
                continue

map_col, side_col = st.columns([4, 1], gap="large")

with map_col:
    map_key = f"map_{st.session_state.get('calc_id', 0)}"
    map_data = st_folium(m, width=1400, height=700, key=map_key, return_on_hover=False)
    if map_data and map_data.get("last_clicked") and st.session_state.get("results"):
        lc = map_data["last_clicked"]
        picked = _select_nearest_building(lc["lat"], lc["lng"], st.session_state["results"])
        if picked:
            picked_id = picked.get("uid") or picked.get("osm_id")
            if st.session_state.get("selected_osm_id") != picked_id:
                st.session_state["selected_osm_id"] = picked_id
                _reset_scores()
            st.session_state["selected_feature"] = picked
            st.session_state["identity_score"] = _scale_score_from_area(picked.get("area"))
    if st.session_state.get("selected_feature") and st.session_state.get("identity_score") is None:
        sf = st.session_state["selected_feature"]
        st.session_state["identity_score"] = _scale_score_from_area(sf.get("area"))
    if st.session_state.get("selected_feature"):
        sf = st.session_state["selected_feature"]
        zone_wkt_3857 = None
        for item in st.session_state.get("results") or []:
            if item.get("name") == sf.get("district"):
                zone_wkt_3857 = item.get("candidate_zone_wkt_3857")
                break
        try:
            base_r = st.session_state.get("zone_base_r") or max(80, max_dist * 0.15)
            centers = []
            for item in st.session_state.get("results") or []:
                if item.get("name") == sf.get("district"):
                    centers = item.get("centers", [])
                    break
            if centers:
                dmin = None
                for lat_c, lon_c in centers:
                    d = _haversine_m(sf["lat"], sf["lon"], lat_c, lon_c)
                    if dmin is None or d < dmin:
                        dmin = d
                if dmin is not None:
                    if dmin <= base_r:
                        st.session_state["centricity_score"] = "2"
                    elif dmin <= base_r * 2:
                        st.session_state["centricity_score"] = "1"
                    else:
                        st.session_state["centricity_score"] = "0"
        except Exception:
            st.session_state["centricity_score"] = None
    if st.session_state["last_error"]:
        st.error(f"Последняя ошибка: {st.session_state['last_error']}")

with side_col:
    st.caption("Выбранный объект")
    if st.session_state["selected_feature"]:
        sf = st.session_state["selected_feature"]
        st.markdown(
            f"""
<div style="line-height:1.25;">
  <div style="font-size:22px; font-weight:700;">{sf.get('address') or 'Адрес не указан'}</div>
  <div style="font-size:16px; font-weight:600; margin-top:2px;">{sf.get('name') or 'Без названия'}</div>
  <div style="font-size:13px; margin-top:2px;">Категория: {sf.get('category') or '—'}</div>
  <div style="font-size:13px; margin-top:2px;">Описание: {sf.get('description') or '—'}</div>
  <div style="font-size:13px; margin-top:2px;">Площадь застройки: {f"{sf.get('area'):.0f} м²" if isinstance(sf.get('area'), (int, float)) else "—"}</div>
  <div style="font-size:13px; margin-top:4px;">OSM ID: {sf.get('osm_id') or '—'}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if sf.get("image"):
            st.image(sf.get("image"), caption="Фото объекта (OSM)", use_container_width=True)
        tagged = st.session_state.get("tagged_buildings") or []
        tagged_id = _match_tagged_for_selected(sf, tagged)
        if tagged_id:
            hidden = tagged_id in st.session_state.get("hidden_tagged_ids", set())
            label = "Показать объект" if hidden else "Скрыть объект"
            if st.button(label, key=f"toggle_tagged_{tagged_id}"):
                hidden_ids = set(st.session_state.get("hidden_tagged_ids", set()))
                if hidden:
                    hidden_ids.discard(tagged_id)
                else:
                    hidden_ids.add(tagged_id)
                st.session_state["hidden_tagged_ids"] = hidden_ids
    else:
        st.info("Кликните по зданию на карте, чтобы выбрать объект.")

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    st.subheader("Оценки")
    id_s = _score_display(st.session_state.get("identity_score"))
    act_raw = st.session_state.get("activity_score") if st.session_state.get("activity_context_id") == st.session_state.get("selected_osm_id") else None
    tr_raw = st.session_state.get("transport_score") if st.session_state.get("transport_context_id") == st.session_state.get("selected_osm_id") else None
    act_s = _score_display(act_raw)
    cen_s = _score_display(st.session_state.get("centricity_score"))
    tr_s = _score_display(tr_raw)
    st.markdown(
        f"""
<div style="display:flex; flex-direction:column; gap:8px;">
  <div style="display:flex; justify-content:space-between; align-items:center; font-size:18px;">
    <span><b>Масштабность</b></span><span style="font-size:32px;"><b>{id_s}</b></span>
  </div>
  <div style="display:flex; justify-content:space-between; align-items:center; font-size:18px;">
    <span><b>Соц.-культ. активность</b></span><span style="font-size:32px;"><b>{act_s}</b></span>
  </div>
  <div style="display:flex; justify-content:space-between; align-items:center; font-size:18px;">
    <span><b>Центричность места</b></span><span style="font-size:32px;"><b>{cen_s}</b></span>
  </div>
  <div style="display:flex; justify-content:space-between; align-items:center; font-size:18px;">
    <span><b>Транспортная доступность</b></span><span style="font-size:32px;"><b>{tr_s}</b></span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    scores = [st.session_state.get("identity_score"), act_raw,
              st.session_state.get("centricity_score"), tr_raw]
    if all(s in {"0", "1", "2"} for s in scores):
        total = sum(int(s) for s in scores)
        color = _total_score_color(total)
        st.markdown(
            f"""
<div style="margin-top:10px; padding:10px 12px; border-radius:8px; 
background:{color}; color:#fff; text-align:center; font-size:20px;">
  <div style="font-size:14px; opacity:0.9;">Итоговая оценка</div>
  <div style="font-size:56px;"><b>{total}/8</b></div>
</div>
""",
            unsafe_allow_html=True,
        )

st.divider()

# Панель оценки выбранного объекта
action_col1, action_col2, action_col3, action_col4 = st.columns([2, 1, 1, 2], gap="large")
with action_col1:
    st.subheader("Масштабность")
    st.markdown(
        "<div style='line-height:1.2; font-size:13px;'>"
        "Оценка считается автоматически по площади застройки: "
        "менее 1500 м² — 0, от 1500 до 5000 м² — 1, более 5000 м² — 2."
        "</div>",
        unsafe_allow_html=True,
    )


with action_col2:
    st.subheader("Соц.-культ. активность")
    activity_minutes = st.slider("Радиус активности (мин пешком)", 5, 30, 10)
    st.session_state["activity_minutes"] = activity_minutes
    if st.button("Рассчитать активность"):
        if not st.session_state["selected_feature"]:
            st.error("Сначала выберите объект на карте.")
        else:
            sf = st.session_state["selected_feature"]
            pois = load_activity_pois(sf["lat"], sf["lon"], activity_radius)
            count = 0 if pois is None or pois.empty else len(pois)
            st.session_state["activity_count"] = count
            st.session_state["activity_pois"] = pois
            st.session_state["activity_context_id"] = st.session_state.get("selected_osm_id")
            counts = {"cafe": 0, "theatre": 0, "museum": 0, "market": 0, "coworking": 0, "sports": 0, "other": 0}
            if pois is not None and not pois.empty:
                for _, row in pois.iterrows():
                    amenity = str(row.get("amenity", "")).lower()
                    tourism = str(row.get("tourism", "")).lower()
                    shop = str(row.get("shop", "")).lower()
                    office = str(row.get("office", "")).lower()
                    leisure = str(row.get("leisure", "")).lower()
                    if amenity in {"cafe", "restaurant", "fast_food"}:
                        counts["cafe"] += 1
                    elif amenity in {"theatre", "cinema", "arts_centre", "community_centre"}:
                        counts["theatre"] += 1
                    elif tourism in {"museum", "gallery", "attraction"}:
                        counts["museum"] += 1
                    elif shop in {"marketplace", "mall"}:
                        counts["market"] += 1
                    elif office in {"coworking"}:
                        counts["coworking"] += 1
                    elif leisure in {"sports_centre", "fitness_centre", "stadium"}:
                        counts["sports"] += 1
                    else:
                        counts["other"] += 1
            st.session_state["activity_counts"] = counts
            if count <= 5:
                st.session_state["activity_score"] = "0"
            elif count <= 10:
                st.session_state["activity_score"] = "1"
            else:
                st.session_state["activity_score"] = "2"
            st.rerun()
    if st.session_state.get("activity_context_id") == st.session_state.get("selected_osm_id"):
        if st.session_state.get("activity_count") is not None:
            st.markdown(f"**Объектов в радиусе:** {st.session_state['activity_count']}")
    if st.session_state.get("activity_context_id") == st.session_state.get("selected_osm_id") and st.session_state.get("activity_counts"):
        counts = st.session_state["activity_counts"]
        labels = {
            "cafe": "Кафе/еда",
            "theatre": "Театры/культура",
            "museum": "Музеи/галереи",
            "market": "Рынки/ТЦ",
            "coworking": "Коворкинги",
            "sports": "Спорт",
            "other": "Другое",
        }
        st.markdown("**Состав активности:**")
        items = []
        for key, label in labels.items():
            if key in counts:
                items.append(f"{label}: {counts[key]}")
        if items:
            st.markdown(
                "<div style='line-height:1.15; font-size:13px;'>" +
                "<br>".join(items) +
                "</div>",
                unsafe_allow_html=True,
            )

with action_col3:
    st.subheader("Транспортная доступность")
    transport_minutes = st.slider("Радиус транспорта (мин пешком)", 5, 30, 15)
    st.session_state["transport_minutes"] = transport_minutes
    if st.button("Рассчитать транспорт"):
        if not st.session_state["selected_feature"]:
            st.error("Сначала выберите объект на карте.")
        else:
            sf = st.session_state["selected_feature"]
            tpois = load_transport_pois(sf["lat"], sf["lon"], transport_radius)
            tcount = 0 if tpois is None or tpois.empty else len(tpois)
            st.session_state["transport_count"] = tcount
            st.session_state["transport_pois"] = tpois
            st.session_state["transport_context_id"] = st.session_state.get("selected_osm_id")
            counts = {"metro": 0, "bus": 0, "other": 0}
            if tpois is not None and not tpois.empty:
                for _, row in tpois.iterrows():
                    railway = str(row.get("railway", "")).lower()
                    station = str(row.get("station", "")).lower()
                    highway = str(row.get("highway", "")).lower()
                    public_transport = str(row.get("public_transport", "")).lower()
                    if railway in {"subway_entrance"} or station in {"subway", "metro"}:
                        counts["metro"] += 1
                    elif highway == "bus_stop" or public_transport in {"stop_position", "platform"}:
                        counts["bus"] += 1
                    else:
                        counts["other"] += 1
            st.session_state["transport_counts"] = counts
            if tcount == 0:
                st.session_state["transport_score"] = "0"
            elif tcount < 6:
                st.session_state["transport_score"] = "1"
            else:
                st.session_state["transport_score"] = "2"
            st.rerun()
    if st.session_state.get("transport_context_id") == st.session_state.get("selected_osm_id"):
        if st.session_state.get("transport_count") is not None:
            st.markdown(f"**Объектов транспорта:** {st.session_state['transport_count']}")
    if st.session_state.get("transport_context_id") == st.session_state.get("selected_osm_id") and st.session_state.get("transport_counts"):
        tc = st.session_state["transport_counts"]
        st.markdown("**Состав транспорта:**")
        st.markdown(
            "<div style='line-height:1.15; font-size:13px;'>" +
            f"Метро: {tc.get('metro', 0)}<br>" +
            f"Автобусные остановки: {tc.get('bus', 0)}<br>" +
            f"Другое: {tc.get('other', 0)}" +
            "</div>",
            unsafe_allow_html=True,
        )

with action_col4:
    st.subheader("Выявить центры")
    area_threshold = st.slider("Скрывать здания меньше (м²)", 200, 2000, 500, step=100)
    st.session_state["area_threshold"] = area_threshold
    if st.button("Выявить центры"):
        # Исключаются типы, которые не подходят для новых общественных центров.
        forbidden_tags = {
            # жилой фонд
            "residential",
            "apartments",
            "house",
            "detached",
            "semidetached_house",
            "terrace",
            "dormitory",
            "bungalow",
            "static_caravan",
            # социальная инфраструктура
            "school",
            "kindergarten",
            "college",
            "university",
            "hospital",
            "policlinic",
            "clinic",
            # специальное назначение и культ
            "police",
            "fire_station",
            "place_of_worship",
            "military",
            # технические/хозяйственные
            "transformer_tower",
            "substation",
            "garage",
            "garages",
            "storage_tank",
            "parking",
            "carport",
            "hangar",
        }
        forbidden_amenity = {
            "school",
            "kindergarten",
            "college",
            "university",
            "hospital",
            "policlinic",
            "clinic",
            "police",
            "fire_station",
            "place_of_worship",
        }
        # По центрам текущего расчета отбираются объекты в двух зонах центричности.
        base_r = st.session_state.get("zone_base_r") or max(80, max_dist * 0.15)
        centers = []
        for item in st.session_state.get("results") or []:
            if item.get("name") == selected_district:
                centers = item.get("centers", [])
                break
        if not centers:
            st.warning("Сначала сделайте расчет центров для выбранного района.")
            tagged = []
        else:
            tagged = _buildings_by_tags_in_radius(
                selected_district,
                centers,
                base_r * 2,
                forbidden_tags,
                forbidden_amenity,
            )
        st.session_state["tagged_buildings"] = tagged
        st.session_state["tagged_context_id"] = st.session_state.get("calc_id")

    st.subheader("Список адресов")
    if st.button("Показать адреса выделенных зданий"):
        st.session_state["show_addresses"] = True
        rows = []
        tagged = st.session_state.get("tagged_buildings") or []
        for b in tagged:
            rows.append(
                {
                    "center_group": (b.get("center_idx") + 1) if b.get("center_idx") is not None else None,
                    "ring": b.get("ring"),
                    "dist_m": b.get("dist_m"),
                    "uid": b.get("uid"),
                    "name": b.get("name") or "",
                    "address": b.get("address") or "",
                    "description": b.get("description") or "",
                    "area": b.get("area"),
                    "osm_id": b.get("osm_id") or "",
                }
            )
        st.session_state["addresses_df"] = pd.DataFrame(rows) if rows else pd.DataFrame()

if st.session_state["show_addresses"]:
    df = st.session_state["addresses_df"]
    if df is None or df.empty:
        st.info("В серой зоне не найдено объектов с адресами или категориями.")
    else:
        threshold = st.session_state.get("area_threshold", 500)
        df = df[df["area"].fillna(0) >= threshold]
        df_sorted = df.sort_values(["center_group", "ring", "area"], ascending=[True, True, False])
        selected = st.session_state.get("selected_feature") or {}
        selected_osm = selected.get("osm_id")
        selected_addr = selected.get("address")
        selected_name = selected.get("name")
        for grp, grp_df in df_sorted.groupby("center_group", dropna=False):
            title = f"Группа {int(grp)}" if pd.notna(grp) else "Группа: не определена"
            st.markdown(f"**{title}**")
            view_df = grp_df[["ring", "name", "address", "description", "area", "osm_id", "uid"]].copy()
            view_df["ring"] = view_df["ring"].apply(lambda x: "Малый круг" if x == 1 else ("Большой круг" if x == 2 else "—"))
            view_df["area"] = view_df["area"].apply(lambda x: f"{x:.0f} м²" if isinstance(x, (int, float)) else "—")
            hidden_ids = st.session_state.get("hidden_tagged_ids", set())
            if selected_osm or selected_addr or selected_name or hidden_ids:
                def _hl(row):
                    match = False
                    if row.get("uid") in hidden_ids:
                        return ["background-color: #7B1E1E; color: white;"] * len(view_df.columns)
                    if selected_osm and row.get("osm_id") == selected_osm:
                        match = True
                    elif selected_addr and row.get("address") == selected_addr and row.get("name") == selected_name:
                        match = True
                    return ["background-color: #2B6CB0; color: white;"] * len(view_df.columns) if match else [""] * len(view_df.columns)
                styler = view_df.style.apply(_hl, axis=1).hide(axis="columns", subset=["uid"])
                st.dataframe(styler, use_container_width=True, height=220)
            else:
                st.dataframe(view_df.drop(columns=["uid"]), use_container_width=True, height=220)
        st.download_button(
            "Скачать CSV (Excel)",
            df_sorted.to_csv(index=False, sep=";", encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="center_zone_addresses.csv",
            mime="text/csv",
        )
