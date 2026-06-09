"""
CosmoSim Academic Educational Platform
========================================
Interactive ΛCDM Cosmological Simulator for resolving student misconceptions
about Gravitationally Bound Systems vs. Hubble Expansion.

Architecture: Streamlit (Python) + Three.js + Plotly.js (HTML components)
Physics: Planck 2020 ΛCDM · Trapezoidal Friedmann solver · 3000-step integration
Data: 25 named real star systems + 200 NASA Exoplanet Archive host stars
"""
import streamlit as st
import streamlit.components.v1 as components
import json
import os
import warnings
warnings.filterwarnings("ignore")

from game_engine import (
    Universe, STAR_CLASSES, WEATHER_MAP, RESOURCES,
    LCDM_TABLE, LCDM_CHART, EPOCH_BENCHMARKS,
    HUBBLE_PTS, HUBBLE_SLOPE, HUBBLE_INTERCEPT, HUBBLE_R2,
    REDSHIFT_DIST, interp_lcdm,
    H0_KMS_MPC, OMEGA_M, OMEGA_L,
    build_background_stars_json,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CosmoSim — ΛCDM Academic Platform",
    layout="wide",
    page_icon="⬡",
    initial_sidebar_state="expanded",
)

# ============================================================
# ACADEMIC DARK LABORATORY THEME
# ============================================================
# Inject global CSS via zero-height component (never leaks as text)
_CSS = ("<link href='https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap' rel='stylesheet'>"
"<style>"
".stApp{background:linear-gradient(160deg,#080e1e 0%,#0c1428 60%,#080e1e 100%)!important;font-family:'Inter',sans-serif!important}"
"h1{font-family:'Space Mono',monospace!important;color:#4fc3f7!important;font-size:22px!important;letter-spacing:1.5px!important;text-shadow:0 0 30px rgba(79,195,247,.25)!important}"
"h2,h3{font-family:'Space Mono',monospace!important;color:#81d4fa!important;letter-spacing:.8px!important}"
"h4{color:#80cbc4!important;font-weight:600!important}"
"p,li,span{color:#b0bec5!important}"
"div[data-testid='stSidebar'],div[data-testid='stSidebar'] div,[data-testid='stSidebarContent']{background:linear-gradient(180deg,#07111f 0%,#0a172e 100%)!important;background-color:#07111f!important}"
"div[data-testid='stSidebar']{border-right:1px solid rgba(79,195,247,.15)!important}"
"div[data-testid='stSidebar'] h1,div[data-testid='stSidebar'] h2,div[data-testid='stSidebar'] h3{color:#4fc3f7!important;font-size:13px!important}"
"div[data-testid='stSidebar'] p,div[data-testid='stSidebar'] span,div[data-testid='stSidebar'] li,div[data-testid='stSidebar'] label,div[data-testid='stSidebar'] small,div[data-testid='stSidebar'] div{color:#90a4ae!important}"
".stMetric{background:rgba(79,195,247,.04)!important;border:1px solid rgba(79,195,247,.15)!important;border-radius:8px!important;padding:8px!important}"
".stMetric label{color:#546e7a!important;font-size:10px!important;text-transform:uppercase!important;letter-spacing:1.5px!important;font-family:'Space Mono',monospace!important}"
".stMetric [data-testid='stMetricValue']{color:#4fc3f7!important;font-family:'Space Mono',monospace!important;font-size:18px!important;font-weight:700!important}"
"div[data-baseweb='select'] div{background:rgba(8,14,32,.95)!important;border-color:rgba(79,195,247,.25)!important;border-radius:6px!important}"
"div[data-baseweb='select'] span,div[data-baseweb='select'] div{color:#b0bec5!important}"
"ul[role='listbox']{background:#0c1428!important}"
"ul[role='listbox'] li{color:#b0bec5!important}"
"ul[role='listbox'] li:hover{background:rgba(79,195,247,.1)!important}"
".stButton button{background:rgba(79,195,247,.1)!important;color:#4fc3f7!important;border:1px solid rgba(79,195,247,.4)!important;border-radius:6px!important;font-family:'Space Mono',monospace!important;font-size:10px!important;font-weight:700!important;letter-spacing:1px!important;text-transform:uppercase!important}"
".stButton button:hover{background:rgba(79,195,247,.22)!important;border-color:rgba(79,195,247,.7)!important}"
"div[data-testid='stAlert']{border-radius:8px!important;border:1px solid rgba(79,195,247,.2)!important;background:rgba(79,195,247,.04)!important}"
"hr{border-color:rgba(79,195,247,.1)!important}"
".sci-card{background:linear-gradient(135deg,rgba(8,14,32,.85) 0%,rgba(79,195,247,.03) 100%);border:1px solid rgba(79,195,247,.14);border-radius:10px;padding:16px 18px;margin:8px 0;transition:border-color .25s}"
".sci-card h4{font-family:'Space Mono',monospace!important;font-size:12px!important;color:#4fc3f7!important;margin-bottom:6px!important}"
".sci-card small{color:#546e7a;line-height:1.7;font-size:12px}"
".sci-card b{color:#81d4fa}"
".edu-box{background:rgba(79,195,247,.04);border:1px solid rgba(79,195,247,.18);border-left:3px solid #4fc3f7;border-radius:8px;padding:14px 18px;margin:10px 0}"
".edu-box h4{font-family:'Space Mono',monospace!important;font-size:11px!important;color:#4fc3f7!important;margin-bottom:6px!important;letter-spacing:1px!important}"
".edu-box p{font-size:13px!important;line-height:1.7;color:#90a4ae!important}"
"::-webkit-scrollbar{width:5px}"
"::-webkit-scrollbar-track{background:rgba(0,0,0,.2)}"
"::-webkit-scrollbar-thumb{background:rgba(79,195,247,.3);border-radius:3px}"
"</style>")
components.html(_CSS, height=0)  # height=0 → no visible space; CSS never rendered as text


# ============================================================
# SESSION STATE INITIALIZATION & i18n LOADING
# ============================================================
I18N_PATH = os.path.join(os.path.dirname(__file__), 'i18n.json')
try:
    with open(I18N_PATH, 'r', encoding='utf-8') as f:
        I18N = json.load(f)
except Exception:
    I18N = {}

def t(key, **kwargs):
    lang = st.session_state.get('lang', 'en')
    val = I18N.get(lang, {}).get(key, key)
    if kwargs:
        try:
            return val.format(**kwargs)
        except Exception:
            pass
    return val

def init_state():
    defaults = {
        'view_mode':       'galaxy',
        'current_system':  None,
        'current_planet':  None,
        'visited_systems': [],
        'visited_planets': [],
        'lang':            'en',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Sync query parameter language to session state
params = st.query_params
if 'lang' in params:
    st.session_state['lang'] = params['lang']

# Render dynamic language selector dropdown floating at the top-right
lang_flags = {'th': '🇹🇭', 'en': '🇬🇧', 'ja': '🇯🇵'}
lang_names = {'th': 'ไทย', 'en': 'English', 'ja': '日本語'}
cur_lang = st.session_state.get('lang', 'en')
cur_flag = lang_flags.get(cur_lang, '🇬🇧')
cur_name = lang_names.get(cur_lang, 'English')

lang_dropdown_html = f"""
<div class="lang-selector-container" id="langSelector">
    <button class="lang-selector-btn">
        {cur_flag} {cur_name} <span style="font-size:8px; margin-left:4px;">▼</span>
    </button>
    <div class="lang-selector-dropdown">
        <a class="lang-option" href="?lang=th" target="_top">🇹🇭 ไทย (Thai)</a>
        <a class="lang-option" href="?lang=en" target="_top">🇬🇧 English (US/UK)</a>
        <a class="lang-option" href="?lang=ja" target="_top">🇯🇵 日本語 (Japanese)</a>
    </div>
</div>
<style>
.lang-selector-container {{
    position: fixed;
    top: 12px;
    right: 50px;
    z-index: 999999;
    font-family: 'Space Mono', monospace;
}}
.lang-selector-btn {{
    background: rgba(8, 14, 32, 0.85);
    border: 1px solid rgba(79, 195, 247, 0.3);
    color: #4fc3f7;
    padding: 8px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 700;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s, background 0.2s;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}}
.lang-selector-container:hover .lang-selector-btn {{
    border-color: rgba(79, 195, 247, 0.7);
    background: rgba(79, 195, 247, 0.15);
}}
.lang-selector-dropdown {{
    display: none;
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 6px;
    background: rgba(7, 12, 26, 0.95);
    border: 1px solid rgba(79, 195, 247, 0.25);
    border-radius: 6px;
    width: 180px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6);
    overflow: hidden;
    backdrop-filter: blur(14px);
}}
.lang-selector-container:hover .lang-selector-dropdown {{
    display: block;
}}
.lang-option {{
    display: block;
    padding: 10px 14px;
    color: #b0bec5;
    font-size: 11px;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.18s, color 0.18s;
}}
.lang-option:hover {{
    background: rgba(79, 195, 247, 0.12);
    color: #4fc3f7;
}}
</style>
"""
st.markdown(lang_dropdown_html, unsafe_allow_html=True)

# ============================================================
# LOAD UNIVERSE (CACHED)
# ============================================================
@st.cache_resource
def load_universe():
    return Universe(seed=42)

@st.cache_resource
def load_bg_stars_json():
    """Build background stars JSON once and cache."""
    return json.dumps(build_background_stars_json())

universe   = load_universe()
SCENE_DIR  = os.path.join(os.path.dirname(__file__), 'scenes')

def load_scene(filename: str) -> str:
    with open(os.path.join(SCENE_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()

# ============================================================
# PRE-SERIALISE SHARED DATA (JSON, computed once)
# ============================================================
SYSTEMS_JSON           = json.dumps(universe.to_json())
LCDM_JSON              = json.dumps(LCDM_TABLE)      # Full table for galaxy slider
LCDM_CHART_JSON        = json.dumps(LCDM_CHART)      # Thinned for charts
EPOCH_JSON             = json.dumps(EPOCH_BENCHMARKS)
HUBBLE_PTS_JSON        = json.dumps(HUBBLE_PTS)
REDSHIFT_JSON          = json.dumps(REDSHIFT_DIST)
BACKGROUND_STARS_JSON  = load_bg_stars_json()        # 200 NASA exoplanet host stars

# Current epoch at present day (for sidebar display)
_present = interp_lcdm(14.0)

# ============================================================
# QUERY-PARAM NAVIGATION BRIDGE (iframe → Streamlit)
# ============================================================
params = st.query_params
if 'action' in params:
    act = params.get('action', '')
    current_lang = st.session_state.get('lang', 'en')
    st.query_params.clear()
    st.query_params['lang'] = current_lang

    if act == 'back_galaxy':
        st.session_state.view_mode      = 'galaxy'
        st.session_state.current_system = None
        st.session_state.current_planet = None
        st.rerun()

    elif act == 'warp':
        si = params.get('sys', None)
        if si is not None:
            idx = int(si)
            st.session_state.view_mode      = 'system'
            st.session_state.current_system = idx
            st.session_state.current_planet = None
            if idx not in st.session_state.visited_systems:
                st.session_state.visited_systems.append(idx)
            st.rerun()

    elif act == 'land':
        pl = params.get('planet', None)
        if pl is not None:
            pidx = int(pl)
            st.session_state.view_mode      = 'planet'
            st.session_state.current_planet = pidx
            key = f"{st.session_state.current_system}_{pidx}"
            if key not in st.session_state.visited_planets:
                st.session_state.visited_planets.append(key)
            st.rerun()

    elif act == 'launch':
        st.session_state.view_mode      = 'system'
        st.session_state.current_planet = None
        st.rerun()

    elif act == 'dashboard':
        st.session_state.view_mode = 'dashboard'
        st.rerun()

    elif act == 'theory':
        st.session_state.view_mode = 'theory'
        st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
mode = st.session_state.view_mode

with st.sidebar:
    st.markdown(f"## {t('app_title')}")
    st.caption(t('app_subtitle'))
    st.divider()

    # ── ΛCDM Metrics ──
    st.markdown(f"### {t('model_title')}")
    c1, c2 = st.columns(2)
    c1.metric("H₀  [km/s/Mpc]", f"{H0_KMS_MPC:.1f}")
    c2.metric("Ωm", f"{OMEGA_M:.3f}")
    c1.metric("ΩΛ", f"{OMEGA_L:.3f}")
    c2.metric("a(t₀)", "1.000")
    st.divider()

    # ── Navigation ──
    st.markdown(f"### {t('nav_title')}")

    if mode == 'galaxy':
        st.info(t('galaxy_map_title'))
        names = [
            f"{'✓' if i in st.session_state.visited_systems else '·'} {s.name}  [{s.star_class}]"
            for i, s in enumerate(universe.systems)
        ]
        sel = st.selectbox(t('select_system'), range(len(names)), format_func=lambda i: names[i])
        preview = universe.systems[sel]
        preview_dict = preview.to_dict(lang=st.session_state['lang'])
        st.caption(f"📖 {preview_dict['desc']}")
        if st.button(t('warp_btn'), use_container_width=True, type="primary"):
            st.session_state.view_mode      = 'system'
            st.session_state.current_system = sel
            st.session_state.current_planet = None
            if sel not in st.session_state.visited_systems:
                st.session_state.visited_systems.append(sel)
            st.rerun()
        st.markdown("---")
        if st.button(t('theory_btn'), use_container_width=True):
            st.session_state.view_mode = 'theory'
            st.rerun()
        if st.button(t('dashboard_btn'), use_container_width=True):
            st.session_state.view_mode = 'dashboard'
            st.rerun()

    elif mode == 'system':
        sys = universe.systems[st.session_state.current_system]
        st.success(f"☀ **{sys.name}**")
        st.caption(f"{t('star_class_label')} {sys.star_class} · {sys.dist_ly} {t('ly_val')}")
        col1, col2 = st.columns(2)
        if col1.button(t('back_galaxy_btn'), use_container_width=True):
            st.session_state.view_mode      = 'galaxy'
            st.session_state.current_system = None
            st.session_state.current_planet = None
            st.rerun()
        if col2.button(t('dashboard_btn'), use_container_width=True):
            st.session_state.view_mode = 'dashboard'
            st.rerun()
        if sys.planets:
            st.markdown("---")
            pnames = [f"{p.icon} {p.name}  ({t('type_' + p.type)})" for p in sys.planets]
            sel_p = st.selectbox(t('select_planet'), range(len(pnames)), format_func=lambda i: pnames[i])
            planet_dict = sys.planets[sel_p].to_dict(lang=st.session_state['lang'])
            st.caption(f"📖 {planet_dict['desc']}")
            if st.button(t('land_btn'), use_container_width=True, type="primary"):
                st.session_state.view_mode      = 'planet'
                st.session_state.current_planet = sel_p
                key = f"{st.session_state.current_system}_{sel_p}"
                if key not in st.session_state.visited_planets:
                    st.session_state.visited_planets.append(key)
                st.rerun()

    elif mode == 'planet':
        sys    = universe.systems[st.session_state.current_system]
        planet = sys.planets[st.session_state.current_planet]
        planet_dict = planet.to_dict(lang=st.session_state['lang'])
        st.success(f"{planet.icon} **{planet.name}**")
        st.caption(f"{planet_dict['type_translated']} · {planet.temperature}°C")
        if st.button(t('launch_btn'), use_container_width=True):
            st.session_state.view_mode      = 'system'
            st.session_state.current_planet = None
            st.rerun()
        if st.button(t('back_galaxy_btn'), use_container_width=True):
            st.session_state.view_mode      = 'galaxy'
            st.session_state.current_system = None
            st.session_state.current_planet = None
            st.rerun()

    elif mode == 'dashboard':
        st.info(t('dashboard_title'))
        if st.button(t('back_galaxy_btn'), use_container_width=True):
            st.session_state.view_mode = 'galaxy'
            st.rerun()
        if st.button(t('theory_panel_btn'), use_container_width=True):
            st.session_state.view_mode = 'theory'
            st.rerun()

    elif mode == 'theory':
        st.info(t('theory_panel_btn'))
        if st.button(t('back_galaxy_btn'), use_container_width=True):
            st.session_state.view_mode = 'galaxy'
            st.rerun()
        if st.button(t('dashboard_btn'), use_container_width=True):
            st.session_state.view_mode = 'dashboard'
            st.rerun()

    st.divider()

    # ── Discovery Log ──
    st.markdown(f"### {t('discovery_title')}")
    st.markdown(f"{t('systems_visited')}: **{len(st.session_state.visited_systems)}** / {len(universe.systems)}")
    st.markdown(f"{t('planets_visited')}: **{len(st.session_state.visited_planets)}**")
    bg_count = len(json.loads(BACKGROUND_STARS_JSON))
    st.markdown(f"{t('bg_stars_loaded')}: **{bg_count}** {t('nasa_hosts')}")

# ============================================================
# MAIN CONTENT
# ============================================================
mode = st.session_state.view_mode

# ── GALAXY MAP ──────────────────────────────────────────────
if mode == 'galaxy':
    st.title(t('galaxy_map_title'))
    st.caption(t('galaxy_map_subtitle', bg_count=bg_count))

    # Controls are embedded in the HTML control bar (no Streamlit widgets overlap canvas)
    html = load_scene('galaxy_map.html')
    html = html.replace('__I18N_JSON__', json.dumps(I18N))
    systems_json_translated = json.dumps(universe.to_json(lang=st.session_state['lang']))
    
    # Translate epoch benchmarks
    translated_epochs = []
    for ep in EPOCH_BENCHMARKS:
        label_key = ep['label'].lower().replace(' ', '_').replace('era', '').strip('_')
        mapping = {
            'reionization': 'reionization',
            'galaxy_formation_peak': 'galaxy_formation',
            'solar_system_born': 'solar_born',
            'dark_energy_dominates': 'dark_energy_dom',
            'present_day': 'present_day'
        }
        key = mapping.get(label_key, label_key)
        translated_epochs.append({
            't': ep['t'],
            'a': ep['a'],
            'z': ep['z'],
            'label': t(key)
        })
    epochs_json = json.dumps(translated_epochs)

    html = html.replace('__SYSTEMS_JSON__',           systems_json_translated)
    html = html.replace('__LCDM_TABLE_JSON__',        LCDM_JSON)
    html = html.replace('__EPOCH_BENCHMARKS_JSON__',  epochs_json)
    html = html.replace('__BACKGROUND_STARS_JSON__',  BACKGROUND_STARS_JSON)
    html = html.replace('__CURRENT_LANG__',           json.dumps(st.session_state['lang']))
    # Height = canvas (540px) + control bar (156px) + small buffer (4px)
    components.html(html, height=700, scrolling=False)

    # ── System cards below ──
    st.divider()
    st.subheader(t('real_systems_header'))
    cols = st.columns(3)
    for i, sys in enumerate(universe.systems[:9]):
        v = "✓" if sys.index in st.session_state.visited_systems else "·"
        sys_dict = sys.to_dict(lang=st.session_state['lang'])
        with cols[i % 3]:
            st.markdown(f"""<div class="sci-card">
            <h4>{v} {sys.name}</h4>
            {t('star_class_label')} <b style="color:{sys.star_info['color']}">{sys.star_class}</b>
            ({sys.star_info['name']})<br>
            📏 <b>{sys.dist_ly}</b> {t('ly_val')} ·
            🪐 <b>{sys.num_planets}</b> {t('planets_val')}<br>
            <small>{sys_dict['desc']}</small>
            </div>""", unsafe_allow_html=True)

# ── STAR SYSTEM ──────────────────────────────────────────────
elif mode == 'system':
    sys = universe.systems[st.session_state.current_system]
    sys_dict = sys.to_dict(lang=st.session_state['lang'])
    st.title(f"☀ {sys.name}")

    m = st.columns(5)
    m[0].metric(t('star_class_label'),  f"{sys.star_class} ({sys.star_info['name']})")
    m[1].metric(t('distance_label'),    f"{sys.dist_ly} {t('ly_val')}")
    m[2].metric(t('planets_label'),     str(sys.num_planets))
    m[3].metric(t('economy_label'),     sys_dict['economy_translated'])
    m[4].metric(t('bound_state_label'), t('gravitational_val'))

    st.markdown(f"""<div class="edu-box">
    <h4>{t('astrophysical_data_header')}</h4>
    <p>{sys_dict['desc']}</p>
    <p>⚠️ <b>{t('key_concept_label')}</b> {t('key_concept_text')}</p>
    </div>""", unsafe_allow_html=True)

    planets_json = json.dumps([p.to_dict(lang=st.session_state['lang']) for p in sys.planets])
    html = load_scene('star_system.html')
    html = html.replace('__I18N_JSON__',     json.dumps(I18N))
    html = html.replace('__PLANETS_JSON__',  planets_json)
    html = html.replace('__STAR_COLOR__',    json.dumps(sys.star_info['color']))  # JSON-safe string
    html = html.replace('__SYSTEM_NAME__',   json.dumps(sys.name))        # JSON-safe: handles apostrophes
    html = html.replace('__STAR_CLASS__',    json.dumps(sys.star_class))
    html = html.replace('__NUM_PLANETS__',   str(sys.num_planets))
    html = html.replace('__DIST_LY__',       json.dumps(str(sys.dist_ly)))
    html = html.replace('__ECONOMY__',       json.dumps(sys.economy))     # JSON-safe raw English key
    html = html.replace('__SYS_DESC__',      json.dumps(sys_dict['desc']))
    html = html.replace('__CURRENT_LANG__',  json.dumps(st.session_state['lang']))
    components.html(html, height=620, scrolling=False)

    if sys.planets:
        st.divider()
        st.subheader(t('planets_in_sys', sys_name=sys.name))
        cols = st.columns(min(3, max(1, len(sys.planets))))
        for i, p in enumerate(sys.planets):
            key = f"{sys.index}_{i}"
            v   = "✓" if key in st.session_state.visited_planets else "·"
            hc  = "#a5d6a7" if p.hazard == "None" else "#ef9a9a"
            p_dict = p.to_dict(lang=st.session_state['lang'])
            with cols[i % len(cols)]:
                st.markdown(f"""<div class="sci-card">
                <h4>{v} {p.icon} {p.name}</h4>
                {t('planet_type_label')}: <b>{p_dict['type_translated']}</b> · 🌡️ <b>{p.temperature}°C</b><br>
                ⚠️ <span style="color:{hc}">{p_dict['hazard_translated']}</span> · 📏 <b>{p.radius}× {t('earth_val')}</b><br>
                💨 {p_dict['atmo']}<br>
                {'💧' if p.has_water else ''} {'💍 Rings' if p.has_rings else ''}<br>
                <small>📖 {p_dict['desc']}</small>
                </div>""", unsafe_allow_html=True)

# ── PLANET SURFACE ───────────────────────────────────────────
elif mode == 'planet':
    sys    = universe.systems[st.session_state.current_system]
    planet = sys.planets[st.session_state.current_planet]
    planet_dict = planet.to_dict(lang=st.session_state['lang'])

    st.title(f"{planet.icon} {planet.name}")

    m = st.columns(5)
    m[0].metric(t('planet_type_label'), planet_dict['type_translated'])
    m[1].metric(t('temperature_label'), f"{planet.temperature}°C")
    m[2].metric(t('hazard_label'),      planet_dict['hazard_translated'])
    m[3].metric(t('radius_label'),      f"{planet.radius}× {t('earth_val')}")
    m[4].metric(t('atmosphere_label'),  planet_dict['atmo'][:16])

    st.markdown(f"""<div class="edu-box">
    <h4>📚 {planet.name}</h4>
    <p>{planet_dict['desc']}</p>
    <p>🏠 {t('host_system_label')}: <b>{sys.name}</b> · 📏 {sys.dist_ly} {t('ly_val')} {t('from_earth_val')} ·
    Orbit: <b>{planet.orbital_dist} AU</b></p>
    </div>""", unsafe_allow_html=True)

    weather = WEATHER_MAP.get(planet.type, 'none')
    weather_translated = t(f"weather_{weather}")
    pcfg = {
        'sky':         planet.sky_color,
        'ground':      planet.ground_color,
        'seed_val':    hash(planet.seed) % 100000,
        'has_water':   planet.has_water,
        'flora':       planet.flora_count,
        'fauna_count': planet.fauna_count,
        'fauna_list':  planet.fauna,
        'resources':   planet_dict['resources'],
        'weather':     weather,
    }

    html = load_scene('planet_surface.html')
    html = html.replace('__I18N_JSON__', json.dumps(I18N))
    html = html.replace('__PLANET_CONFIG__',   json.dumps(pcfg))
    html = html.replace('__PLANET_NAME__',     planet.name)
    html = html.replace('__PLANET_TYPE__',     planet.type)            # raw English — JS does classify(type) + t('type_'+PLANET_TYPE)
    html = html.replace('__TEMPERATURE__',     str(int(planet.temperature)))
    html = html.replace('__HAZARD__',          planet.hazard)          # raw English — JS does t('hazard_'+HAZARD)
    html = html.replace('__FLORA__',           str(planet.flora_count))
    html = html.replace('__FAUNA__',           str(planet.fauna_count))
    html = html.replace('__ATMO__',            planet_dict['atmo'])
    html = html.replace('__SYSTEM_SEED__',     planet.system_seed)
    html = html.replace('__CURRENT_LANG__',    st.session_state['lang'])
    components.html(html, height=620, scrolling=False)

    # ── Scientific Data Columns ──
    st.divider()
    col_data, col_phys = st.columns(2)

    with col_data:
        st.subheader(t('physical_properties_header'))
        st.markdown(f"""
- 📏 {t('radius_label')}: **{planet.radius}× {t('earth_val')}**
- 💨 {t('atmosphere_label')}: **{planet_dict['atmo']}**
- 🌊 {t('liquid_water_label')}: **{t('yes_val') if planet.has_water else t('no_val')}**
- 💍 {t('ring_system_label')}: **{t('yes_val') if planet.has_rings else t('no_val')}**
- 🌤️ {t('weather_label')}: **{weather_translated}**
- 🌿 {t('flora_species_label')}: **{planet.flora_count}**
- 🐾 {t('fauna_species_label')}: **{planet.fauna_count}**
        """)

    with col_phys:
        st.subheader(t('orbital_mechanics_header'))
        st.markdown(f"""
- 🏠 {t('host_system_label')}: **{sys.name}**
- 📏 {t('system_distance_label')}: **{sys.dist_ly} {t('ly_val')}**
- 🔄 {t('orbital_distance_label')}: **{planet.orbital_dist} AU**
- ⭐ {t('star_class_label')}: **{sys.star_class}** ({sys.star_info['name']})
- ⚠️ {t('hazard_label')}: **{planet_dict['hazard_translated']}**
- 🌡️ {t('temperature_label')}: **{planet.temperature}°C**
        """)

# ── SCIENTIFIC DASHBOARD ─────────────────────────────────────
elif mode == 'dashboard':
    st.title(t('dashboard_title'))
    st.caption(t('dashboard_subtitle'))

    # Translate epoch benchmarks
    translated_epochs = []
    for ep in EPOCH_BENCHMARKS:
        label_key = ep['label'].lower().replace(' ', '_').replace('era', '').strip('_')
        mapping = {
            'reionization': 'reionization',
            'galaxy_formation_peak': 'galaxy_formation',
            'solar_system_born': 'solar_born',
            'dark_energy_dominates': 'dark_energy_dom',
            'present_day': 'present_day'
        }
        key = mapping.get(label_key, label_key)
        translated_epochs.append({
            't': ep['t'],
            'a': ep['a'],
            'z': ep['z'],
            'label': t(key)
        })
    epochs_json = json.dumps(translated_epochs)

    html = load_scene('dashboard.html')
    html = html.replace('__I18N_JSON__', json.dumps(I18N))
    html = html.replace('__LCDM_CHART_JSON__',      LCDM_CHART_JSON)
    html = html.replace('__HUBBLE_DATA_JSON__',      HUBBLE_PTS_JSON)
    html = html.replace('__HUBBLE_SLOPE__',          str(HUBBLE_SLOPE))
    html = html.replace('__HUBBLE_INTERCEPT__',      str(HUBBLE_INTERCEPT))
    html = html.replace('__HUBBLE_R2__',             str(HUBBLE_R2))
    html = html.replace('__REDSHIFT_DATA_JSON__',    REDSHIFT_JSON)
    html = html.replace('__EPOCH_BENCHMARKS_JSON__', epochs_json)
    html = html.replace('__CURRENT_LANG__',          json.dumps(st.session_state['lang']))
    components.html(html, height=700, scrolling=False)

# ── THEORY & EQUATION PANEL ──────────────────────────────────
elif mode == 'theory':
    st.title(t('theory_title'))
    st.caption(t('theory_subtitle'))
    html = load_scene('theory_panel.html')
    html = html.replace('__I18N_JSON__', json.dumps(I18N))
    html = html.replace('__CURRENT_LANG__', json.dumps(st.session_state['lang']))
    components.html(html, height=1400, scrolling=True)

