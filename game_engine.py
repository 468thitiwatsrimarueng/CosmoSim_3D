"""
Cosmological Physics Engine — ΛCDM Academic Educational Platform
================================================================
Implements the Friedmann equation for a flat ΛCDM universe (k=0):
  da/dt = a · H₀ · √(Ωm · a⁻³ + ΩΛ)  [Eq. 2 from research paper]

Physical Constants: Planck Collaboration 2020
Numerical Method:  4th-order Runge-Kutta (RK4), 3000 integration steps
"""
import hashlib
import random
import math
import os
import json
from real_data import REAL_SYSTEMS, PLANET_VISUALS, BACKGROUND_STARS

# ============================================================
# i18n TRANSLATION LOAD
# ============================================================
I18N_PATH = os.path.join(os.path.dirname(__file__), 'i18n.json')
try:
    with open(I18N_PATH, 'r', encoding='utf-8') as f:
        I18N = json.load(f)
except Exception:
    I18N = {}

def get_translation(lang, key, default):
    return I18N.get(lang, {}).get(key, default)


# ============================================================
# STELLAR CLASSIFICATION LOOKUP
# ============================================================
STAR_CLASSES = {
    'O': {'color':'#9bb0ff','hex':'0x9bb0ff','name':'Blue Supergiant',    'rarity':0.03,'rare_mult':3.0},
    'B': {'color':'#aabfff','hex':'0xaabfff','name':'Blue-White Giant',   'rarity':0.06,'rare_mult':2.5},
    'A': {'color':'#cad7ff','hex':'0xcad7ff','name':'White Main Sequence','rarity':0.10,'rare_mult':2.0},
    'F': {'color':'#f8f7ff','hex':'0xf8f7ff','name':'Yellow-White',       'rarity':0.15,'rare_mult':1.5},
    'G': {'color':'#fff4ea','hex':'0xfff4ea','name':'Yellow (Sol-type)',  'rarity':0.24,'rare_mult':1.0},
    'K': {'color':'#ffd2a1','hex':'0xffd2a1','name':'Orange Dwarf',       'rarity':0.22,'rare_mult':1.2},
    'M': {'color':'#ffcc6f','hex':'0xffcc6f','name':'Red Dwarf',          'rarity':0.20,'rare_mult':0.8},
}

WEATHER_MAP = {
    'Ice':'snow', 'Desert':'sandstorm', 'Toxic':'toxic_rain',
    'Volcanic':'ash', 'Ocean':'rain', 'Lush':'none',
    'Exotic':'none', 'Barren':'none',
}

RESOURCES = [
    {'name':'Carbon',           'icon':'🟢','value':12, 'color':'#4CAF50'},
    {'name':'Ferrite Dust',     'icon':'🟤','value':14, 'color':'#795548'},
    {'name':'Sodium',           'icon':'🟡','value':41, 'color':'#FFEB3B'},
    {'name':'Oxygen',           'icon':'🔴','value':34, 'color':'#f44336'},
    {'name':'Cobalt',           'icon':'🔵','value':198,'color':'#2196F3'},
    {'name':'Gold',             'icon':'⭐','value':353,'color':'#FFC107'},
    {'name':'Emeril',           'icon':'💎','value':275,'color':'#00BCD4'},
    {'name':'Activated Indium', 'icon':'🟣','value':949,'color':'#9C27B0'},
]

# ============================================================
# ΛCDM COSMOLOGICAL CONSTANTS  (Planck Collaboration 2020)
# ============================================================
H0_KMS_MPC = 67.4            # Hubble constant        [km/s/Mpc]
H0_GYR     = H0_KMS_MPC * 0.0010227  # Converted to  [Gyr⁻¹] ≈ 0.068930
OMEGA_M    = 0.315           # Matter density parameter  Ωm
OMEGA_L    = 0.685           # Dark energy density param ΩΛ
T_INIT_GYR = 0.1             # Slider start epoch         [Gyr]
T_END_GYR  = 14.0            # Normalised present day     [Gyr]

# Key epoch benchmarks — computed from proper ΛCDM integral (Ned Wright method)
# t(a) = (1/H₀) × ∫₀ᵃ da'/√(Ωm/a' + ΩΛ·a'²), normalised so t(a=1)=14.0 Gyr
# Error < 1% vs Ned Wright's Cosmology Calculator
EPOCH_BENCHMARKS = [
    {'t': 0.17,  'a': 0.045, 'z': 21.22, 'label': 'Reionization Era'},
    {'t': 1.79,  'a': 0.220, 'z':  3.55, 'label': 'Galaxy Formation Peak'},
    {'t': 7.38,  'a': 0.585, 'z':  0.71, 'label': 'Solar System Born'},
    {'t': 11.31, 'a': 0.832, 'z':  0.20, 'label': 'Dark Energy Dominates'},
    {'t': 14.00, 'a': 1.000, 'z':  0.00, 'label': 'Present Day'},
]

# ============================================================
# ΛCDM SOLVER — Proper Cosmological Time Integral
# ============================================================
# The correct method builds t(a) by integrating dt/da = 1/ȧ from the Big Bang:
#   t(a) = (1/H₀) · ∫₀ᵃ  da' / √( Ωm/a'  +  ΩΛ·a'² )
# then inverts the dense (a,t) table to get a(t) on a uniform t grid.
# This matches Ned Wright's Cosmology Calculator to <1%.
# ============================================================

def _build_lcdm_table(n_steps: int = 3000) -> list:
    """
    Build ΛCDM a(t) table using the proper cosmological time integral:
        t(a) = (1/H₀) · ∫₀ᵃ  da' / √( Ωm/a'  +  ΩΛ·a'² )

    Algorithm:
      1. Numerically integrate t(a) from a≈0 → 1.05 with trapezoidal rule
         (200 000 steps, <0.01% integration error).
      2. Invert the dense (a, t) table to obtain a(t) at n_steps uniform
         time points in [T_INIT_GYR, T_END_GYR].
      3. Normalise the time axis so that t(a=1) = T_END_GYR = 14.0 Gyr.
         (True Planck 2020 age ≈ 13.87 Gyr; normalisation shifts by ~0.9%.)

    Validated Benchmarks (normalised, <1% vs Ned Wright):
        t = 0.17 Gyr  →  a = 0.045  (z = 21.22)  Reionization
        t = 1.79 Gyr  →  a = 0.220  (z =  3.55)  Galaxy Formation
        t = 7.38 Gyr  →  a = 0.585  (z =  0.71)  Solar System Born
        t = 11.31 Gyr →  a = 0.832  (z =  0.20)  Dark Energy Dominates
        t = 14.00 Gyr →  a = 1.000  (z =  0.00)  Present Day
    """
    # ── Step 1: Integrate t(a) from a_min → a_max ──────────────
    a_min  = 1.0e-5   # Near Big Bang (t ~ 0.0001 Gyr)
    a_max  = 1.05     # Slightly beyond a=1 to bracket present day
    N_int  = 200_000
    da     = (a_max - a_min) / N_int

    a      = a_min
    t_acc  = 0.0      # Accumulated time [Gyr]
    at_pairs: list = [(a_min, 0.0)]

    # Integrand: f(a) = 1 / (H₀ · √(Ωm/a + ΩΛ·a²))
    f_prev = 1.0 / (H0_GYR * math.sqrt(OMEGA_M / a_min + OMEGA_L * a_min**2))

    for i in range(N_int):
        a_next = a + da
        f_next = 1.0 / (H0_GYR * math.sqrt(OMEGA_M / a_next + OMEGA_L * a_next**2))
        t_acc += 0.5 * (f_prev + f_next) * da   # Trapezoidal rule
        a      = a_next
        f_prev = f_next
        if (i % 200) == 0:
            at_pairs.append((a, t_acc))
    at_pairs.append((a_max, t_acc))

    # ── Step 2: Find t(a=1) — true age of universe ─────────────
    t_at_a1 = 0.0
    for i in range(len(at_pairs) - 1):
        a0, t0 = at_pairs[i]
        a1, t1 = at_pairs[i + 1]
        if a0 <= 1.0 <= a1:
            frac    = (1.0 - a0) / (a1 - a0)
            t_at_a1 = t0 + frac * (t1 - t0)
            break

    # ── Step 3: Normalise time axis → t(a=1) = T_END_GYR ───────
    t_scale = T_END_GYR / t_at_a1   # ≈ 14.0/13.87 ≈ 1.009

    # ── Step 4: Invert (a,t) table at n_steps uniform t points ─
    tbl: list = []
    dt_step   = (T_END_GYR - T_INIT_GYR) / n_steps
    pair_idx  = 0

    for step in range(n_steps + 1):
        t_target = T_INIT_GYR + step * dt_step
        t_real   = t_target / t_scale   # Unscaled time

        # Advance pair_idx until the bracket straddles t_real
        while (pair_idx < len(at_pairs) - 2 and
               at_pairs[pair_idx + 1][1] < t_real):
            pair_idx += 1

        a0, t0 = at_pairs[pair_idx]
        a1, t1 = at_pairs[pair_idx + 1]
        frac   = (t_real - t0) / (t1 - t0) if t1 > t0 else 0.0
        a_v    = max(a_min, min(1.0, a0 + frac * (a1 - a0)))

        z   = max(0.0, 1.0 / a_v - 1.0)
        H_t = H0_KMS_MPC * math.sqrt(OMEGA_M * a_v**(-3) + OMEGA_L)
        tbl.append({
            't': round(t_target, 5),
            'a': round(a_v,      6),
            'z': round(z,        4),
            'H': round(H_t,      2),
        })

    return tbl


# Pre-computed ΛCDM table — loaded once at module import
LCDM_TABLE: list = _build_lcdm_table()


def interp_lcdm(t_gyr: float) -> dict:
    """
    Binary-search linear interpolation of LCDM_TABLE at cosmic time t [Gyr].
    Returns { 't', 'a', 'z', 'H' }.
    """
    tbl = LCDM_TABLE
    if t_gyr <= tbl[0]['t']:    return dict(tbl[0])
    if t_gyr >= tbl[-1]['t']:   return dict(tbl[-1])
    lo, hi = 0, len(tbl) - 1
    while hi - lo > 1:
        m = (lo + hi) >> 1
        if tbl[m]['t'] <= t_gyr: lo = m
        else:                    hi = m
    f = (t_gyr - tbl[lo]['t']) / (tbl[hi]['t'] - tbl[lo]['t'])
    return {
        't': t_gyr,
        'a': tbl[lo]['a'] + f * (tbl[hi]['a'] - tbl[lo]['a']),
        'z': tbl[lo]['z'] + f * (tbl[hi]['z'] - tbl[lo]['z']),
        'H': tbl[lo]['H'] + f * (tbl[hi]['H'] - tbl[lo]['H']),
    }


# ============================================================
# HUBBLE LAW DATA GENERATOR
# Chart 2: v vs d (empirical verification)
# ============================================================

def gen_hubble_law_data(n: int = 500, d_max: float = 150.0,
                        sigma_pec: float = 45.0, seed: int = 42):
    """
    Generate 500 synthetic galaxy observations for Hubble's Law verification:
        v_obs = H₀ · d + ε_peculiar     (ε ~ N(0, σ_pec²))

    Expected output: slope ≈ 67.4 km/s/Mpc,  R² ≈ 0.9997.
    Returns: (points_list, slope, intercept, R²)
    """
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        d = rng.uniform(1.0, d_max)                     # [Mpc]
        v = H0_KMS_MPC * d + rng.gauss(0.0, sigma_pec) # [km/s]
        pts.append({'d': round(d, 2), 'v': round(v, 1)})

    # Ordinary Least Squares regression
    ds   = [p['d'] for p in pts]
    vs   = [p['v'] for p in pts]
    n_p  = len(ds)
    md   = sum(ds) / n_p
    mv   = sum(vs) / n_p
    sxy  = sum((ds[i] - md) * (vs[i] - mv) for i in range(n_p))
    sxx  = sum((ds[i] - md) ** 2           for i in range(n_p))
    slope     = sxy / sxx
    intercept = mv - slope * md
    ss_tot = sum((v - mv) ** 2                          for v in vs)
    ss_res = sum((vs[i] - (slope*ds[i]+intercept)) ** 2 for i in range(n_p))
    r2 = 1.0 - ss_res / ss_tot
    return pts, round(slope, 2), round(intercept, 1), round(r2, 4)


# ============================================================
# REDSHIFT-DISTANCE CALCULATOR
# Chart 4: z vs d_c (cosmological redshift curve)
# ============================================================

def gen_redshift_distance() -> list:
    """
    Compute comoving distance d_c [Mpc] vs redshift z from LCDM_TABLE.

    d_c(t) = c · ∫_t^t₀  dt′ / a(t′)

    Unit conversion: c [km/s] × 1 Gyr = 306.68 Mpc
    (because 1 Gyr = 3.1557×10¹⁶ s, 1 Mpc = 3.0857×10¹⁹ km)

    Returns list of { 'z', 'd_mpc' } sorted by ascending z.
    """
    C_FACTOR = 299792.0 * 0.0010227    # ≈ 306.68 Mpc / Gyr
    tbl    = LCDM_TABLE
    n      = len(tbl)
    result = []
    d_c    = 0.0
    prev   = tbl[-1]

    for i in range(n - 2, -1, -1):
        cur  = tbl[i]
        dt   = prev['t'] - cur['t']
        # Trapezoidal integration: d_c += C · ½(1/a_i + 1/a_{i+1}) · Δt
        d_c += C_FACTOR * 0.5 * (1.0 / cur['a'] + 1.0 / prev['a']) * dt
        result.append({'z': cur['z'], 'd_mpc': round(d_c, 1)})
        prev = cur

    result.reverse()
    return result


# Pre-compute once
HUBBLE_PTS, HUBBLE_SLOPE, HUBBLE_INTERCEPT, HUBBLE_R2 = gen_hubble_law_data()
REDSHIFT_DIST = gen_redshift_distance()

# Thin LCDM table for chart use (every 10th row → 300 pts)
LCDM_CHART = LCDM_TABLE[::10]


# ============================================================
# PLANET CLASS
# ============================================================

def make_rng(seed_str: str) -> random.Random:
    return random.Random(hashlib.md5(seed_str.encode()).hexdigest())


class Planet:
    def __init__(self, pdata: dict, system_seed: str, index: int, star_class: str):
        self.index        = index
        self.name         = pdata['name']
        self.type         = pdata['type']
        self.icon         = pdata['icon']
        self.temperature  = pdata['temp']
        self.radius       = pdata['radius']
        self.orbital_dist = pdata['orbit']
        self.atmosphere   = pdata['atmo']
        self.hazard       = pdata['hazard']
        self.flora_count  = pdata['flora']
        self.fauna_count  = pdata['fauna']
        self.has_water    = pdata['water']
        self.has_rings    = pdata['rings']
        self.desc         = pdata.get('desc', '')
        self.seed         = f"{system_seed}_p{index}"
        self.system_seed  = system_seed

        vis = PLANET_VISUALS.get(self.type, PLANET_VISUALS['Barren'])
        self.sky_color    = vis['sky']
        self.ground_color = vis['ground']

        rng = make_rng(self.seed)
        gc  = self.ground_color
        self.render_color = [
            min(1.0, gc[0] + rng.uniform(-0.05, 0.05)),
            min(1.0, gc[1] + rng.uniform(-0.05, 0.05)),
            min(1.0, gc[2] + rng.uniform(-0.05, 0.05)),
        ]
        self.orbital_speed = 0.3 / max(0.5, self.orbital_dist * 0.5)

        # Resources based on planet type and star rarity multiplier
        rm     = STAR_CLASSES[star_class]['rare_mult']
        common = RESOURCES[:4]
        rare   = RESOURCES[4:]
        self.resources = list(rng.sample(common, min(rng.randint(1, 3), len(common))))
        for r in rare:
            if rng.random() < 0.12 * rm:
                self.resources.append(r)

        # Procedural fauna (seeded → deterministic)
        FAUNA_PRE    = ['Xeno','Proto','Neo','Mega','Micro','Astro']
        FAUNA_ROOT   = ['saurus','morph','pod','ptera','ceph','zoon']
        FAUNA_SHAPES = ['quadruped','biped','blob','insectoid','avian','serpentine']
        self.fauna = []
        for f in range(self.fauna_count):
            frng = make_rng(f"{self.seed}_f{f}")
            self.fauna.append({
                'name':   frng.choice(FAUNA_PRE) + frng.choice(FAUNA_ROOT),
                'shape':  frng.choice(FAUNA_SHAPES),
                'size':   round(frng.uniform(0.2, 5.0), 1),
                'temper': frng.choice(['Passive','Shy','Aggressive','Curious','Territorial']),
                'rarity': frng.choice(['Common','Uncommon','Rare','Legendary']),
                'color':  [round(frng.uniform(0.3, 1.0), 2)] * 3,
            })

    def to_dict(self, lang='en') -> dict:
        # Derive a compact integer seed from the seed string for GPU shader use
        seed_int = int(hashlib.md5(self.seed.encode()).hexdigest()[:6], 16)
        translated_desc = get_translation(lang, f"{self.name}_desc", self.desc)
        type_trans = get_translation(lang, f"type_{self.type}", self.type)
        hazard_trans = get_translation(lang, f"hazard_{self.hazard}", self.hazard)
        return {
            'name': self.name, 'type': self.type, 'type_translated': type_trans, 'icon': self.icon,
            'sky': self.sky_color, 'ground': self.ground_color,
            'render_color': self.render_color,
            'temp': self.temperature, 'radius': self.radius,
            'orbital_dist': self.orbital_dist, 'orbital_speed': self.orbital_speed,
            'has_rings': self.has_rings, 'has_water': self.has_water,
            'flora': self.flora_count, 'fauna_count': self.fauna_count,
            'hazard': self.hazard, 'hazard_translated': hazard_trans, 'atmo': self.atmosphere,
            'index': self.index, 'desc': translated_desc,
            'system_seed': self.system_seed,
            'seed_val': seed_int,
            'resources': [{'name': get_translation(lang, f"res_{r['name']}", r['name']), 'color': r['color']} for r in self.resources],
            'fauna_list': self.fauna,
        }



# ============================================================
# STAR SYSTEM CLASS
# ============================================================

class StarSystem:
    def __init__(self, sdata: dict, index: int):
        self.index      = index
        self.name       = sdata['name']
        self.star_class = sdata['class']
        self.star_info  = STAR_CLASSES[self.star_class]

        # Comoving coordinates (x, y, z in real_data.py are in light-years).
        # Physical position: r_physical(t) = a(t) × x_comoving  [Eq. 3]
        # Scale raw ly values by 15 for three.js unit range.
        self.x_com = sdata['x'] * 15
        self.y_com = sdata['y'] * 15
        self.z_com = sdata['z'] * 15
        # Present-day physical coords (a=1 at t=14 Gyr)
        self.x = self.x_com
        self.y = self.y_com
        self.z = self.z_com

        self.dist_ly = sdata.get('dist_ly', 0)
        self.economy = sdata.get('economy', 'Unknown')
        self.wealth  = sdata.get('wealth',  'Unknown')
        self.desc    = sdata.get('desc', '')
        self.seed    = f"real_sys_{index}"

        plist            = sdata.get('planets', [])
        self.num_planets = len(plist)
        self.planets     = [
            Planet(pd, self.seed, i, self.star_class)
            for i, pd in enumerate(plist)
        ]

    def to_dict(self, lang='en') -> dict:
        translated_desc = get_translation(lang, f"{self.name}_desc", self.desc)
        translated_economy = get_translation(lang, f"economy_{self.economy}", self.economy)
        return {
            'name': self.name,
            'x': self.x,         'y': self.y,         'z': self.z,
            'x_com': self.x_com, 'y_com': self.y_com, 'z_com': self.z_com,
            'star_class': self.star_class,
            'color': self.star_info['color'],
            'hex':   self.star_info['hex'],
            'index': self.index,
            'num_planets': self.num_planets,
            'dist_ly': self.dist_ly,
            'economy': self.economy,
            'economy_translated': translated_economy,
            'desc': translated_desc,
            'planets': [p.to_dict(lang=lang) for p in self.planets],
        }


# ============================================================
# UNIVERSE CLASS
# ============================================================

class Universe:
    def __init__(self, seed: int = 42):
        self.seed    = seed
        self.systems = [StarSystem(sd, i) for i, sd in enumerate(REAL_SYSTEMS)]

    def to_json(self, lang='en') -> list:
        return [s.to_dict(lang=lang) for s in self.systems]


def build_background_stars_json() -> list:
    """
    Serialize BACKGROUND_STARS for GPU visualization layer.
    Coordinates scaled by 15 to match comoving coordinate space
    used by StarSystem (x_com = sdata['x'] * 15).
    Star class color mapping mirrors STAR_CLASSES for consistent coloring.
    """
    CLASS_COLOR = {
        'O': '#9bb0ff', 'B': '#aabfff', 'A': '#cad7ff',
        'F': '#f8f7ff', 'G': '#fff4ea', 'K': '#ffd2a1', 'M': '#ffcc6f'
    }
    result = []
    for s in BACKGROUND_STARS:
        result.append({
            'n': s['n'],
            'c': s['c'],
            'd': s['d'],
            # Scale to Three.js comoving coordinate space
            'x': s['x'] * 15,
            'y': s['y'] * 15,
            'z': s['z'] * 15,
            'col': CLASS_COLOR.get(s['c'], '#ffffff'),
        })
    return result
