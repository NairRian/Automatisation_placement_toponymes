"""
Moteur de placement de toponymes cartographiques.
Génère toutes les positions candidates, les note selon une hiérarchie pondérée,
les conflits avec les vecteurs et les chevauchements entre étiquettes.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Énumérations : types de vecteurs & classes internes
# ---------------------------------------------------------------------------

class VectorType(Enum):
    # Ponctuels
    VILLE = "ville"
    AEROPORT = "aeroport"
    PORT = "port"
    SOMMET = "sommet"
    # Linéaires
    ROUTE = "route"
    FRONTIERE_NATIONALE = "frontiere_nationale"
    FRONTIERE_REGIONALE = "frontiere_regionale"
    VOIE_FERREE = "voie_ferree"
    HYDRO = "hydro"
    # Surfaciques
    PAYS = "pays"
    REGION = "region"
    OCEAN_MER = "ocean_mer"
    LAC = "lac"
    RESERVE = "reserve"
    ZONE_URBAINE = "zone_urbaine"
    MONTAGNE = "montagne"


class VilleClasse(Enum):
    CAPITALE = "capitale"
    CHEF_LIEU = "chef_lieu"
    VILLE = "ville"


class AeroportClasse(Enum):
    MILITAIRE = "militaire"
    CIVIL = "civil"
    CIVILO_MILITAIRE = "civilo_militaire"
    INDIFFERENCIE = "indifferencie"


class RouteClasse(Enum):
    PRIMAIRE = "primaire"
    SECONDAIRE = "secondaire"
    TERTIAIRE = "tertiaire"


# ---------------------------------------------------------------------------
# Règles de labellisation
# ---------------------------------------------------------------------------

LABELED_VECTORS = {
    VectorType.VILLE,
    VectorType.AEROPORT,
    VectorType.PORT,
    VectorType.SOMMET,
    VectorType.ROUTE,          # seulement primaire (filtré dans get_label_config)
    VectorType.FRONTIERE_NATIONALE,
    VectorType.FRONTIERE_REGIONALE,
    VectorType.VOIE_FERREE,
    VectorType.HYDRO,
    VectorType.PAYS,
    VectorType.REGION,
    VectorType.OCEAN_MER,
    VectorType.LAC,
    VectorType.RESERVE,
    VectorType.MONTAGNE,
    # ZONE_URBAINE : non labellisée
}


def needs_label(vtype: VectorType, classe=None) -> bool:
    if vtype == VectorType.ZONE_URBAINE:
        return False
    if vtype == VectorType.ROUTE:
        return classe == RouteClasse.PRIMAIRE
    return vtype in LABELED_VECTORS


# ---------------------------------------------------------------------------
# Géométries simplifiées
# ---------------------------------------------------------------------------

@dataclass
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass
class BBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center(self) -> Point:
        return Point((self.xmin + self.xmax) / 2, (self.ymin + self.ymax) / 2)

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.xmax <= other.xmin or other.xmax <= self.xmin or
            self.ymax <= other.ymin or other.ymax <= self.ymin
        )

    def contains(self, pt: Point) -> bool:
        return self.xmin <= pt.x <= self.xmax and self.ymin <= pt.y <= self.ymax

    def expand(self, margin: float) -> "BBox":
        return BBox(self.xmin - margin, self.ymin - margin,
                    self.xmax + margin, self.ymax + margin)


# ---------------------------------------------------------------------------
# Positions candidates — hiérarchie & offsets
# ---------------------------------------------------------------------------

"""
Schéma de numérotation des positions candidates pour un point ancre :

  7   0   1
  6  [P]  2
  5   4   3

Positions 0–7 : octants autour du symbole (sens horaire depuis nord)
Position  8   : sur le symbole (pour objets surfaciques — centroïde)
Position  9   : le long d'un axe (pour linéaires — milieu de segment)
"""


class PositionID(Enum):
    N  = 0   # nord
    NE = 1
    E  = 2   # est
    SE = 3
    S  = 4   # sud
    SW = 5
    W  = 6   # ouest
    NW = 7
    CENTER = 8   # centroïde (surfaciques)
    ALONG  = 9   # le long d'un axe (linéaires)


# Hiérarchie de préférence par type de vecteur
# (liste ordonnée du plus préféré au moins préféré)
POSITION_HIERARCHY: dict[VectorType, list[PositionID]] = {
    VectorType.VILLE: [
        PositionID.NE, PositionID.N, PositionID.E,
        PositionID.SE, PositionID.NW, PositionID.S,
        PositionID.SW, PositionID.W,
    ],
    VectorType.AEROPORT: [
        PositionID.N, PositionID.NE, PositionID.NW,
        PositionID.E, PositionID.W,
        PositionID.SE, PositionID.SW, PositionID.S,
    ],
    VectorType.PORT: [
        PositionID.E, PositionID.NE, PositionID.SE,
        PositionID.N, PositionID.S,
        PositionID.NW, PositionID.SW, PositionID.W,
    ],
    VectorType.SOMMET: [
        PositionID.N, PositionID.NE, PositionID.NW,
        PositionID.E, PositionID.W,
        PositionID.SE, PositionID.SW, PositionID.S,
    ],
    VectorType.ROUTE: [
        PositionID.ALONG,
        PositionID.N, PositionID.S,
    ],
    VectorType.FRONTIERE_NATIONALE: [PositionID.ALONG],
    VectorType.FRONTIERE_REGIONALE: [PositionID.ALONG],
    VectorType.VOIE_FERREE: [PositionID.ALONG, PositionID.N, PositionID.S],
    VectorType.HYDRO: [PositionID.ALONG, PositionID.N, PositionID.S],
    VectorType.PAYS: [PositionID.CENTER],
    VectorType.REGION: [PositionID.CENTER],
    VectorType.OCEAN_MER: [PositionID.CENTER],
    VectorType.LAC: [PositionID.CENTER],
    VectorType.RESERVE: [PositionID.CENTER],
    VectorType.ZONE_URBAINE: [],  # non labellisée
    VectorType.MONTAGNE: [PositionID.CENTER, PositionID.N],
}

# Scores de départ selon rang dans la hiérarchie (position 0 = meilleure)
RANK_BASE_SCORE = 100.0
RANK_DECAY = 10.0   # perte par rang


# Offset en unités cartographiques selon la taille du symbole
DEFAULT_SYMBOL_RADIUS = 5.0
LABEL_OFFSET_FACTOR = 1.2   # distance = rayon * facteur


def _octant_offset(pos: PositionID, radius: float) -> tuple[float, float]:
    """Retourne (dx, dy) en unités carte pour une position octante."""
    r = radius * LABEL_OFFSET_FACTOR
    d = r / math.sqrt(2)
    offsets = {
        PositionID.N:  (0,  r),
        PositionID.NE: (d,  d),
        PositionID.E:  (r,  0),
        PositionID.SE: (d, -d),
        PositionID.S:  (0, -r),
        PositionID.SW: (-d, -d),
        PositionID.W:  (-r, 0),
        PositionID.NW: (-d,  d),
    }
    return offsets.get(pos, (0, 0))


# ---------------------------------------------------------------------------
# Structures de données principales
# ---------------------------------------------------------------------------

@dataclass
class LabelConfig:
    """Paramètres typographiques d'une étiquette."""
    font_size: float = 8.0
    font_bold: bool = False
    font_italic: bool = False
    letter_spacing: float = 0.0
    char_width_ratio: float = 0.6   # largeur moyenne d'un caractère / font_size

    def text_bbox(self, text: str, anchor: Point, angle_deg: float = 0.0) -> BBox:
        """Calcule la bbox approchée de l'étiquette à partir de l'ancre."""
        w = len(text) * self.font_size * self.char_width_ratio
        h = self.font_size * 1.2
        hw, hh = w / 2, h / 2
        return BBox(anchor.x - hw, anchor.y - hh, anchor.x + hw, anchor.y + hh)


@dataclass
class CandidatePosition:
    """Une position candidate pour un toponyme."""
    position_id: PositionID
    rank: int                        # rang dans la hiérarchie (0 = meilleur)
    anchor: Point                    # point d'ancrage de l'étiquette
    angle_deg: float = 0.0           # rotation de l'étiquette
    score: float = 0.0               # score final (plus élevé = meilleur)
    penalties: dict[str, float] = field(default_factory=dict)

    @property
    def base_score(self) -> float:
        return max(0.0, RANK_BASE_SCORE - self.rank * RANK_DECAY)


@dataclass
class VectorFeature:
    """Un objet géographique vecteur."""
    fid: str
    vtype: VectorType
    name: str
    geometry_points: list[Point]         # pour ponctuel : [point] ; linéaire : liste ; surfacique : contour
    classe: Optional[object] = None      # VilleClasse, AeroportClasse, RouteClasse …
    symbol_radius: float = DEFAULT_SYMBOL_RADIUS
    bbox: Optional[BBox] = None
    label_config: Optional[LabelConfig] = None

    def __post_init__(self):
        if self.bbox is None and self.geometry_points:
            xs = [p.x for p in self.geometry_points]
            ys = [p.y for p in self.geometry_points]
            self.bbox = BBox(min(xs), min(ys), max(xs), max(ys))
        if self.label_config is None:
            self.label_config = _default_label_config(self.vtype, self.classe)

    @property
    def anchor_point(self) -> Point:
        """Point de référence géométrique de l'objet."""
        if self.vtype in (VectorType.PAYS, VectorType.REGION,
                          VectorType.OCEAN_MER, VectorType.LAC,
                          VectorType.RESERVE, VectorType.ZONE_URBAINE,
                          VectorType.MONTAGNE):
            return self.bbox.center
        if self.vtype in (VectorType.ROUTE, VectorType.FRONTIERE_NATIONALE,
                          VectorType.FRONTIERE_REGIONALE, VectorType.VOIE_FERREE,
                          VectorType.HYDRO):
            # milieu du segment le plus long
            return _midpoint_of_longest_segment(self.geometry_points)
        return self.geometry_points[0]


def _default_label_config(vtype: VectorType, classe=None) -> LabelConfig:
    configs = {
        VectorType.VILLE: {
            VilleClasse.CAPITALE:  LabelConfig(font_size=12, font_bold=True),
            VilleClasse.CHEF_LIEU: LabelConfig(font_size=10, font_bold=True),
            VilleClasse.VILLE:     LabelConfig(font_size=8),
        },
        VectorType.AEROPORT: LabelConfig(font_size=7, font_italic=True),
        VectorType.PORT:     LabelConfig(font_size=7, font_italic=True),
        VectorType.SOMMET:   LabelConfig(font_size=7, font_italic=True),
        VectorType.ROUTE:    LabelConfig(font_size=6, letter_spacing=0.5),
        VectorType.HYDRO:    LabelConfig(font_size=7, font_italic=True),
        VectorType.PAYS:     LabelConfig(font_size=14, font_bold=True, letter_spacing=2.0),
        VectorType.REGION:   LabelConfig(font_size=10, letter_spacing=1.0),
        VectorType.OCEAN_MER:LabelConfig(font_size=12, font_italic=True, letter_spacing=2.0),
        VectorType.LAC:      LabelConfig(font_size=9,  font_italic=True),
        VectorType.RESERVE:  LabelConfig(font_size=8),
        VectorType.MONTAGNE: LabelConfig(font_size=9,  font_bold=True, font_italic=True),
    }
    cfg = configs.get(vtype)
    if isinstance(cfg, dict):
        return cfg.get(classe, LabelConfig())
    return cfg or LabelConfig()


def _midpoint_of_longest_segment(pts: list[Point]) -> Point:
    if len(pts) < 2:
        return pts[0] if pts else Point(0, 0)
    best_len, best_mid = -1, pts[0]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg_len = a.distance_to(b)
        if seg_len > best_len:
            best_len = seg_len
            best_mid = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
    return best_mid


def _segment_angle(pts: list[Point]) -> float:
    """Angle moyen du segment le plus long (en degrés)."""
    if len(pts) < 2:
        return 0.0
    best_len, best_angle = -1, 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        seg_len = a.distance_to(b)
        if seg_len > best_len:
            best_len = seg_len
            best_angle = math.degrees(math.atan2(b.y - a.y, b.x - a.x))
    # On garde l'angle entre -90 et +90 pour lisibilité
    while best_angle > 90:
        best_angle -= 180
    while best_angle < -90:
        best_angle += 180
    return best_angle


# ---------------------------------------------------------------------------
# Générateur de positions candidates
# ---------------------------------------------------------------------------

class CandidateGenerator:
    """
    Pour chaque objet vecteur labellisé, génère la liste ordonnée
    des positions candidates (CandidatePosition).
    """

    def generate(self, feature: VectorFeature) -> list[CandidatePosition]:
        if not needs_label(feature.vtype, feature.classe):
            return []

        hierarchy = POSITION_HIERARCHY.get(feature.vtype, [])
        candidates: list[CandidatePosition] = []
        anchor = feature.anchor_point

        for rank, pos_id in enumerate(hierarchy):
            if pos_id == PositionID.CENTER:
                label_anchor = anchor
                angle = 0.0
            elif pos_id == PositionID.ALONG:
                label_anchor = anchor
                angle = _segment_angle(feature.geometry_points)
            else:
                dx, dy = _octant_offset(pos_id, feature.symbol_radius)
                label_anchor = Point(anchor.x + dx, anchor.y + dy)
                angle = 0.0

            cand = CandidatePosition(
                position_id=pos_id,
                rank=rank,
                anchor=label_anchor,
                angle_deg=angle,
                score=max(0.0, RANK_BASE_SCORE - rank * RANK_DECAY),
            )
            candidates.append(cand)

        return candidates


# ---------------------------------------------------------------------------
# Poids de pénalité
# ---------------------------------------------------------------------------

class PenaltyWeights:
    # Conflits avec vecteurs
    OVERLAP_POINT_SYMBOL    = 40.0   # étiquette sur un symbole ponctuel
    OVERLAP_LINE_MAJOR      = 35.0   # étiquette coupe une frontière nationale ou route primaire
    OVERLAP_LINE_MINOR      = 20.0   # étiquette coupe une route secondaire/tertiaire, voie ferrée
    OVERLAP_WATER_LINE      = 15.0   # étiquette coupe un cours d'eau
    OVERLAP_POLYGON_URBAN   = 10.0   # étiquette dans une zone urbaine (pour ponctuel/sommet)
    OVERLAP_POLYGON_WATER   = 25.0   # étiquette sur un plan d'eau (pour ponctuel hors hydro)

    # Conflits avec d'autres étiquettes
    LABEL_OVERLAP_FULL      = 80.0   # chevauchement total avec une autre étiquette
    LABEL_OVERLAP_PARTIAL   = 30.0   # chevauchement partiel (> seuil)
    LABEL_OVERLAP_THRESHOLD = 0.15   # fraction de surface pour déclencher pénalité partielle

    # Bonus de rang (déjà dans base_score, mais on peut pondérer)
    RANK_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Scorer : notation des positions candidates
# ---------------------------------------------------------------------------

class ToponymScorer:
    """
    Analyse et note chaque CandidatePosition selon :
    1. Son rang dans la hiérarchie (base score)
    2. Ses intersections avec les vecteurs
    3. Ses chevauchements avec les autres étiquettes déjà placées
    """

    def __init__(self, features: list[VectorFeature],
                 weights: PenaltyWeights | None = None):
        self.features = features
        self.w = weights or PenaltyWeights()
        self._candidate_gen = CandidateGenerator()

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------

    def score_all(self) -> dict[str, list[CandidatePosition]]:
        """
        Retourne un dict {fid: [CandidatePosition trié par score décroissant]}.
        Les candidates sont notées en tenant compte de toutes les features.
        """
        # 1. Générer toutes les candidates
        all_candidates: dict[str, list[CandidatePosition]] = {}
        for feat in self.features:
            cands = self._candidate_gen.generate(feat)
            if cands:
                all_candidates[feat.fid] = cands

        # 2. Appliquer les pénalités vecteur (indépendantes entre features)
        for feat in self.features:
            if feat.fid not in all_candidates:
                continue
            for cand in all_candidates[feat.fid]:
                self._apply_vector_penalties(feat, cand)

        # 3. Trier par score décroissant (avant résolution des conflits inter-étiquettes)
        for fid in all_candidates:
            all_candidates[fid].sort(key=lambda c: c.score, reverse=True)

        # 4. Pénalités inter-étiquettes (label–label)
        self._apply_interlabel_penalties(all_candidates)

        # 5. Re-trier après pénalités label–label
        for fid in all_candidates:
            all_candidates[fid].sort(key=lambda c: c.score, reverse=True)

        return all_candidates

    # ------------------------------------------------------------------
    # Pénalités vecteur
    # ------------------------------------------------------------------

    def _label_bbox(self, feat: VectorFeature, cand: CandidatePosition) -> BBox:
        return feat.label_config.text_bbox(feat.name, cand.anchor, cand.angle_deg)

    def _apply_vector_penalties(self, feat: VectorFeature,
                                 cand: CandidatePosition) -> None:
        lbbox = self._label_bbox(feat, cand)

        for other in self.features:
            if other.fid == feat.fid:
                continue

            # ---- Vecteurs ponctuels : éviter le symbole ----
            if other.vtype in (VectorType.VILLE, VectorType.AEROPORT,
                               VectorType.PORT, VectorType.SOMMET):
                sym_bbox = BBox(
                    other.anchor_point.x - other.symbol_radius,
                    other.anchor_point.y - other.symbol_radius,
                    other.anchor_point.x + other.symbol_radius,
                    other.anchor_point.y + other.symbol_radius,
                )
                if lbbox.intersects(sym_bbox):
                    cand.penalties["overlap_point_symbol"] = (
                        cand.penalties.get("overlap_point_symbol", 0)
                        + self.w.OVERLAP_POINT_SYMBOL
                    )

            # ---- Vecteurs linéaires ----
            elif other.vtype in (VectorType.FRONTIERE_NATIONALE,
                                  VectorType.ROUTE):
                # Route primaire ou frontière nationale → pénalité forte
                is_major = (other.vtype == VectorType.FRONTIERE_NATIONALE or
                            other.classe == RouteClasse.PRIMAIRE)
                if self._label_crosses_polyline(lbbox, other.geometry_points):
                    key = "overlap_line_major" if is_major else "overlap_line_minor"
                    penalty = (self.w.OVERLAP_LINE_MAJOR if is_major
                               else self.w.OVERLAP_LINE_MINOR)
                    cand.penalties[key] = (
                        cand.penalties.get(key, 0) + penalty
                    )

            elif other.vtype in (VectorType.VOIE_FERREE,
                                  VectorType.FRONTIERE_REGIONALE):
                if self._label_crosses_polyline(lbbox, other.geometry_points):
                    cand.penalties["overlap_line_minor"] = (
                        cand.penalties.get("overlap_line_minor", 0)
                        + self.w.OVERLAP_LINE_MINOR
                    )

            elif other.vtype == VectorType.HYDRO:
                if self._label_crosses_polyline(lbbox, other.geometry_points):
                    cand.penalties["overlap_water_line"] = (
                        cand.penalties.get("overlap_water_line", 0)
                        + self.w.OVERLAP_WATER_LINE
                    )

            # ---- Vecteurs surfaciques ----
            elif other.vtype == VectorType.ZONE_URBAINE:
                if other.bbox and lbbox.intersects(other.bbox):
                    cand.penalties["overlap_polygon_urban"] = (
                        cand.penalties.get("overlap_polygon_urban", 0)
                        + self.w.OVERLAP_POLYGON_URBAN
                    )

            elif other.vtype in (VectorType.LAC, VectorType.OCEAN_MER):
                # Pénaliser si l'étiquette (non hydro) tombe sur l'eau
                if feat.vtype not in (VectorType.LAC, VectorType.OCEAN_MER,
                                       VectorType.HYDRO):
                    if other.bbox and lbbox.intersects(other.bbox):
                        cand.penalties["overlap_polygon_water"] = (
                            cand.penalties.get("overlap_polygon_water", 0)
                            + self.w.OVERLAP_POLYGON_WATER
                        )

        # Appliquer les pénalités au score
        total_penalty = sum(cand.penalties.values())
        cand.score = max(0.0, cand.base_score - total_penalty)

    # ------------------------------------------------------------------
    # Intersection étiquette / polyligne (bbox grossière + test segment)
    # ------------------------------------------------------------------

    @staticmethod
    def _label_crosses_polyline(lbbox: BBox, pts: list[Point]) -> bool:
        """Test rapide : la bbox de l'étiquette intersecte-t-elle la polyligne ?"""
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            seg_bbox = BBox(min(a.x, b.x), min(a.y, b.y),
                            max(a.x, b.x), max(a.y, b.y))
            if lbbox.intersects(seg_bbox):
                # Test fin : le segment coupe-t-il la bbox ?
                if ToponymScorer._segment_intersects_bbox(a, b, lbbox):
                    return True
        return False

    @staticmethod
    def _segment_intersects_bbox(a: Point, b: Point, bbox: BBox) -> bool:
        """Cohen-Sutherland simplifié."""
        def code(p: Point) -> int:
            c = 0
            if p.x < bbox.xmin: c |= 1
            if p.x > bbox.xmax: c |= 2
            if p.y < bbox.ymin: c |= 4
            if p.y > bbox.ymax: c |= 8
            return c

        ca, cb = code(a), code(b)
        if ca & cb:
            return False   # entièrement hors
        if not (ca | cb):
            return True    # entièrement dedans
        return True        # intersection probable (approximation conservative)

    # ------------------------------------------------------------------
    # Pénalités inter-étiquettes
    # ------------------------------------------------------------------

    def _apply_interlabel_penalties(
        self,
        all_candidates: dict[str, list[CandidatePosition]],
    ) -> None:
        """
        Pour chaque paire de features, compare la meilleure candidate de chacune
        et applique des pénalités si leurs bboxes se chevauchent.
        """
        fids = list(all_candidates.keys())
        feat_map = {f.fid: f for f in self.features}

        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                fi = feat_map[fids[i]]
                fj = feat_map[fids[j]]

                # Comparer toutes les candidates des deux features
                for ci in all_candidates[fids[i]]:
                    bbox_i = self._label_bbox(fi, ci)
                    for cj in all_candidates[fids[j]]:
                        bbox_j = self._label_bbox(fj, cj)
                        if not bbox_i.intersects(bbox_j):
                            continue

                        overlap = self._overlap_fraction(bbox_i, bbox_j)
                        if overlap >= 1.0 - 1e-6:
                            pen = self.w.LABEL_OVERLAP_FULL
                            key = "label_overlap_full"
                        elif overlap > self.w.LABEL_OVERLAP_THRESHOLD:
                            pen = self.w.LABEL_OVERLAP_PARTIAL
                            key = "label_overlap_partial"
                        else:
                            continue

                        ci.penalties[key] = ci.penalties.get(key, 0) + pen
                        cj.penalties[key] = cj.penalties.get(key, 0) + pen
                        ci.score = max(0.0, ci.score - pen)
                        cj.score = max(0.0, cj.score - pen)

    @staticmethod
    def _overlap_fraction(a: BBox, b: BBox) -> float:
        """Fraction de chevauchement : aire_intersection / min(aire_a, aire_b)."""
        ix = max(0.0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
        iy = max(0.0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
        inter = ix * iy
        area_a = a.width * a.height
        area_b = b.width * b.height
        denom = min(area_a, area_b)
        return inter / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Résolveur glouton : sélectionne la meilleure candidate par feature
# ---------------------------------------------------------------------------

class GreedyLabelResolver:
    """
    Parcourt les features par ordre de priorité (classe d'importance)
    et sélectionne la meilleure candidate non conflictuelle.
    """

    # Ordre de priorité : on place d'abord les labels les plus importants
    PRIORITY_ORDER = [
        (VectorType.VILLE,    VilleClasse.CAPITALE),
        (VectorType.PAYS,     None),
        (VectorType.OCEAN_MER,None),
        (VectorType.VILLE,    VilleClasse.CHEF_LIEU),
        (VectorType.REGION,   None),
        (VectorType.MONTAGNE, None),
        (VectorType.VILLE,    VilleClasse.VILLE),
        (VectorType.PORT,     None),
        (VectorType.SOMMET,   None),
        (VectorType.AEROPORT, None),
        (VectorType.HYDRO,    None),
        (VectorType.LAC,      None),
        (VectorType.ROUTE,    RouteClasse.PRIMAIRE),
        (VectorType.VOIE_FERREE, None),
        (VectorType.FRONTIERE_NATIONALE, None),
        (VectorType.FRONTIERE_REGIONALE, None),
        (VectorType.RESERVE,  None),
    ]

    def resolve(
        self,
        features: list[VectorFeature],
        scored: dict[str, list[CandidatePosition]],
    ) -> dict[str, CandidatePosition]:
        """
        Retourne {fid: meilleure_candidate_placée}.
        """
        feat_map = {f.fid: f for f in features}
        placed_bboxes: list[tuple[str, BBox]] = []
        result: dict[str, CandidatePosition] = {}

        def _priority_key(feat: VectorFeature) -> int:
            for idx, (vt, cl) in enumerate(self.PRIORITY_ORDER):
                if feat.vtype == vt and (cl is None or feat.classe == cl):
                    return idx
            return len(self.PRIORITY_ORDER)

        ordered = sorted(features, key=_priority_key)

        for feat in ordered:
            if feat.fid not in scored:
                continue
            candidates = scored[feat.fid]  # déjà triées par score décroissant

            chosen = None
            for cand in candidates:
                lbbox = feat.label_config.text_bbox(
                    feat.name, cand.anchor, cand.angle_deg
                )
                conflict = any(
                    lbbox.intersects(pb)
                    for _, pb in placed_bboxes
                )
                if not conflict:
                    chosen = cand
                    placed_bboxes.append((feat.fid, lbbox))
                    break

            # Si aucune candidate sans conflit, prendre la meilleure quand même
            if chosen is None and candidates:
                chosen = candidates[0]
                lbbox = feat.label_config.text_bbox(
                    feat.name, chosen.anchor, chosen.angle_deg
                )
                placed_bboxes.append((feat.fid, lbbox))

            if chosen:
                result[feat.fid] = chosen

        return result


# ---------------------------------------------------------------------------
# Rapport de placement
# ---------------------------------------------------------------------------

def placement_report(
    features: list[VectorFeature],
    scored: dict[str, list[CandidatePosition]],
    placed: dict[str, CandidatePosition],
) -> str:
    feat_map = {f.fid: f for f in features}
    lines = []
    lines.append("=" * 72)
    lines.append("RAPPORT DE PLACEMENT DES TOPONYMES")
    lines.append("=" * 72)

    for fid, cand in placed.items():
        feat = feat_map[fid]
        all_cands = scored.get(fid, [])
        lines.append(f"\n[{feat.fid}] {feat.name}  ({feat.vtype.value}"
                     f"{' / ' + str(feat.classe.value) if feat.classe else ''})")
        lines.append(f"  Position choisie : {cand.position_id.name}"
                     f"  angle={cand.angle_deg:.1f}°"
                     f"  ancre=({cand.anchor.x:.1f}, {cand.anchor.y:.1f})")
        lines.append(f"  Score final      : {cand.score:.1f}"
                     f"  (base={cand.base_score:.1f})")
        if cand.penalties:
            for k, v in cand.penalties.items():
                lines.append(f"    pénalité {k:<30s} : -{v:.1f}")
        lines.append(f"  Nombre de candidates évaluées : {len(all_cands)}")
        # Top 3 alternatives
        alts = [c for c in all_cands if c is not cand][:3]
        if alts:
            lines.append("  Alternatives (top 3) :")
            for a in alts:
                lines.append(f"    {a.position_id.name:<8s} score={a.score:.1f}"
                             f"  pénalités={sum(a.penalties.values()):.1f}")

    lines.append("\n" + "=" * 72)
    lines.append(f"Total labellisé : {len(placed)} / {len(features)} features")
    return "\n".join(lines)
