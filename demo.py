"""
Démonstration du moteur de placement de toponymes.
Crée un jeu de données synthétique réaliste et affiche le rapport complet.
"""

from toponym_engine import (
    VectorFeature, VectorType, VilleClasse, AeroportClasse, RouteClasse,
    Point, CandidateGenerator, ToponymScorer, GreedyLabelResolver,
    placement_report, needs_label,
)


def build_test_dataset() -> list[VectorFeature]:
    """Jeu de données synthétique couvrant tous les types de vecteurs."""
    features = []

    # ── Surfaciques ────────────────────────────────────────────────────────────
    features.append(VectorFeature(
        fid="PAY_001", vtype=VectorType.PAYS, name="FRANCE",
        geometry_points=[
            Point(0, 0), Point(200, 0), Point(200, 180), Point(0, 180), Point(0, 0)
        ],
    ))
    features.append(VectorFeature(
        fid="REG_001", vtype=VectorType.REGION, name="Île-de-France",
        geometry_points=[
            Point(85, 80), Point(125, 80), Point(125, 120), Point(85, 120), Point(85, 80)
        ],
    ))
    features.append(VectorFeature(
        fid="OCE_001", vtype=VectorType.OCEAN_MER, name="Mer Méditerranée",
        geometry_points=[
            Point(30, 0), Point(200, 0), Point(200, 40), Point(30, 40), Point(30, 0)
        ],
    ))
    features.append(VectorFeature(
        fid="LAC_001", vtype=VectorType.LAC, name="Lac Léman",
        geometry_points=[
            Point(150, 70), Point(170, 70), Point(170, 80), Point(150, 80), Point(150, 70)
        ],
    ))
    features.append(VectorFeature(
        fid="RES_001", vtype=VectorType.RESERVE, name="Parc National des Cévennes",
        geometry_points=[
            Point(100, 50), Point(130, 50), Point(130, 70), Point(100, 70), Point(100, 50)
        ],
    ))
    features.append(VectorFeature(
        fid="URB_001", vtype=VectorType.ZONE_URBAINE, name="Zone urbaine Paris",
        geometry_points=[
            Point(96, 96), Point(110, 96), Point(110, 106), Point(96, 106), Point(96, 96)
        ],
    ))
    features.append(VectorFeature(
        fid="MTG_001", vtype=VectorType.MONTAGNE, name="Alpes",
        geometry_points=[
            Point(140, 60), Point(200, 60), Point(200, 100), Point(140, 100), Point(140, 60)
        ],
    ))

    # ── Linéaires ──────────────────────────────────────────────────────────────
    features.append(VectorFeature(
        fid="FRN_001", vtype=VectorType.FRONTIERE_NATIONALE, name="Frontière franco-espagnole",
        geometry_points=[Point(0, 20), Point(50, 22), Point(100, 18), Point(150, 21), Point(200, 20)],
    ))
    features.append(VectorFeature(
        fid="FRR_001", vtype=VectorType.FRONTIERE_REGIONALE, name="Limite Île-de-France / Centre",
        geometry_points=[Point(85, 80), Point(125, 80)],
    ))
    features.append(VectorFeature(
        fid="RTE_001", vtype=VectorType.ROUTE, name="A6 - Autoroute du Soleil",
        geometry_points=[Point(103, 100), Point(105, 85), Point(108, 70), Point(112, 55), Point(115, 40)],
        classe=RouteClasse.PRIMAIRE,
    ))
    features.append(VectorFeature(
        fid="RTE_002", vtype=VectorType.ROUTE, name="D907",
        geometry_points=[Point(80, 90), Point(95, 92), Point(105, 95)],
        classe=RouteClasse.SECONDAIRE,
    ))
    features.append(VectorFeature(
        fid="RTE_003", vtype=VectorType.ROUTE, name="Chemin des Crêtes",
        geometry_points=[Point(145, 65), Point(155, 68), Point(162, 72)],
        classe=RouteClasse.TERTIAIRE,
    ))
    features.append(VectorFeature(
        fid="FER_001", vtype=VectorType.VOIE_FERREE, name="LGV Sud-Est",
        geometry_points=[Point(103, 98), Point(106, 82), Point(110, 65), Point(114, 50)],
    ))
    features.append(VectorFeature(
        fid="HYD_001", vtype=VectorType.HYDRO, name="Rhône",
        geometry_points=[Point(130, 100), Point(128, 80), Point(125, 60), Point(122, 40), Point(118, 25)],
    ))
    features.append(VectorFeature(
        fid="HYD_002", vtype=VectorType.HYDRO, name="Seine",
        geometry_points=[Point(70, 105), Point(85, 102), Point(103, 100), Point(120, 103)],
    ))

    # ── Ponctuels ──────────────────────────────────────────────────────────────
    features.append(VectorFeature(
        fid="VIL_001", vtype=VectorType.VILLE, name="Paris",
        geometry_points=[Point(103, 100)],
        classe=VilleClasse.CAPITALE,
        symbol_radius=8.0,
    ))
    features.append(VectorFeature(
        fid="VIL_002", vtype=VectorType.VILLE, name="Lyon",
        geometry_points=[Point(120, 65)],
        classe=VilleClasse.CHEF_LIEU,
        symbol_radius=6.0,
    ))
    features.append(VectorFeature(
        fid="VIL_003", vtype=VectorType.VILLE, name="Marseille",
        geometry_points=[Point(115, 30)],
        classe=VilleClasse.CHEF_LIEU,
        symbol_radius=6.0,
    ))
    features.append(VectorFeature(
        fid="VIL_004", vtype=VectorType.VILLE, name="Grenoble",
        geometry_points=[Point(142, 70)],
        classe=VilleClasse.VILLE,
        symbol_radius=4.0,
    ))
    features.append(VectorFeature(
        fid="VIL_005", vtype=VectorType.VILLE, name="Chambéry",
        geometry_points=[Point(148, 75)],
        classe=VilleClasse.VILLE,
        symbol_radius=4.0,
    ))
    features.append(VectorFeature(
        fid="AER_001", vtype=VectorType.AEROPORT, name="CDG",
        geometry_points=[Point(108, 107)],
        classe=AeroportClasse.CIVIL,
        symbol_radius=5.0,
    ))
    features.append(VectorFeature(
        fid="AER_002", vtype=VectorType.AEROPORT, name="Istres-Le Tubé",
        geometry_points=[Point(105, 35)],
        classe=AeroportClasse.MILITAIRE,
        symbol_radius=5.0,
    ))
    features.append(VectorFeature(
        fid="AER_003", vtype=VectorType.AEROPORT, name="Lyon-Saint Exupéry",
        geometry_points=[Point(125, 63)],
        classe=AeroportClasse.CIVILO_MILITAIRE,
        symbol_radius=5.0,
    ))
    features.append(VectorFeature(
        fid="POR_001", vtype=VectorType.PORT, name="Port de Marseille",
        geometry_points=[Point(116, 28)],
        symbol_radius=5.0,
    ))
    features.append(VectorFeature(
        fid="SOM_001", vtype=VectorType.SOMMET, name="Mont Blanc",
        geometry_points=[Point(162, 82)],
        symbol_radius=4.0,
    ))
    features.append(VectorFeature(
        fid="SOM_002", vtype=VectorType.SOMMET, name="Pic du Midi",
        geometry_points=[Point(75, 22)],
        symbol_radius=4.0,
    ))

    return features


def main():
    print("Initialisation du jeu de données...")
    features = build_test_dataset()

    print(f"  {len(features)} features chargées")
    labeled = [f for f in features if needs_label(f.vtype, f.classe)]
    print(f"  {len(labeled)} features labellisées\n")

    # ── Génération des candidates ──────────────────────────────────────────────
    print("Génération des positions candidates...")
    gen = CandidateGenerator()
    total_candidates = 0
    cand_summary = {}
    for feat in labeled:
        cands = gen.generate(feat)
        cand_summary[feat.fid] = len(cands)
        total_candidates += len(cands)
        print(f"  [{feat.fid}] {feat.name:<35s} → {len(cands)} candidates "
              f"({', '.join(c.position_id.name for c in cands)})")

    print(f"\n  Total candidates générées : {total_candidates}\n")

    # ── Scoring ────────────────────────────────────────────────────────────────
    print("Notation des positions candidates...")
    scorer = ToponymScorer(features)
    scored = scorer.score_all()
    print("  Scoring terminé.\n")

    # ── Résolution gloutonne ───────────────────────────────────────────────────
    print("Résolution des conflits (algorithme glouton)...")
    resolver = GreedyLabelResolver()
    placed = resolver.resolve(features, scored)
    print(f"  {len(placed)} labels placés.\n")

    # ── Rapport ───────────────────────────────────────────────────────────────
    report = placement_report(features, scored, placed)
    print(report)

    # ── Résumé des scores ──────────────────────────────────────────────────────
    print("\nRÉSUMÉ DES SCORES PAR TYPE DE VECTEUR")
    print("-" * 50)
    feat_map = {f.fid: f for f in features}
    by_type: dict[str, list[float]] = {}
    for fid, cand in placed.items():
        vtype = feat_map[fid].vtype.value
        by_type.setdefault(vtype, []).append(cand.score)

    for vtype, scores in sorted(by_type.items()):
        avg = sum(scores) / len(scores)
        print(f"  {vtype:<25s} n={len(scores)}  "
              f"moy={avg:.1f}  min={min(scores):.1f}  max={max(scores):.1f}")


if __name__ == "__main__":
    main()
