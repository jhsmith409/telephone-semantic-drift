# Copyright (c) 2026 James H. Smith. MIT License.
"""Construction-domain prompts for Paper 3, shaped like TeraContext SOW output
(trade statement of work assembled from CSI MasterFormat spec sections).

Two complexity levels mirroring spaghetti (short summary SOW) and lasagna (detailed
SOW). Quantities are internally consistent (volumes computed from stated dimensions) so
drift can be scored by exact numeric preservation and by exclusion-list integrity, in
addition to cosine similarity. Motivation: TeraContext's SOW pipeline is sequential
(extract -> classify -> assemble -> summarize) and has shown runaway exclusion-list
degeneration in production (docs/PROBLEM_exclusion_degeneration.md, 2026-06-12).
"""

CONCRETE_PROMPTS = {
    "concrete_short": (
        "Summary Statement of Work, Trade: Concrete, Section 03 30 00 Cast-in-Place "
        "Concrete. Scope: 1. Furnish and place 185 CY of 4,000 psi normal-weight concrete "
        "for spread footings and grade beams, 4 in slump, 6 percent air. 2. Furnish and "
        "install 9.2 tons of ASTM A615 Grade 60 reinforcing steel with 3 in cover at "
        "earth-formed surfaces. 3. Provide edge formwork, 7-day moist curing, and one set "
        "of four ASTM C31 test cylinders per 50 CY placed, broken per ASTM C39 at 7 and "
        "28 days. Major exclusions: excavation, backfill, and dewatering by the Sitework "
        "contractor."
    ),
    "concrete_long": (
        "Statement of Work, Trade: Concrete (Cast-in-Place), Building B Warehouse "
        "Expansion. Spec sections used: 03 20 00 Concrete Reinforcing, 03 30 00 "
        "Cast-in-Place Concrete, 03 35 00 Concrete Finishing, 07 26 00 Vapor Retarders. "
        "The Concrete subcontractor shall furnish all labor, materials, equipment, and "
        "supervision for the following scope. "
        "Foundations: "
        "1. Spread footings, 24 each, 6 ft x 6 ft x 18 in, 4,000 psi concrete, 48 CY total. "
        "2. Continuous wall footings, 620 LF, 2 ft wide x 12 in deep, 4,000 psi, 46 CY. "
        "3. Foundation walls, 620 LF, 8 in thick x 4 ft high, 4,000 psi, 61 CY, formed both faces. "
        "4. Loading dock pits, 4 each, 10 ft x 8 ft x 4 ft deep with 8 in walls and slab, "
        "5,000 psi, 22 CY, with embedded steel angles at all edges. "
        "5. Anchor bolts: install 96 each 1 in diameter x 24 in ASTM F1554 Grade 36 anchor "
        "bolts per templates furnished by the Steel Erector. "
        "Slabs: "
        "6. Slab-on-grade, 42,000 SF, 6 in thick, 4,500 psi, 778 CY, placed over 15-mil "
        "vapor retarder per ASTM E1745 Class A. "
        "7. Equipment pads, 6 each, 4 ft x 6 ft x 6 in, 4,000 psi, 3 CY, with 3/4 in chamfered edges. "
        "8. Sawcut control joints at 15 ft maximum spacing each way, 1/4 of slab depth, cut "
        "within 12 hours of finishing. "
        "9. Slab finish: hard steel trowel, floor flatness FF 35 and floor levelness FL 25 per "
        "ASTM E1155, tested within 72 hours of placement. "
        "Reinforcing: "
        "10. Slab reinforcing: #4 bars at 18 in on center each way, ASTM A615 Grade 60, on "
        "chairs at 3 ft on center. "
        "11. Footing and wall reinforcing: 14.5 tons ASTM A615 Grade 60, 3 in cover against "
        "earth, 2 in cover at formed surfaces; furnish mill certificates. "
        "Materials and quality: "
        "12. Mix designs: maximum water-cement ratio 0.45, 4 in slump plus or minus 1 in, air "
        "entrainment 6 percent plus or minus 1.5 percent for exterior concrete, Type II cement, "
        "25 percent Class F fly ash replacement permitted. "
        "13. Curing: dissipating-resin curing compound at 200 SF per gallon within 30 minutes "
        "of final finishing; moist-cure footings and walls for 7 days. "
        "14. Cold weather: maintain concrete at 55 degrees F minimum for 3 days after placement "
        "when ambient temperature is below 40 degrees F, per ACI 306. "
        "15. Tolerances per ACI 117: footings plus 2 in or minus 1/2 in in plan; walls plus or "
        "minus 1/4 in in 10 ft. "
        "16. Testing: one set of four cylinders per 100 CY or fraction thereof per day per mix, "
        "ASTM C31 field cured, ASTM C39 breaks at 7 and 28 days, slump and air per ASTM C143 "
        "and ASTM C231 at each set. "
        "17. Submittals within 14 calendar days of award: concrete mix designs, reinforcing "
        "shop drawings, vapor retarder and curing compound product data, mill certificates. "
        "Schedule: "
        "18. Footings complete within 15 working days of mobilization; slab-on-grade placed in "
        "6 pours of approximately 7,000 SF each; 28-day design strength required before steel "
        "erection loads. "
        "Exclusions (by others): "
        "19. Excavation, backfill, dewatering, and compacted subgrade by the Sitework contractor. "
        "20. The 4 in granular capillary break under slabs by the Sitework contractor. "
        "21. Testing agency fees paid by the Owner; anchor bolt templates furnished by the "
        "Steel Erector; dumpster for concrete debris provided by the General Contractor. "
        "Alternates: Alternate 1: substitute 6 x 6 W2.9 x W2.9 welded wire reinforcement for "
        "the #4 slab bars, deduct $18,400. "
        "Base bid: $1,284,600 including a 5 percent contingency."
    ),
}

# Papers 1-2 system prompts whose wording is domain-neutral. The others say "recipe",
# "cook", or "three steps" and would confound a cross-domain comparison.
DOMAIN_NEUTRAL_SYSPROMPT_IDS = [1, 2, 4, 10, 17, 20]
