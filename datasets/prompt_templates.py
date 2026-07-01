"""
WinCLIP-style Compositional Prompt Ensemble (CPE) + detailed defect descriptions.

References:
  - WinCLIP (CVPR 2023): compositional prompt ensemble for anomaly detection
  - AnomalyGPT (2024): LLM-generated fine-grained anomaly descriptions
  - FiLo (2024): adaptive text templates with defect localization

Design:
  1. Template pool: multiple sentence structures (WinCLIP-style CPE)
  2. Defect descriptions: natural-language, visually descriptive (AnomalyGPT-style)
  3. During training: randomly sample one template + fill in the description
  4. 10% of samples use null prompt "" for CFG training

Usage (conda env):
    conda activate omg
"""

import random
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# WinCLIP-style template pool
# ---------------------------------------------------------------------------
# Each template contains {} placeholders for category/defect insertion.
# Templates are randomly sampled at each __getitem__ call.

PROMPT_TEMPLATES = [
    # WinCLIP originals (visual inspection focus)
    "photo of a {} for visual inspection",
    "cropped image showing a {}",
    "close-up view of a {}",
    "high-resolution industrial photo of a {}",
    # Extended: quality-control oriented
    "a {} captured during automated quality inspection",
    "a {} under bright factory lighting",
    "detailed inspection image of a {}",
    "a {} on a production line conveyor",
    "zoomed-in view of a {}",
    "a {} with uniform illumination for defect detection",
    "industrial camera snapshot of a {}",
    # Short variants
    "a {}",
    "image of a {}",
    "photo of a {}",
]

# Category display names (natural language, no underscores)
CATEGORY_NAMES = {
    "bottle":      "glass bottle",
    "cable":       "electrical cable",
    "capsule":     "pharmaceutical capsule",
    "carpet":      "textile carpet",
    "grid":        "metal grid",
    "hazelnut":    "hazelnut",
    "leather":     "leather surface",
    "metal_nut":   "metal nut",
    "pill":        "pharmaceutical pill",
    "screw":       "metal screw",
    "tile":        "ceramic tile",
    "toothbrush":  "toothbrush",
    "transistor":  "electronic transistor",
    "wood":        "wooden surface",
    "zipper":      "fabric zipper",
}

# ---------------------------------------------------------------------------
# AnomalyGPT-style defect descriptions per (category, defect_type)
# ---------------------------------------------------------------------------
# Each defect is described with visually grounded natural language.
# These descriptions are inserted into templates in place of the bare defect_type.

DEFECT_DESCRIPTIONS = {
    "bottle": {
        "good":            "flawless glass bottle with a smooth transparent surface and intact body",
        "broken_large":    "glass bottle with a large shattered opening, sharp irregular fracture edges, and missing glass fragments",
        "broken_small":    "glass bottle with a small chip or crack near the rim, minor glass splintering visible",
        "contamination":   "glass bottle with dark particulate debris and floating impurities inside the transparent body",
    },
    "cable": {
        "good":                     "pristine electrical cable with straight wires and intact insulation",
        "bent_wire":                "electrical cable with a sharply bent or kinked wire disrupting the straight conductor path",
        "cable_swap":               "electrical cable where two wires are swapped in position compared to the correct layout",
        "combined":                 "electrical cable showing multiple defects simultaneously including bent wires and damaged insulation",
        "cut_inner_insulation":     "electrical cable with the inner insulation layer cut open exposing the conductor beneath",
        "cut_outer_insulation":     "electrical cable with the outer protective sheath cut or sliced revealing inner layers",
        "missing_cable":            "electrical cable assembly where one or more cable strands are completely absent",
        "missing_wire":             "electrical cable where an individual wire conductor is missing from the bundle",
        "poke_insulation":          "electrical cable with a punctured hole in the insulation, as if poked by a sharp object",
    },
    "capsule": {
        "good":            "flawless pharmaceutical capsule with intact shell and clean print markings",
        "crack":           "pharmaceutical capsule with a visible crack or fissure along its shell surface",
        "faulty_imprint":  "pharmaceutical capsule with smeared, misaligned, or partially missing printed text",
        "poke":            "pharmaceutical capsule with a small puncture hole in the shell",
        "scratch":         "pharmaceutical capsule with linear scratch marks across its colored coating",
        "squeeze":         "pharmaceutical capsule that is deformed, crushed, or dented from compression",
    },
    "carpet": {
        "good":                 "pristine carpet surface with uniform texture and consistent fiber alignment",
        "color":                "carpet with discolored patches or uneven dye distribution across the fibers",
        "cut":                  "carpet with a clean linear cut or slice through the fiber layer",
        "hole":                 "carpet with a punched hole or void where fibers are missing",
        "metal_contamination":  "carpet with small metallic fragments, shavings, or debris embedded in the fibers",
        "thread":               "carpet with a loose, unraveled, or protruding thread on the surface",
    },
    "grid": {
        "good":                 "pristine metal grid with regular, evenly spaced bars and clean edges",
        "bent":                 "metal grid with deformed or bent bars deviating from the regular pattern",
        "broken":               "metal grid with a fractured or snapped bar creating a gap in the structure",
        "glue":                 "metal grid with excess adhesive residue or glue stains on the bars",
        "metal_contamination":  "metal grid with foreign metallic particles or debris stuck between the bars",
        "thread":               "metal grid with a fibrous thread tangled or wrapped around the bars",
    },
    "hazelnut": {
        "good":   "flawless whole hazelnut with smooth brown shell and intact surface",
        "crack":  "hazelnut with a visible crack or fissure splitting the outer shell",
        "cut":    "hazelnut with a clean incision or knife cut through the shell",
        "hole":   "hazelnut with a circular hole bored into the shell surface",
        "print":  "hazelnut with foreign printed markings or ink stains on the shell",
    },
    "leather": {
        "good":  "pristine leather surface with uniform texture, consistent color, and no visible defects",
        "color": "leather surface with uneven or patchy discoloration and color variation",
        "cut":   "leather surface with a sharp linear cut or slice penetrating the material",
        "fold":  "leather surface with a permanent crease or fold line deforming the material",
        "glue":  "leather surface with sticky adhesive residue or glue spots contaminating the finish",
        "poke":  "leather surface with a small puncture hole from a sharp object",
    },
    "metal_nut": {
        "good":    "flawless metal nut with clean hexagonal shape, smooth threads, and uniform surface finish",
        "bent":    "metal nut with deformed or warped edges distorting its hexagonal shape",
        "color":   "metal nut with surface discoloration, tarnish, or uneven coating color",
        "cut":     "metal nut with a cut or gouge on its surface from mechanical damage",
        "flip":    "metal nut that is flipped upside down showing the incorrect orientation",
        "scratch": "metal nut with linear scratch marks on its metallic surface",
    },
    "pill": {
        "good":           "flawless pharmaceutical pill with smooth coating, uniform color, and clear imprint",
        "color":          "pharmaceutical pill with discolored spots, uneven coating, or faded coloring",
        "combined":       "pharmaceutical pill showing multiple defect types simultaneously",
        "contamination":  "pharmaceutical pill with foreign particles, specks, or debris embedded in the coating",
        "crack":          "pharmaceutical pill with a visible crack or fracture across its surface",
        "faulty_imprint": "pharmaceutical pill with smeared, double-struck, or illegible printed text",
        "pill_type":      "pharmaceutical pill of the wrong shape, size, or color mixed in with correct pills",
        "scratch":        "pharmaceutical pill with linear abrasion marks on the coating surface",
    },
    "screw": {
        "good":              "flawless metal screw with clean threads, intact head, and uniform surface",
        "manipulated_front": "metal screw with a damaged or stripped front section of the head",
        "scratch_head":      "metal screw with scratch marks on the screw head surface",
        "scratch_neck":      "metal screw with scratch marks on the smooth neck section below the head",
        "thread_side":       "metal screw with damaged or flattened threads on the side of the shaft",
        "thread_top":        "metal screw with damaged or stripped threads near the top of the shaft",
    },
    "tile": {
        "good":        "pristine ceramic tile with uniform color, smooth surface, and clean edges",
        "crack":       "ceramic tile with a visible crack line running across the surface",
        "glue_strip":  "ceramic tile with an adhesive strip or glue residue on the surface",
        "gray_stroke": "ceramic tile with a gray paint stroke or smear mark across the surface",
        "oil":         "ceramic tile with oil stains or greasy residue contaminating the surface",
        "rough":       "ceramic tile with a rough, uneven, or abraded surface texture",
    },
    "toothbrush": {
        "good":      "flawless toothbrush with straight bristles, clean handle, and intact body",
        "defective": "toothbrush with bent, missing, or deformed bristles, or a damaged handle",
    },
    "transistor": {
        "good":         "flawless electronic transistor with straight leads, intact casing, and clean markings",
        "bent_lead":    "electronic transistor with a bent or deformed metal lead leg",
        "cut_lead":     "electronic transistor with a cut or truncated metal lead that is too short",
        "damaged_case": "electronic transistor with a cracked, chipped, or deformed plastic casing",
        "misplaced":    "electronic transistor that is shifted, rotated, or incorrectly positioned",
    },
    "wood": {
        "good":     "pristine wooden surface with natural grain, uniform color, and no visible flaws",
        "color":    "wooden surface with discolored patches or unnatural color variation",
        "combined": "wooden surface showing multiple defect types at once",
        "hole":     "wooden surface with a punched, drilled, or bored hole",
        "liquid":   "wooden surface with liquid stains or water marks",
        "scratch":  "wooden surface with linear scratch marks across the grain",
    },
    "zipper": {
        "good":            "flawless fabric zipper with aligned teeth and smooth operation",
        "broken_teeth":    "fabric zipper with broken, chipped, or missing teeth along the zipper chain",
        "combined":        "fabric zipper showing multiple defect types at once",
        "fabric_border":   "fabric zipper with frayed, torn, or unraveling fabric at the border",
        "fabric_interior": "fabric zipper with torn or cut fabric in the interior panel between the teeth",
        "rough":           "fabric zipper with rough, jagged, or uneven teeth edges",
        "split_teeth":     "fabric zipper where the teeth have split apart creating a gap in the chain",
        "squeezed_teeth":  "fabric zipper with compressed, flattened, or crushed teeth sections",
    },
}

# ---------------------------------------------------------------------------
# CFG null prompt
# ---------------------------------------------------------------------------
NULL_PROMPT = ""  # CLIP encodes "" as [SOS, EOS, PAD, ..., PAD]

# Vowel-starting words for a/an selection
_VOWEL_PATTERNS = ("a ", "e ", "i ", "o ", "u ", "el", "un", "ir", "ov")


def _fix_article(template: str) -> str:
    """
    Replace ' a {}' with ' an {}' when the description starts with a vowel sound.
    Works by checking the description that will fill the template.
    This is handled in build_prompt() below.
    """
    return template


def build_prompt(
    category: str,
    defect_type: str,
    null_prompt_prob: float = 0.0,
    vlm_description: Optional[str] = None,
) -> str:
    """
    Build a prompt for the given category and defect type.

    Args:
        category:        MVTec category name (e.g. "bottle").
        defect_type:     Defect type (e.g. "broken_large", "good").
        null_prompt_prob: Probability (0-1) of returning "" for CFG training.
        vlm_description: If provided, use this VLM-generated description directly
                        (it's already a self-contained rich sentence — no template
                        wrapping). Falls back to DEFECT_DESCRIPTIONS if None/empty.

    Returns:
        str: Generated prompt, or "" if null_prompt_prob triggers.
    """
    # CFG null prompt
    if random.random() < null_prompt_prob:
        return NULL_PROMPT

    cat_name = CATEGORY_NAMES.get(category, category)

    # ---- Resolve description ----
    if vlm_description and vlm_description.strip():
        # VLM 描述已是完整段落，直接作为 prompt（不套模板）
        return vlm_description.strip()

    # Fallback: hand-written DEFECT_DESCRIPTIONS + template wrapping
    defect_info = DEFECT_DESCRIPTIONS.get(category, {})
    description = defect_info.get(defect_type)

    if description is None:
        if defect_type == "good":
            description = f"defect-free {cat_name} with no visible flaws"
        else:
            defect_name = defect_type.replace("_", " ")
            description = f"a {cat_name} with a {defect_name} defect"

    template = random.choice(PROMPT_TEMPLATES)

    if description and description[0].lower() in "aeiou":
        template = template.replace(" a {}", " an {}")
        if template.startswith("a {}"):
            template = "an {}" + template[4:]

    prompt = template.format(description)
    return prompt


def get_all_prompts_for_pair(category: str, defect_type: str) -> list:
    """
    Return all possible prompts for a (category, defect_type) pair.
    Used for evaluation or exhaustive prompting.
    """
    cat_name = CATEGORY_NAMES.get(category, category)
    defect_info = DEFECT_DESCRIPTIONS.get(category, {})
    description = defect_info.get(defect_type)

    if description is None:
        if defect_type == "good":
            description = f"defect-free {cat_name} with no visible flaws"
        else:
            defect_name = defect_type.replace("_", " ")
            description = f"a {cat_name} with a {defect_name} defect"

    return [t.format(description) for t in PROMPT_TEMPLATES]


def list_all_defect_types() -> dict:
    """Return {category: [defect_types]} for all categories."""
    return {cat: list(defects.keys()) for cat, defects in DEFECT_DESCRIPTIONS.items()}
