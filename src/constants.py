ALLOYING_ELEMENTS = [
    "Ag",
    "Al",
    "B",
    "Be",
    "Bi",
    "Cd",
    "Co",
    "Cr",
    "Cu",
    "Er",
    "Eu",
    "Fe",
    "Ga",
    "Li",
    "Mg",
    "Mn",
    "Ni",
    "Sc",
    "Si",
    "Sn",
    "Ti",
    "V",
    "Zn",
    "Zr",
]


PROCESSING_COLUMNS = ["Processing"]


FEATURE_COLUMNS = PROCESSING_COLUMNS + ALLOYING_ELEMENTS

TARGET_COLUMNS = ["Yield Strength (MPa)", "Tensile Strength (MPa)", "Elongation (%)"]

IMAGE_PARAMETERS = dict(
    height=500,
    width=500,
    font_family="Arial",
    font_size=20,
    font_color="dark gray",
    showlegend=True,
    yaxis=dict(
        gridcolor="#d3d3d3", linecolor="lightslategray", zerolinecolor="#d3d3d3"
    ),
    xaxis=dict(
        gridcolor="#d3d3d3",
        linecolor="lightslategray",
    ),
    plot_bgcolor="white",
)


NAME_REPLACE = {
    "Processing_Artificial aged": "Artificial \n aged",
    "Processing_Solutionised": "Solutionised",
    "Processing_Solutionised + Artificially over aged": "Solutionised  \n+ Artificially \nover aged",
    "Processing_Solutionised + Naturally aged": "Solutionised \n+ Naturally \naged",
    "Processing_Solutionised  + Artificially peak aged": "Solutionised  \n+ Artificially \npeak aged",
    "Processing_No Processing": "No Processing",
    "Processing_Naturally aged": "Nationally \n Aged",
    "Processing_Strain Harderned (Hard)": "Strain \n Harderned (Hard)",
    "Processing_Strain hardened": "Strain \n Harderned",
    "Processing_Solutionised + Cold Worked + Naturally aged": "Solutionised \n + Cold Worked +\n Naturally aged",
}

N_JOBS = 8

TICK_FONT_SIZE = 15
FONT_SIZE = 20

RANGE_DICTIONARY = {
    "Yield Strength (MPa)": {
        "learning_curve": [0, 100],
        "feature_importance": [0, 0.45],
        "unit": "MPa"
    },
    "Tensile Strength (MPa)": {
        "learning_curve": [0, 100],
        "feature_importance": [0, 0.65],
        "unit": "MPa"
    },
    "Elongation (%)": {
        "learning_curve": [0, 8],
        "feature_importance": [0, 0.4],
        "unit": "%"
    },
}

N_ITER = 100

PRECISION = 2
