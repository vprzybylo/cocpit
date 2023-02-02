from datetime import date

campaigns = [
    # "AIRS_II",
    "ARM",
    # "ATTREX",
    "CRYSTAL_FACE_UND",
    "CRYSTAL_FACE_NASA",
    # "ICE_L",
    # "IPHEX",
    # "ISDAC",
    # "MACPEX",
    # "MC3E",
    "MIDCIX",
    "MPACE",
    # "POSIDON",
    "OLYMPEX",
]

campaigns_rename = [
    # "AIRS II",
    "ARM",
    # "ATTREX",
    "CRYSTAL FACE (UND)",
    "CRYSTAL FACE (NASA)",
    # "ICE L",
    # "IPHEX",
    # "ISDAC",
    # "MACPEX",
    # "MC3E",
    "MIDCIX",
    "MPACE",
    # "POSIDON",
    "OLYMPEX",
]

particle_types = [
    "agg",
    "budding",
    "bullet",
    "column",
    "compact_irreg",
    "planar_polycrystal",
    "rimed",
]

particle_types_rename = [
    "aggregate",
    "budding rosette",
    "bullet rosette",
    "column",
    "compact irregular",
    "planar polycrystal",
    "rimed",
]

env_properties = ["Ice Water Content", "Temperature", "Pressure"]

particle_properties = [
    "Contour Area",
    "Circularity",
    "Solidity",
    "Complexity",
    "Equivalent Diameter",
    "Convex Perimeter",
    "Hull Area",
    "Perimeter",
    "Aspect Ratio",
    "Area Ratio",
    "Roundness",
    "Perimeter-Area Ratio",
]

vertical_vars = ["Ice Water Content", "Temperature", "Pressure"]

campaign_start_dates = {
    "AIRS II": date(2003, 11, 14),
    "ARM": date(2000, 3, 9),
    "ATTREX": date(2014, 3, 4),
    "CRYSTAL FACE (UND)": date(2002, 7, 13),
    "CRYSTAL FACE (NASA)": date(2002, 7, 13),
    "ICE L": date(2007, 11, 16),
    "IPHEX": date(2014, 4, 22),
    "ISDAC": date(2008, 4, 1),
    "MACPEX": date(2011, 4, 3),
    "MC3E": date(2011, 5, 23),
    "MIDCIX": date(2004, 4, 19),
    "MPACE": date(2004, 10, 5),
    "POSIDON": date(2016, 10, 12),
    "OLYMPEX": date(2015, 11, 13),
}

campaign_end_dates = {
    "AIRS II": date(2003, 11, 19),
    "ARM": date(2000, 3, 19),
    "ATTREX": date(2014, 3, 5),
    "CRYSTAL FACE (UND)": date(2002, 7, 23),
    "CRYSTAL FACE (NASA)": date(2002, 7, 23),
    "ICE L": date(2007, 12, 16),
    "IPHEX": date(2014, 6, 13),
    "ISDAC": date(2008, 4, 29),
    "MACPEX": date(2011, 4, 25),
    "MC3E": date(2011, 6, 2),
    "MIDCIX": date(2004, 4, 30),
    "MPACE": date(2004, 10, 21),
    "POSIDON": date(2016, 10, 28),
    "OLYMPEX": date(2015, 12, 5),
}

min_dates = {
    "AIRS II": date(2003, 11, 14),
    "ARM": date(2000, 3, 9),
    "ATTREX": date(2014, 3, 4),
    "CRYSTAL FACE (UND)": date(2002, 7, 3),
    "CRYSTAL FACE (NASA)": date(2002, 7, 9),
    "ICE L": date(2007, 11, 16),
    "IPHEX": date(2014, 4, 22),
    "ISDAC": date(2008, 4, 1),
    "MACPEX": date(2011, 4, 3),
    "MC3E": date(2011, 5, 23),
    "MIDCIX": date(2004, 4, 19),
    "MPACE": date(2004, 10, 5),
    "POSIDON": date(2016, 10, 12),
    "OLYMPEX": date(2015, 11, 13),
}

max_dates = {
    "AIRS II": date(2003, 11, 19),
    "ARM": date(2000, 3, 19),
    "ATTREX": date(2014, 3, 5),
    "CRYSTAL FACE (UND)": date(2002, 7, 29),
    "CRYSTAL FACE (NASA)": date(2002, 7, 29),
    "ICE L": date(2007, 12, 16),
    "IPHEX": date(2014, 6, 13),
    "ISDAC": date(2008, 4, 29),
    "MACPEX": date(2011, 4, 25),
    "MC3E": date(2011, 6, 2),
    "MIDCIX": date(2004, 5, 6),
    "MPACE": date(2004, 10, 21),
    "POSIDON": date(2016, 10, 28),
    "OLYMPEX": date(2015, 12, 5),
}

Ctopo = [
    [0, "rgb(0, 0, 70)"],
    [0.2, "rgb(0,90,150)"],
    [0.4, "rgb(150,180,230)"],
    [0.5, "rgb(210,230,250)"],
    [0.50001, "rgb(0,120,0)"],
    [0.57, "rgb(220,180,130)"],
    [0.65, "rgb(120,100,0)"],
    [0.75, "rgb(80,70,0)"],
    [0.9, "rgb(200,200,200)"],
    [1.0, "rgb(255,255,255)"],
]
# default for UND to preload
part_type_count = {
    "agg": 2350,
    "budding": 158,
    "bullet": 14,
    "column": 411,
    "compact": 13483,
    "planar": 1036,
    "rimed": 293,
}


color_discrete_map = {
    "aggregate": "rgb(130, 63, 63)",
    "budding rosette": "rgb(198, 137, 100)",
    "bullet rosette": "rgb(217, 175, 107)",
    "compact irregular": "rgb(82,106,131)",
    "column": "rgb(124, 124, 124)",
    "planar polycrystal": "rgb(114, 98, 136)",
    "rimed": "rgb(82, 113, 91)",
}
