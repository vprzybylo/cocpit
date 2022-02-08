from datetime import date


col_names_env = [
    'filename',
    'date',
    'Latitude',
    'Longitude',
    'Altitude',
    'Pressure',
    'Temperature',
    'Ice Water Content',
]

col_names = [
    'filename',
    'date',
    'Frame Width',
    'Frame Height',
    'Particle Width',
    'Particle Height',
    'Cutoff',
    'Aggregate',
    'Budding',
    'Bullet Rosette',
    'Column',
    'Compact Irregular',
    'Fragment',
    'Planar Polycrystal',
    'Rimed',
    'Sphere',
    'Classification',
    'Blur',
    'Contours',
    'Edges',
    'Std',
    'Contour Area',
    'Contrast',
    'Circularity',
    'Solidity',
    'Complexity',
    'Equivalent Diameter',
    'Convex Perimeter',
    'Hull Area',
    'Perimeter',
    'Aspect Ratio',
    'Extreme Points',
    'Area Ratio',
    'Roundness',
    'Perimeter-Area Ratio',
]

campaigns = [
    "AIRS_II",
    "ARM",
    "ATTREX",
    "CRYSTAL_FACE_UND",
    "CRYSTAL_FACE_NASA",
    "ICE_L",
    "IPHEX",
    "ISDAC",
    "MACPEX",
    "MC3E",
    "MIDCIX",
    "MPACE",
    "POSIDON",
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
    "budding",
    "bullet",
    "column",
    "compact irregular",
    "planar polycrystal",
    "rimed",
]

particle_properties = [
    'Contour Area',
    'Circularity',
    'Solidity',
    'Complexity',
    'Equivalent Diameter',
    'Convex Perimeter',
    'Hull Area',
    'Perimeter',
    'Aspect Ratio',
    'Area Ratio',
    'Roundness',
    'Perimeter-Area Ratio',
]

vertical_vars = ['Ice Water Content', 'Temperature', 'Pressure']

campaign_start_dates = {
    "AIRS_II": date(2003, 11, 14),
    "ARM": date(2000, 3, 9),
    "ATTREX": date(2014, 3, 4),
    "CRYSTAL_FACE_UND": date(2002, 7, 13),
    "CRYSTAL_FACE_NASA": date(2002, 7, 20),
    "ICE_L": date(2007, 11, 16),
    "IPHEX": date(2014, 4, 22),
    "ISDAC": date(2008, 4, 1),
    "MACPEX": date(2011, 4, 3),
    "MC3E": date(2011, 5, 23),
    "MIDCIX": date(2004, 4, 19),
    "MPACE": date(2004, 9, 30),
    "POSIDON": date(2016, 10, 12),
    "OLYMPEX": date(2015, 11, 12),
}

campaign_end_dates = {
    "AIRS_II": date(2003, 11, 19),
    "ARM": date(2000, 3, 19),
    "ATTREX": date(2014, 3, 5),
    "CRYSTAL_FACE_UND": date(2002, 7, 16),
    "CRYSTAL_FACE_NASA": date(2002, 7, 21),
    "ICE_L": date(2007, 12, 16),
    "IPHEX": date(2014, 6, 13),
    "ISDAC": date(2008, 4, 29),
    "MACPEX": date(2011, 4, 25),
    "MC3E": date(2011, 6, 2),
    "MIDCIX": date(2004, 5, 6),
    "MPACE": date(2004, 10, 22),
    "POSIDON": date(2016, 10, 28),
    "OLYMPEX": date(2015, 12, 20),
}

min_dates = {
    "AIRS_II": date(2003, 11, 14),
    "ARM": date(2000, 3, 9),
    "ATTREX": date(2014, 3, 4),
    "CRYSTAL_FACE_UND": date(2002, 7, 3),
    "CRYSTAL_FACE_NASA": date(2002, 7, 9),
    "ICE_L": date(2007, 11, 16),
    "IPHEX": date(2014, 4, 22),
    "ISDAC": date(2008, 4, 1),
    "MACPEX": date(2011, 4, 3),
    "MC3E": date(2011, 5, 23),
    "MIDCIX": date(2004, 4, 19),
    "MPACE": date(2004, 9, 30),
    "POSIDON": date(2016, 10, 12),
    "OLYMPEX": date(2015, 11, 12),
}

max_dates = {
    "AIRS_II": date(2003, 11, 19),
    "ARM": date(2000, 3, 19),
    "ATTREX": date(2014, 3, 5),
    "CRYSTAL_FACE_UND": date(2002, 7, 29),
    "CRYSTAL_FACE_NASA": date(2002, 7, 29),
    "ICE_L": date(2007, 12, 16),
    "IPHEX": date(2014, 6, 13),
    "ISDAC": date(2008, 4, 29),
    "MACPEX": date(2011, 4, 25),
    "MC3E": date(2011, 6, 2),
    "MIDCIX": date(2004, 5, 6),
    "MPACE": date(2004, 10, 22),
    "POSIDON": date(2016, 10, 28),
    "OLYMPEX": date(2015, 12, 20),
}

Ctopo = [
    [0, 'rgb(0, 0, 70)'],
    [0.2, 'rgb(0,90,150)'],
    [0.4, 'rgb(150,180,230)'],
    [0.5, 'rgb(210,230,250)'],
    [0.50001, 'rgb(0,120,0)'],
    [0.57, 'rgb(220,180,130)'],
    [0.65, 'rgb(120,100,0)'],
    [0.75, 'rgb(80,70,0)'],
    [0.9, 'rgb(200,200,200)'],
    [1.0, 'rgb(255,255,255)'],
]
