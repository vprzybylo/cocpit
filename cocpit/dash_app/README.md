# DASH app 

## Folder Structure
```
├── README.md
├── app.py
    - main app to run using python
    - creates layout and registers callbacks
├── globals.py
    - holds all global config vars such as dates, campaigns, colors, etc. 
├── assets
│   ├── main.css and images and icons
├── callbacks
│   ├── environment.py
        - plot figures for environmental attributes of ice crystals
│   ├── geometric.py
        - plot figures for geometric attributes of ice crystals
│   ├── navbar.py
        - performs all of the preprocessing/filtering on the df to update figures and legend
│   └── topo_map and topo_flat: deprecated
│   └── topographic.py
        - density_contour and top down map with scatter plot
├── data
│   ├── ETOP.. deprecated
├── file_system_store
    ├── cached session storage from dcc.Store
├── globe
│   ├── deprecated due to loading time
└── layout
    ├── banners.py
        - image and flight count banners under dropdown menu
    ├── content.py
        - main figures
    ├── header.py
        - very top text banner 
    ├── legend.py
        - particle type legend and count
    ├── navbar_collapse.py
        - navbar for selections, which collapses
    ├── sidebar.py
        - deprecated - a diff layout to the navbar
└── processing_scripts
    ├── merge_env.py
        - merge particle property df with environmental properties
        - executable script
    ├── process.py
        - cleans df and updates figure layouts
```