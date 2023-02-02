# DASH app 

## Data Storage
    The code lives in /home/vanessa/hulk/cocpit/cocpit/dash_app
    Final databases from the ML classification model live in /home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0
    Environmental data provided by Carl Schmitt lives at https://drive.google.com/drive/folders/1x05HbKzr0UoGYmGbiwjdowizawYdtgLI?usp=sharing
        """From Carl: I used the "other" probes to determine the percentage of Particle number, Particle projected area, or Particle mass that was from particles larger than 1000 microns.  PSD IWC is also in all of the files, but I am a bit concerned about reliability there."""
    The classification databases and environmental databases are merged using /home/vanessa/hulk/cocpit/cocpit/dash_app/processing_scripts/merge_env.py 
    and saved to /home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0/merged_env for each campaign

## Units of csv
    filename,date,frame width [pixels],frame height [pixels],particle width [microns],particle height [microns],cutoff [%],aggregate [%],budding rosette [%],bullet rosette [%],column [%],compact irregular [%],fragment [%],planar polycrystal [%],rimed [%],sphere [%],classification,perim [pixels],hull_area [pixels],convex_perim [pixels],blur,contours [#],contrast,cnt_area [pixels],circularity,solidity,complexity,equiv_d,phi,extreme_points,filled_circular_area_ratio,roundness,perim_area_ratio
    
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