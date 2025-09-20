# ecoStacks-assignment

You need to have python and these libraries installed:\
-pandas\
-geopandas\
-rasterio\
-numpy\
-shapely\
\
optional:\
-tifffile\
-matplotlib\
-rasterstats\
\
after having installed python, add these libraries using the following command:\
"pip install pandas, geopandas, matplotlib, shapely, rasterio, numpy, rasterstats"\
\
Alternatively:
Simply download the .yml file, from there run "conda env create -f ecoStacksAssignmentEnv.yml" to create a conda env from this yml.
Now you have a env with everything you need to run assignment.py, just activate the newly created env using "conda activate ecoStacksAssignmentEnv".

to run program:\
ensure  the python script is in the same folder as the tiff and csv files;\
run python using the following command: "python assignment.py"
