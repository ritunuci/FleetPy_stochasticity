What the output files mean
nodes_with_h3.csv

This adds one new important column:

h3_id

Each graph node now belongs to one H3 hexagon.

Example meaning:

node_index = 25
h3_id = 8828342c13fffff

means node 25 lies inside that H3 hexagon.

edges_with_h3.csv

This file adds:

from_h3_id
to_h3_id
mid_h3_id
same_origin_dest_hex

Meaning:

from_h3_id

is the hexagon containing the starting node of the edge.

to_h3_id

is the hexagon containing the ending node of the edge.

mid_h3_id

is the hexagon containing the midpoint of the edge.

same_origin_dest_hex

is True if both edge endpoints are inside the same hexagon.

For most network aggregation tasks, I would use:

mid_h3_id

as the edge-level hex assignment.

h3_hexagons.geojson

This is the polygon layer of the H3 hexagons.

You can open it in:

QGIS
ArcGIS
GeoPandas
kepler.gl
folium

It includes summary columns such as:

num_nodes
num_edge_midpoints
total_edge_distance