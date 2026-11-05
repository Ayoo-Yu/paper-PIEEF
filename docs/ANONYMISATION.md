# Anonymisation

The operational sites are called Wind Farm A and Wind Farm B. Layout east/north
coordinates and elevation are in metres relative to their respective minima.
The origin, coordinate reference system, absolute elevation, geographic
coordinates and original identifiers are omitted. Turbine order is retained:
T01–T29 for A and T01–T72 for B. SCADA suffixes, static tables, preprocessing,
predictions and figures use the same order. B model labels are type_01/type_02.

Weather fields use generic names (data/SCHEMA.md). A retains eleven ECMWF
fields required by the release model and height controls. B retains all
weather channels. B nwp_geopotential values are unchanged; no tensor or
normalisation statistic is changed for that channel. Surface pressure remains
available for density correction.

Timestamps, valid measured targets, predictions and event masks retain their
scientific values. Layout translation preserves relative positions, distances
and elevation differences within floating-point tolerance. Preprocessing
coordinates are translated consistently. Source paths and machine tracking
are removed; code identifiers and configuration keys use public names.

Wind Farm A terrain input is withheld to keep the site anonymous. No A terrain
raster or terrain-derived grid, mesh, surface, STL or geometry is distributed.
The CFD code, lookup tables, intersector, wake, roughness and quality checks
are provided. Rebuilding the original A terrain-flow simulation requires the
withheld input. Kelmarsh may retain its public name and public coordinates.

Figure 6 is regenerated using relative elevation. Its colour scale differs
from the manuscript figure by a constant offset. Its source omits geographic
coordinates and absolute elevation. No name mapping or translation origin is
distributed. Frozen training statistics are retained.
