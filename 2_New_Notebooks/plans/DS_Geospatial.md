---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# 🗺️ Geospatial Data Science
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Scientist · Wiki + Tutoriel

**Dataset(s)** : NYC Taxi (lat/lon), Natural Earth, OpenStreetMap

Analyse de données géospatiales : projections, jointures spatiales, viz cartographiques, ML géo.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Cadrer : projections, CRS, géodésie
<!-- #endregion -->

<!-- #region -->
Notions : EPSG, WGS84 (4326), Web Mercator (3857), Lambert93 (FR), latlon vs xy. Transformation entre CRS avec `pyproj`.
<!-- #endregion -->

<!-- #region -->
## 2. GeoPandas — la lib de référence
<!-- #endregion -->

<!-- #region -->
GeoSeries, GeoDataFrame, geometry column. Lecture shapefile / GeoJSON / GeoPackage. Code visé : load + plot.
<!-- #endregion -->

<!-- #region -->
## 3. Shapely — primitives géométriques
<!-- #endregion -->

<!-- #region -->
Point, LineString, Polygon, MultiPolygon. Opérations : intersection, union, difference, buffer, convex hull, distance.
<!-- #endregion -->

<!-- #region -->
## 4. Spatial joins
<!-- #endregion -->

<!-- #region -->
`gpd.sjoin` avec predicates : within, intersects, contains, touches, nearest. Cas : associer chaque point taxi à son borough.
<!-- #endregion -->

<!-- #region -->
## 5. Indexation spatiale
<!-- #endregion -->

<!-- #region -->
R-tree, k-d tree, `rtree`. Pourquoi indispensable sur > 100k géométries.
<!-- #endregion -->

<!-- #region -->
## 6. Visualisation cartographique
<!-- #endregion -->

<!-- #region -->
`folium` (Leaflet, interactif), `geoplot` (statique stat), `plotly.express` (choropleth/mapbox), `kepler.gl` (super avancé).
<!-- #endregion -->

<!-- #region -->
## 7. Géocodage
<!-- #endregion -->

<!-- #region -->
`geopy` : Nominatim (OSM gratuit), Google Maps API. Rate limits, caching.
<!-- #endregion -->

<!-- #region -->
## 8. Raster vs vector
<!-- #endregion -->

<!-- #region -->
Vector : points/lignes/polygones. Raster : grille de pixels (satellite). `rasterio`, NetCDF, xarray pour multi-band.
<!-- #endregion -->

<!-- #region -->
## 9. DuckDB spatial + PostGIS
<!-- #endregion -->

<!-- #region -->
DuckDB extension spatial = mini PostGIS embedded. PostGIS pour stack serveur. ST_* fonctions.
<!-- #endregion -->

<!-- #region -->
## 10. ML géospatial
<!-- #endregion -->

<!-- #region -->
Features : distance to X, density grid, kNN-density. Choropleth aggregation. Graph Neural Networks sur routes.
<!-- #endregion -->

<!-- #region -->
## 11. H3 (Uber) hexagons
<!-- #endregion -->

<!-- #region -->
Indexation hiérarchique de la sphère par hexagones. Idéal pour aggregation grids uniformes.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [GeoPandas docs](https://geopandas.org/)
- [Shapely docs](https://shapely.readthedocs.io/)
- [H3 Uber](https://h3geo.org/)
- [Geocomputation with Python](https://py.geocompx.org/)
<!-- #endregion -->
