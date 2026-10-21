# Pip-Boy 3000 · Fallout: New Vegas 3D Maps

An interactive 3D map of the Mojave Wasteland and The Strip, built with
[CesiumJS](https://cesium.com/platform/cesiumjs/) from data extracted directly from the game files.
The interface is styled after the Pip-Boy 3000.

- **Terrain**: the game's heightmaps (`LAND` records) as Cesium terrain tiles, textured with the game's
  terrain LOD images and a baked hillshade, with lakes filled to their water level.
- **Buildings and props**: ~133,000 placed objects with their full-detail models (3D Tiles 1.1).
  Distant LOD blocks are replaced by the full models up close. Repeated objects use GPU instancing.
  Rocks, plants and clutter switch to a reduced mesh at mid distance.
- **Locations**: every map marker of the worldspace with its in-game world-map icon, searchable from the
  panel; building entrances on The Strip.
- **Freeside** is a work in progress: the scripts build it, but its data is not published yet. The map
  menu only lists maps whose data is present, so it appears when you build it locally.
- **Radio** (local only, see below): the in-game stations with their playlists, Mr. New Vegas with
  subtitles, and Black Mountain Radio fading into static outside its transmitter's range.

## Viewing it

The site is static. Serve the `site` directory with any web server, for example:

```sh
cd site
python -m http.server 8080
```

Then open <http://localhost:8080>. Use `?map=mojave` or `?map=strip` to open a specific map.
Drag to pan, Ctrl+drag to tilt and scroll to zoom.

## Building the data

The data is generated from your own copy of the game. The scripts need Python 3 with NumPy and GDAL
(the Python bundled with QGIS works; on Windows run `o4w_env.bat` first and set `PYTHONNOUSERSITE=1`).
`DATA` is the game's `Data` directory.

```sh
# terrain, imagery and map markers (one per worldspace)
python build_cesium.py "$DATA" --world WastelandNV -o site/maps/mojave --title "Mojave Wasteland" --tileset "Buildings=models/tileset.json"
python build_cesium.py "$DATA" --world TheStripWorldNew -o site/maps/strip --title "The Strip" --no-water --extra-markers strip_labels.json --tileset "Buildings=models/tileset.json"
python build_cesium.py "$DATA" --world FreesideWorld -o site/maps/freeside --title "Freeside" --no-water --extra-markers freeside_labels.json --tileset "Buildings=models/tileset.json"

# distant LOD building blocks of the Mojave
python build_buildings.py "$DATA" --world WastelandNV -o site/maps/mojave/buildings

# full-detail placed objects
python build_models.py "$DATA" --world WastelandNV -o site/maps/mojave/models --tile-cells 4 --lod-tileset site/maps/mojave/buildings/tileset.json
python build_models.py "$DATA" --world TheStripWorldNew -o site/maps/strip/models --labels strip_labels.json
python build_models.py "$DATA" --world FreesideWorld --world FreesideNorthWorld --world FreesideFortWorld -o site/maps/freeside/models --labels freeside_labels.json

# world-map icons for the location markers
python build_icons.py "$DATA" -o site/icons

# radio stations (local use only)
python build_radio.py "$DATA" -o site/radio
```

To produce the entrance labels before the terrain step, run `build_models.py` (with `--labels`) first.
`fnv_terrain.py` also writes the heightmaps as GeoTIFF DEMs for GIS software.

| Script | Purpose |
| --- | --- |
| `bsa.py` | Reader for Fallout 3 / New Vegas BSA archives |
| `nif.py` | Minimal NIF (Gamebryo 20.2.0.7) geometry reader |
| `fnv_terrain.py` | Heightmaps, water levels and map markers from the ESM; GeoTIFF export |
| `lod_texture.py` | DDS decoding and terrain LOD texture mosaics |
| `build_cesium.py` | Terrain and imagery tiles, `world.json` |
| `build_buildings.py` | Distant LOD building blocks as 3D Tiles |
| `build_models.py` | Full-detail placed objects as 3D Tiles, with shared meshes, instancing and LODs |
| `build_icons.py` | The game's world-map icons, tinted amber, for the location markers |
| `build_radio.py` | Radio stations, playlists, DJ lines and transmitter ranges |

## Radio and copyright

The radio audio is licensed commercial music and voice acting. `site/radio` is excluded from the
repository (`.gitignore`) and the radio tab only appears when that directory exists, so the published
site has no radio. Do not publish it.

## Credits

Fallout: New Vegas © Bethesda Softworks / ZeniMax Media, developed by Obsidian Entertainment. The terrain,
textures and models in this site are extracted from the game. This is an unofficial fan project, not
affiliated with or endorsed by Bethesda, ZeniMax or Obsidian. Rendering by CesiumJS (Apache 2.0).
