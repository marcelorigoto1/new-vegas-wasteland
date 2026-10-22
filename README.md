# Pip-Boy 3000 · Fallout: New Vegas 3D Maps

Interactive 3D maps of the Mojave Wasteland and The Strip (Fallout: New Vegas), the Capital Wasteland
(Fallout 3) and Skyrim, built with
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

Fallout 3 GOTY (`$FO3` is its `Data` directory) uses the same formats; `--game fo3` reads Fallout3.esm with the five
DLCs, places the Capital Wasteland at Washington, D.C. and gives it a green Pip-Boy. Megaton and the downtown D.C.
sectors are separate worldspaces in the Wasteland's coordinates: their markers are merged automatically, and their
models with `--replace-parent`, which drops the Wasteland's low-detail stand-ins in the cells they cover.

```sh
python build_cesium.py "$FO3" --game fo3 --world Wasteland -o site/maps/capital --title "Capital Wasteland" \
    --tileset "Buildings=models/tileset.json"
python build_buildings.py "$FO3" --game fo3 --world Wasteland -o site/maps/capital/buildings
python build_models.py "$FO3" --game fo3 --world Wasteland --world MegatonWorld --world DLC02BaileysCrossroads \
    --world DCworld01 --world DCworld03 --world DCworld05 --world DCworld06 --world DCworld08 --world DCworld09 \
    --world DCworld10 --world DCworld11 --world DCworld12 --world DCworld15 --world DCworld17 --world DCworld18 \
    --replace-parent -o site/maps/capital/models --tile-cells 4 --lod-tileset site/maps/capital/buildings/tileset.json
python build_icons.py "$FO3" --game fo3
```

Skyrim Special Edition (`$SKYRIM` is its `Data` directory). `--game skyrim` reads Skyrim.esm with Update and the
three DLCs, the LZ4-compressed BSAs and the localized names in `Skyrim - Interface.bsa`; Skyrim pages use their own
theme and the vector map icons of `interface\map.swf`.

```sh
python build_cesium.py "$SKYRIM" --game skyrim --world Tamriel -o site/maps/skyrim --title Skyrim
python build_icons.py "$SKYRIM" --game skyrim

# distant object LOD (the game's .bto blocks), then the full models of Tamriel and the walled cities
python build_buildings.py "$SKYRIM" --game skyrim --world Tamriel -o site/maps/skyrim/lod
python build_models.py "$SKYRIM" --game skyrim --world Tamriel --world WhiterunWorld --world SolitudeWorld \
    --world WindhelmWorld --world RiftenWorld --world MarkarthWorld -o site/maps/skyrim/models \
    --tile-cells 4 --lod-tileset site/maps/skyrim/lod/tileset.json
python build_cesium.py "$SKYRIM" --game skyrim --world Tamriel -o site/maps/skyrim --title Skyrim \
    --skip-tiles --tileset "Buildings=models/tileset.json"
```

For a quick test of one area, `build_models.py --cells X0 Y0 X1 Y1` keeps only the references in that range
of cells.

Skyrim's data (about 830 MB) does not fit in this site's GitHub Pages limit together with the Fallout maps, so
`site/maps/skyrim` is its own git repository, [tes-skyrim](https://github.com/marcelorigoto1/tes-skyrim), published
with GitHub Pages from its `main` branch. `maps.json` points the viewer at it (`"data"`); a local copy in
`site/maps/skyrim` takes precedence, so rebuilt data can be checked before it is pushed.

Add `--skip-tiles` to `build_cesium.py` to rewrite only `world.json` (markers) without rebuilding the tiles.

To produce the entrance labels before the terrain step, run `build_models.py` (with `--labels`) first.
`fnv_terrain.py` also writes the heightmaps as GeoTIFF DEMs for GIS software.

| Script | Purpose |
| --- | --- |
| `bsa.py` | Reader for Fallout 3 / New Vegas (zlib) and Skyrim SE (LZ4) BSA archives |
| `swf.py` | Renders vector symbols of a Flash (SWF) file to SVG, for the Skyrim map icons |
| `nif.py` | Minimal NIF (Gamebryo 20.2.0.7) geometry reader |
| `fnv_terrain.py` | Heightmaps, water levels and map markers from the ESM (FNV and Skyrim); GeoTIFF export |
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
affiliated with or endorsed by Bethesda, ZeniMax or Obsidian. Fallout 3 and The Elder Scrolls V: Skyrim © Bethesda Softworks / ZeniMax Media, likewise unaffiliated. Rendering by CesiumJS (Apache 2.0).
