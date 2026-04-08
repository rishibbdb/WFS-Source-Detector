# Wide-Field-Survey Source Seed Detector (DRIPS) 

A Claude description of the algorithm ( which is 98% true )

A comprehensive pipeline for automated detection and analysis of gamma-ray sources with application to HAWC (High Altitude Water Cherenkov Observatory) data. The pipeline performs blob detection, filtering, and generates source seeds for 3ML (Multi-Mission Maximum Likelihood) fitting.

## Features

- **Automated Blob Detection**: Multi-scale detection using Difference of Gaussians (DoG)
- **Logic Based Filtering**: Separates point sources (PS) from extended (EXT) sources
- **ROI Management**: Configurable Region of Interest with automatic coordinate conversion
- **YAML-based Configuration**: Easy-to-use configuration file format
- **3ML Integration**: Generates ready-to-use 3ML model files for source fitting
- **Diffuse Template Support**: Optional incorporation of diffuse background templates (e.g., HERMES data)
- **Parallel Processing**: Uses ThreadPoolExecutor for efficient multi-threaded blob detection

## Future Modifications
- Include a plugins for the source detection on Fermi-LAT, H.E.S.S., VERITAS, IceCube, LHAASO
- Implementation of Gammapy source seeding routines

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Pandas
- Astropy (for astronomical coordinate transformations)
- Matplotlib (for visualization)
- scikit-image (for blob detection algorithms)
- PyYAML (for configuration management)
- 3ML and astromodels (for model generation)
- ROOT (optional, for threeML compatibility)

### Setup

```bash
git clone git@github.com:rishibbdb/WFS-Source-Detector.git
cd hawc-source-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a YAML configuration file (e.g., `config.yaml`):

```yaml
paths:
  significance_map: /path/to/map.fits
  main_dir: ./output
  use_dbe: false
  hermes_template: /path/to/hermes.fits  # Optional

coordinates:
  coord_sys: C  # 'C' for RA/Dec, 'G' for Galactic
  ra: 284.333   # RA in degrees (if coord_sys is C)
  dec: 2.8      # Dec in degrees (if coord_sys is C)
  l: null       # Galactic longitude (if coord_sys is G)
  b: null       # Galactic latitude (if coord_sys is G)
  roi_x: 2.0    # ROI half-width in X (degrees)
  roi_y: 5.0    # ROI half-width in Y (degrees)
```

### Configuration Notes

- **coord_sys**: Use `'C'` for ICRS (RA/Dec) or `'G'` for Galactic coordinates
- **roi_x/roi_y**: The pipeline internally expands roi_x by +3° to capture extended sources at ROI edges. Final results are filtered back to original ROI.
- **hermes_template**: Optional path to Hermes diffuse background FITS file. Only used if `use_dbe: true`.

## Usage

### Basic Source Seeding

```bash
python pipeline_sourcedetector.py config.yaml
```

This runs the complete pipeline:
1. Blob detection
2. Filtering (PS vs EXT classification)
3. Saves YAML source catalog
4. Generates 3ML model file

### Source Seeding Only

```bash
python pipeline_sourcedetector.py config.yaml --source-seeding
```

Skips 3ML model generation, saves only source seeds to YAML.

### Verbose Output

```bash
python pipeline_sourcedetector.py config.yaml --verbose
```

Enables detailed logging throughout pipeline execution.

### Skip Model File

```bash
python pipeline_sourcedetector.py config.yaml --skip-model-save
```

Runs pipeline but skips 3ML `.model` file generation.

## Output

The pipeline generates several outputs in the directory specified by `paths.main_dir`:

### Files Generated

| File | Description |
|------|-------------|
| `SkyImage.png` | Original significance map |
| `NormalisedImage.png` | Normalized map for blob detection |
| `SkyImage_Blobs.png` | Map with detected blobs overlaid |
| `FilteredBlobs_Labels.png` | Final filtered sources on sky |
| `Source Seeds.png` | Source seeds overlaid on original map |
| `filtered_sources.yaml` | Catalog of detected sources (YAML) |
| `curModel.model` | 3ML model file for fitting (if generated) |
| `drips_table.png` | Summary table image (if generated) |

### YAML Catalog Format

```yaml
roi_info:
  center_l: 45.123
  center_b: -12.456
  roi_x: 2.0
  roi_y: 5.0
  total_sources_detected: 8
  sources_in_roi: 5

sources:
  - name: Drip0
    ra: 284.123
    dec: 2.456
    ext: 0.25
    circle_radius_deg: 0.25
    sigma_radius: 0.168
  - name: Drip1
    ra: 284.567
    dec: 2.789
    ext: 0.30
    circle_radius_deg: 0.30
    sigma_radius: 0.202
```

### 3ML Model Format

```python
##################################BEGINSOURCE##################################
source_name = "URM"
# Hermes diffuse background (optional)
##################################ENDSOURCE###################################

##################################BEGINSOURCE##################################
source_name = "Drip0"
source_ra  = 284.123
source_dec = 2.456
spectrum = threeML.Powerlaw()
source0 = threeML.PointSource(...)
# ... full 3ML specification
###################################ENDSOURCE###################################

model = threeML.Model(URM, source0, source1, ...)
```

## Pipeline Details

### Blob Detection Algorithm

1. **Normalization**: Map is normalized to [0, 1] range
2. **DoG Filtering**: Difference of Gaussians applied at multiple scales (0.25°, 0.3°, 0.4°, 0.5°)
3. **Thresholding**: Blobs detected above 2σ significance
4. **Morphology**: Separate detection for point sources and extended sources
5. **Deduplication**: Overlapping blobs merged across scales
6. **Significance Cut**: Final 5.1σ threshold applied

### Source Classification

Sources are classified as Point Source (PS) or Extended (EXT) based on:

- **Brightness Fraction**: Fraction of pixels > 5σ threshold
- **Overlap Analysis**: Spatial overlap between PS and EXT candidates
- **Separation**: Angular separation in sky coordinates
- **Test Statistic (TS)**: Relative brightness comparison

#### Classification Rules

| Condition | Action |
|-----------|--------|
| Brightness < 50% | Artifact - remove |
| All PS brighter than EXT | Keep PS, remove EXT |
| Mixed brightnesses | Keep both, remove dimmer PS |
| Angular separation < 0.1° | Merge as single source |
| Separation > 0.3° | Classify as separate PS |
| High TS difference | Classify as separate source |

## Architecture

### Classes

#### `PipelineConfig`
Manages YAML configuration with dot-notation access:
```python
config = PipelineConfig("config.yaml")
roi_x = config.get('coordinates.roi_x', default=1.0)
```

#### `SourceSeedDetector`
Main pipeline class with methods:
- `load_hawc_data()` - Load and crop FITS map
- `normalise_image()` - Prepare map for blob detection
- `blob_detection()` - Multi-scale DoG detection
- `group_blobs()` - Match PS and EXT blobs
- `blob_filters()` - Apply classification rules
- `save_model()` - Export YAML catalog
- `run()` - Execute full pipeline

## Extending the Pipeline

### Custom Blob Detection Parameters

Edit in `__init__`:
```python
self.SIG_THRESHOLD = 5.0        # Minimum significance
self.SMEAR_RADII = [0.25, 0.3, 0.4, 0.5]  # Detection scales
```

### Custom Classification Rules

Modify `blob_filters()` method or `_tag_small_blob()` helper function.

### Additional Output Formats

Extend `save_model()` to export additional catalog formats (FITS, CSV, etc.).

## Performance Notes

- **Typical Runtime**: 1-5 minutes depending on map size and significance
- **Memory Usage**: ~500 MB for standard 4°×10° ROI
- **Parallelization**: 2-thread ThreadPoolExecutor for PS and EXT detection

### Optimization Tips

- Reduce `SMEAR_RADII` for faster processing
- Increase `SIG_THRESHOLD` to skip faint sources
- Use smaller ROI when possible

## Known Issues

- Bright extended sources near ROI edges may be partially detected due to map cropping
- Very close sources (< 0.1°) are sometimes merged incorrectly
- Hermes template support requires specific FITS format compatibility

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for HAWC Observatory collaboration
- Uses scikit-image for blob detection algorithms
- 3ML framework for source modeling
- Astropy for coordinate transformations

## Support

For issues, questions, or suggestions:
- Open an [GitHub Issue](https://github.com/rishibbdb/WFS-Source-Detector/issues)
- Contact: rbabu@mtu.edu, rbabu@icecube.wisc.edu

## References

- [HAWC Observatory](https://www.hawc-observatory.org/)
- [3ML Documentation](https://threeml.readthedocs.io/)
- [Astropy Documentation](https://docs.astropy.org/)
- [scikit-image Documentation](https://scikit-image.org/)

---

**Version**: 1.0.0  
**Last Updated**: April 2026
