# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AperX is a Python package for astronomical source detection and aperture photometry. It provides routines for PSF generation, source detection, and photometry on astronomical images, primarily designed for JWST and HST mosaics.

## Key Commands

### Installation and Setup
```bash
pip install -e .  # Install package in development mode
```

### Running the Main Pipeline
```bash
aperx -c config.toml  # Run aperture photometry pipeline with config file
```

### Development Commands
```bash
python -m aperx.engine -c example_config.toml  # Run main pipeline directly
```

## Recent Architecture Changes

The codebase has been refactored to use **per-tile catalogs** instead of a single global catalog:

### Key Changes
- **PSF Generation**: Now a standalone function `generate_psfs()` that operates globally across all tiles
- **Per-Tile Processing**: Each tile gets its own `Catalog()` object for independent processing
- **Catalog Merging**: New `merge_catalogs()` function combines per-tile catalogs after processing
- **Method Signatures**: Removed `tile` parameters from all `Catalog` methods since each catalog handles one tile

### Updated Workflow
1. **Global PSF Generation**: `generate_psfs()` operates on all images across all tiles
2. **Per-Tile Catalogs**: Create separate `Catalog` objects for each tile's images
3. **Parallel Processing**: Process each tile independently with its own catalog
4. **Catalog Merging**: Use `merge_catalogs()` to combine results from all tiles
5. **Post-Processing**: Apply corrections and calibrations to merged catalog

## Architecture

### Core Components

- **`aperx/engine.py`**: Main entry point and pipeline orchestrator
  - Contains `main()` function that handles CLI arguments and coordinates the full pipeline
  - Parses configuration files and manages the workflow through different processing stages
  
- **`aperx/catalog.py`**: Catalog management and photometry operations
  - Handles source detection, aperture photometry, and catalog merging
  - Contains functions for PSF corrections and error calibration

- **`aperx/image.py`**: Image data structures and operations
  - `Image` dataclass for individual mosaic tiles with sci/err/wht/psf extensions
  - `Images` container class for managing collections of images
  - Handles PSF matching and convolution operations

- **`aperx/psf.py`**: PSF generation and management
  - Point Spread Function modeling and operations

- **`aperx/utils.py`**: Utility functions and configuration parsing
  - Configuration file parsing and validation
  - Logging setup and common helper functions

### Processing Pipeline

The pipeline consists of several configurable stages:

1. **PSF Generation** (`psf_generation`): Generate PSFs from stellar sources
2. **PSF Homogenization** (`psf_homogenization`): Match PSFs across filters  
3. **Source Detection** (`source_detection`): Detect sources using hot+cold detection schemes
4. **Photometry** (`photometry`): Perform aperture and AUTO photometry
5. **Post-processing** (`post_processing`): Merge tiles, apply corrections, and calibrations

### Configuration System

Uses TOML configuration files with hierarchical structure:
- Mosaic specifications with filter/tile mapping
- Processing step parameters with nested sections
- Supports template substitution with `[keywords]` for flexible file paths

### Data Extensions

Images use multiple FITS extensions:
- `sci`: Science data
- `err`: Error/uncertainty maps  
- `wht`: Weight maps
- `psf`: Point spread function data

## Dependencies

Key astronomical Python packages:
- `astropy`: FITS I/O, WCS, coordinates, units
- `photutils`: Aperture photometry utilities
- `sep`: Source Extractor Python implementation
- `scipy`, `numpy`: Core scientific computing
- `matplotlib`: Plotting and visualization