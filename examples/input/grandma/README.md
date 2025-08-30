# Grandma Dataset - CAP4D Format

This directory contains the grandma dataset converted from Pixel3DMM format to CAP4D format.

## Dataset Information

- **Original Source**: Pixel3DMM tracking output (`grandma_nV1_noPho_noMICA_FLAME23_uv2000.0_n1000.0`)
- **Conversion Date**: Generated automatically from Pixel3DMM checkpoint
- **Format**: CAP4D compatible format (same as Lincoln dataset)

## File Structure

```
grandma/
├── fit.npz                    # FLAME parameters and camera data
├── reference_images.json     # Reference frame configuration
├── alignment.npz             # Alignment data (minimal)
├── images/
│   └── cam0/
│       └── grandma.jpg       # Input image
└── visualization/
    └── vis_cam0.mp4          # Tracking result video
```

## Dataset Details

### FLAME Parameters (fit.npz)

- **Shape Parameters**: 150 coefficients (range: -0.379 to 0.429)
- **Expression Parameters**: 65 coefficients (range: -0.739 to 0.841) 
- **Head Rotation**: 3D axis-angle rotation (magnitude: 0.080 radians / 4.6 degrees)
- **Translation**: 3D translation (magnitude: 0.006)
- **Eye Rotation**: 3D eye rotation parameters

### Camera Parameters

- **Focal Length**: fx=896.5, fy=896.5 pixels
- **Principal Point**: cx=128.0, cy=128.0 pixels  
- **Image Resolution**: 256x256 pixels
- **Extrinsics**: 4x4 camera transformation matrix

### Reference Configuration

- **Single View**: cam0 only
- **Single Frame**: timestep 0
- **Camera Order**: ['cam0']

## Usage

This dataset can be used directly with CAP4D inference pipeline:

```python
# Example usage with CAP4D
from cap4d.inference.data.reference_data import ReferenceDataset

# Load the dataset
dataset = ReferenceDataset(
    data_path="/path/to/grandma",
    resolution=512,
    downsample_ratio=8
)

# The dataset is now ready for CAP4D inference
```

## Conversion Details

This dataset was converted from Pixel3DMM format using the following transformations:

1. **Rotation Format**: rotation_6d → axis-angle (Rodrigues)
2. **Coordinate System**: Pixel3DMM OpenGL → CAP4D OpenCV/PyTorch3D
3. **Camera Model**: Custom intrinsics → Standard camera matrices
4. **Parameter Structure**: Frame-based → CAP4D npz format

## Validation Results

All converted parameters passed validation checks:

- ✅ Parameter shapes match CAP4D format
- ✅ No NaN or infinite values detected
- ✅ Parameter ranges are reasonable
- ✅ Camera intrinsics are valid
- ✅ File structure matches Lincoln dataset

## Notes

- This is a single-frame, single-view dataset
- Original tracking was performed using FLAME2023 model
- Image resolution was standardized to match CAP4D expectations
- Coordinate system transformations have been applied for compatibility

## Troubleshooting

If you encounter issues using this dataset:

1. Verify all required files are present
2. Check that image files can be loaded properly
3. Ensure CAP4D dependencies are correctly installed
4. Compare parameter ranges with reference datasets

For more information about the conversion process, see:
- `convert_grandma_to_cap4d.py` - Conversion script
- `test_converted_data.py` - Validation script
- `FLAME_Parameter_Analysis_Documentation.md` - Detailed analysis