# Model Files

This directory contains model files for KineticAugment.

## Required Files

Download SMPL-X models from: https://smpl-x.is.tue.mpg.de/

Place files as follows:
```
models/
├── smplx/
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_FEMALE.npz
├── mano/
│   ├── MANO_LEFT.pkl
│   └── MANO_RIGHT.pkl
└── vposer/ (optional)
    └── vposer_v1_0/
```

## Verification

Run the download script to check installation:
```bash
python scripts/download_models.py --check
```
