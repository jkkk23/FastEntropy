# FastEntropy

This is the compact code release. It contains only the
best evaluated path:

- multi-level spherical-coordinate occupancy representation;
- phi-first two-stage occupancy prediction;
- hierarchical count-symbol routing with three structural groups and four
  specialized entropy experts per group;
- cross-expert aligned context propagation with group-boundary feature
  alignment and gated lineage memory;
- expert-guided zero-bit geometry refinement with a fixed routing threshold
  of 0.5.

The release includes separate final model bundles for SemanticKITTI and Ford.
Training exposes only the requested optimization and input settings. Compression
and decompression expose only the input folder, output folder, model bundle,
`dd`, and dataset.

## Project Structure

```text
fastEntropy/
├── compress.py                    # Folder-based point-cloud compression
├── decompress.py                  # Bitstream decoding and geometry refinement
├── train.py                       # Entropy-model training
├── train_postprocess.py           # Zero-bit geometry-refinement training
├── config.py                      # Fixed architecture and dataset presets
├── network.py                     # Entropy experts and context propagation
├── kit/
│   ├── checkpoint.py              # Strict checkpoint loading
│   ├── codec.py                   # Entropy coding and decoding pipeline
│   ├── data_preset.py             # SemanticKITTI and Ford input handling
│   ├── nn.py                      # Shared sparse-network components
│   ├── offset.py                  # Center-offset refinement branch
│   ├── op.py                      # Octree and entropy-coding operations
│   ├── postprocess_training.py    # Refinement-target construction
│   ├── spherical.py               # Spherical quantization and headers
│   └── tree_refiner.py            # Expert-guided zero-bit refiner
├── models/
│   ├── semantickitti/
│   │   ├── entropy_model.pt
│   │   └── postprocess.pt
│   └── ford/
│       ├── entropy_model.pt
│       └── postprocess.pt
├── tests/
│   └── test_release.py            # Release-structure checks
├── environment.yml                # Reproducible Conda environment
├── requirements.txt               # Exact Python dependencies
└── README.md
```

## Environment

The code was validated with:

```text
Python       3.10
PyTorch      2.2.1
CUDA         11.8
TorchSparse  2.1.0
torchac      0.9.3
NumPy        1.26.4
```

The reproducible Conda specification is in `environment.yml`. To create it:

```bash
conda env create -f environment.yml
conda activate fastentropy
```

TorchSparse compiles CUDA extensions during installation. A working CUDA 11.8
toolchain and C++ compiler are therefore required. `torchac` also compiles its
CPU arithmetic-coding extension on first import.

## Data

SemanticKITTI is a large-scale outdoor LiDAR sequence dataset collected from a
vehicle-mounted scanner. Following the paper protocol, sequences 00--10 are
used for training and sequences 11--21 for evaluation. With
`--dataset semantickitti`, the code recursively reads raw `.bin` scans
containing float32 `(x, y, z, reflectance)` rows. Coordinates are interpreted
in metres.

Ford is an outdoor mobile-mapping LiDAR dataset with millimetre-precision
geometry. Following the paper protocol, Sequence 01 is used for training and
Sequences 02--03 for evaluation. With `--dataset ford`, the code recursively
reads Ford q1mm `.ply` files. Native millimetre coordinates are converted to
metres internally and restored to millimetres in the reconstructed PLY.

Large datasets should be kept outside the submitted source directory. A
recommended sibling-directory layout is:

```text
../data/
├── semantickitti/
│   ├── train/                     # Sequences 00--10, raw .bin scans
│   └── test/                      # Sequences 11--21, raw .bin scans
└── ford/
    ├── train/                     # Sequence 01, q1mm .ply scans
    └── test/                      # Sequences 02--03, q1mm .ply scans
```

The directory names are not hardcoded; each entry point recursively discovers
the expected file type below `--input_folder`. Use separate folders for
original point clouds, compressed bitstreams, and reconstructed point clouds.
Input basenames must be unique because output files are written as
`<basename>.bin` or `<basename>.ply`.

Supported quality levels are:

```text
SemanticKITTI: dd = 0, 1, 2, 3, 4, 5
Ford:  dd = 1, 2, 3, 4, 5, 6
```

## Pretrained models

Each directory under `models/` is one indivisible model bundle:

```text
models/semantickitti/
  entropy_model.pt
  postprocess.pt

models/ford/
  entropy_model.pt
  postprocess.pt
```

Do not mix the entropy model and geometry-refinement model across datasets. The
decoder checks the refinement training geometry against `--dataset` before
processing a bitstream.

## Compression

```bash
python compress.py \
  --input_folder /path/to/semantickitti_bin_folder \
  --output_folder ./outputs/semantickitti_dd3/compressed \
  --model ./models/semantickitti \
  --dd 3 \
  --dataset semantickitti
```

Ford uses the same five arguments:

```bash
python compress.py \
  --input_folder /path/to/ford_q1mm_ply_folder \
  --output_folder ./outputs/ford_dd3/compressed \
  --model ./models/ford \
  --dd 3 \
  --dataset ford
```

## Decompression

Expert-guided zero-bit geometry refinement is always enabled. The release does
not include ablation branches or per-module timing branches.

```bash
python decompress.py \
  --input_folder ./outputs/semantickitti_dd3/compressed \
  --output_folder ./outputs/semantickitti_dd3/reconstructed \
  --model ./models/semantickitti \
  --dd 3 \
  --dataset semantickitti
```

The `dd` and dataset must match those used for compression.

## Training

Training automatically fits the count and occupancy-symbol prototypes used by
hierarchical expert routing, then trains the fixed submitted architecture from
scratch. It writes checkpoints to `checkpoints/<dataset>/`; this output
location is fixed so that no additional experiment-routing option is exposed.

```bash
python train.py \
  --input_folder /path/to/training_data \
  --channels 16 \
  --kernel_size 3 \
  --learning_rate 0.0005 \
  --lr_decay 0.1 \
  --lr_decay_steps 50000 170000 \
  --max_steps 220000 \
  --dataset semantickitti
```

For Ford, use the Ford training split folder and `--dataset ford`. The included
geometry-refinement checkpoints belong to the included pretrained
entropy models. Training a new entropy checkpoint requires running the fixed
refinement training below before reporting final reconstruction results.

## Zero-bit geometry-refinement training

`train_postprocess.py` is the self-contained training entry point for the
expert-guided zero-bit geometry-refinement module. Only the training folder,
frozen entropy model, and dataset are public inputs:

```bash
python train_postprocess.py \
  --input_folder /path/to/training_data \
  --model ./models/semantickitti \
  --dataset semantickitti
```

`--model` may be an entropy checkpoint or a directory containing
`entropy_model.pt`. Ford uses the same command with its training folder, model,
and `--dataset ford`.

The submitted schedule is fixed in the code and runs without stage-selection
options:

```text
center-offset branch:    50,000 steps
tree branch experts:     80,000 steps
routing calibration:     30,000 steps
routed joint training:   40,000 steps
```

The center-offset branch uses continuous spherical-mean targets. The tree
branch uses the 12 entropy-expert identities selected by count-symbol routing,
the next-level 8-bit occupancy objective, unified geometry/count routing
targets, a 5,000-step joint warm-up, and deployment threshold `0.5`. The
entropy model remains frozen. The final checkpoint is written to
`checkpoints/postprocess/<dataset>/postprocess.pt` and can be paired with the
entropy checkpoint as a normal model bundle.

## Validation

Run the release checks with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -v
python -B -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py')]"
```

The final release was additionally tested end-to-end in the environment
described above on real SemanticKITTI and Ford frames at every supported `dd`.
At `dd=3`, both datasets produced byte-identical compressed bitstreams and
byte-identical reconstructed PLY files relative to the original best-model
implementation.
