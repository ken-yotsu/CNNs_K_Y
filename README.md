# CNNs_K_Y
CNNs
[README.md](https://github.com/user-attachments/files/26403033/README.md)
# ResNet50 regression training (ZIP dataset)

This repository contains the Python scripts used for training and evaluating a ResNet50-based regression model.

> **Data are not included.** The original image dataset and labels cannot be publicly shared due to organizational data governance and usage restrictions.
> To enable *executable verification* of the pipeline, this repo includes a **dummy dataset generator** that produces ZIP files and a `score.csv` matching the expected format.

## Expected data layout

Place the following under `dataset/`:

- `ALC100train.zip`
- `ALC100val.zip`
- `ALC100test.zip`
- `score.csv`

### ZIP internal naming convention

The training script expects image files whose **basename** starts with an integer ID followed by an underscore:

- Example: `00012_xxx.JPG`  → ID = `00012`

The label is obtained as:

- `label = float(score_csv[id][0])`

This reproduces the behavior of the original analysis script.

> ⚠️ If your IDs are 1-based (start at 1) rather than 0-based, adjust `id_offset` in `configs/config.yaml`.

## Quick start (dummy run)

Create a dummy dataset with the same structure and run training:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/make_dummy_dataset.py --out_dir dataset --n_train 100 --n_val 20 --n_test 20
python src/train.py --config configs/config.yaml
```

## Outputs

- `tmp/model-XX.h5` : best checkpoints (by `val_loss`)
- `tmp/training.log` : training log (CSV)
- `tmp/history.json` : training history
- `model/model.hdf5` : full saved model (with optimizer)
- `model/model-opt.hdf5` : lightweight model (no optimizer; inference only)
- `log/` : TensorBoard logs

## Notes on metrics

- The loss used in training is **RMSE** (root mean squared error).

## Citation / manuscript note

When submitting to journals that require code sharing (e.g., JID Innovations), include the public repository URL and release tag in your Data Availability Statement.
A ready-to-edit template is provided in `docs/DAS_template.txt`.
