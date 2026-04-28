from brainsr.data.dataset import MRISliceDataset


def test_dataset_loads_lr_hr_pairs(sample_dir):
    ds = MRISliceDataset(root=sample_dir, split="train", scale=4, deterministic_lr=True)
    assert len(ds) > 0
    lr, hr = ds[0]
    assert hr.shape[0] == 1 and hr.ndim == 3
    assert lr.shape[0] == 1 and lr.ndim == 3
    assert hr.shape[-1] == lr.shape[-1] * 4
    assert 0.0 <= float(hr.min())
    assert float(hr.max()) <= 1.0
    assert 0.0 <= float(lr.min())
    assert float(lr.max()) <= 1.0


def test_splits_disjoint(sample_dir):
    train = set(MRISliceDataset(root=sample_dir, split="train").files)
    val = set(MRISliceDataset(root=sample_dir, split="val").files)
    test = set(MRISliceDataset(root=sample_dir, split="test").files)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
