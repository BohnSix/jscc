import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=[],
    max_samples=500000,
    dataset_dir="/data/bohnsix/datasets/openimages",
)
