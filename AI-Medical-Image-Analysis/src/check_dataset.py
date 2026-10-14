"""Quick dataset check without loading TensorFlow model weights."""
from src.config import DATA_ROOT, CLASS_NAMES
from src.data_utils import collect_image_paths

def main():
    for split in ("train", "test"):
        _, labels = collect_image_paths(DATA_ROOT / split)
        print(f"{split}: {len(labels)} images")
        for index, name in enumerate(CLASS_NAMES):
            print(f"  {name}: {(labels == index).sum()}")
    print("Dataset folders OK")

if __name__ == "__main__":
    main()
