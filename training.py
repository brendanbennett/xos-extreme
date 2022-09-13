from pathlib import Path

def load_training_data(revision, training_dir):
    revisions_paths = [x for x in Path(".", training_dir).iterdir() if x.is_dir()]
    revision_path = Path(".", training_dir, str(revision))
    if revision_path not in revisions_paths:
        raise IOError(f"No saved data for revision {revision}")