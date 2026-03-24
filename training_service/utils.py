import os


def parse_int_or_tuple(val):
    """Parse an int or comma-separated pair into int or (int, int) tuple."""
    try:
        s = str(val)
        if ',' in s:
            return tuple(map(int, s.split(',')))
        return int(s)
    except Exception:
        raise ValueError(f"Invalid numeric input: '{val}'. Please enter an integer or comma-separated pair.")


def get_default_writable_folder():
    """Return a writable folder in the home directory."""
    home_dir = os.path.expanduser("~")
    default_path = os.path.join(home_dir, "neuralnt_service_data")
    os.makedirs(default_path, exist_ok=True)
    return default_path
