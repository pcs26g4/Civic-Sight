import os
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

# Load .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

BUCKET = "weights"
REMOTE_FILE = "best.pt"

PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
LOCAL_PATH = WEIGHTS_DIR / REMOTE_FILE

WEIGHTS_DIR.mkdir(exist_ok=True)

def download_model():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "Missing Supabase credentials. Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env"
        )

    if LOCAL_PATH.exists():
        print("✅ Model already exists locally")
        return

    print("⬇️ Downloading model from Supabase...")

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    data = supabase.storage.from_(BUCKET).download(REMOTE_FILE)

    with open(LOCAL_PATH, "wb") as f:
        f.write(data)

    print(f"✅ Model downloaded successfully → {LOCAL_PATH}")

if __name__ == "__main__":
    download_model()
