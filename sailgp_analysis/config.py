from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "DataChallenge_Export"
OUTPUT_DIR = REPO_ROOT / "analysis_output"
WEB_DATA_DIR = REPO_ROOT / "web" / "data"
STATE_FILE = OUTPUT_DIR / "agent_state.json"
SNAPSHOT_FILE = WEB_DATA_DIR / "snapshot.json"
VENUES = ["Bermuda", "Halifax"]
