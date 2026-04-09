import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "src.agent.workflow",
        *sys.argv[1:],
    ]
    subprocess.run(cmd, check=True, cwd=project_root)


if __name__ == "__main__":
    main()
