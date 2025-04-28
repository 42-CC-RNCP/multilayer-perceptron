# AI Project Repository Template

This is a template for organizing an AI project repository.


## Initial Setup

Activate the virtual environment by poetry:

```bash
source $(poetry env info --path)/bin/activate
```

Replace `REPLACE_WITH_YOUR_NAME` and `REPLACE_WITH_YOUR_EMAIL` with your name in the following files:
- `pyproject.toml`
- `.devcontainer/devcontainer.json`

Replace `REPLACE_WITH_YOUR_NAME` with your project name in the following folder:
- `src/REPLACE_WITH_YOUR_NAME`

Or run the script `setup.sh` to do this automatically:

1. Replace the placeholders in the files.
```bash
.scripts/setup.sh
```

2. Install the dependencies:
```bash
.scripts/install.sh
```

3. Rebuild and open the devcontainer:

4. Activate the virtual environment by poetry:
```bash
.scripts/activate.sh
```