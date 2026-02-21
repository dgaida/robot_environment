#!/usr/bin/env python3
import re
import os
import sys


def update_version():
    pyproject_path = "pyproject.toml"
    if not os.path.exists(pyproject_path):
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)

    with open(pyproject_path, "r") as f:
        content = f.read()

    # Match version = "X.Y.Z" in [project] section
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)

    current_version = match.group(1)
    try:
        major, minor, patch = map(int, current_version.split("."))
    except ValueError:
        print(f"Error: Invalid version format: {current_version}")
        sys.exit(1)

    new_version = f"{major}.{minor}.{patch + 1}"

    print(f"Updating version from {current_version} to {new_version}")

    # Update pyproject.toml
    new_content = re.sub(r'(^version\s*=\s*")([^"]+)(")', rf"\g<1>{new_version}\g<3>", content, flags=re.MULTILINE)
    with open(pyproject_path, "w") as f:
        f.write(new_content)

    # Update robot_environment/__init__.py
    init_path = "robot_environment/__init__.py"
    if os.path.exists(init_path):
        with open(init_path, "r") as f:
            init_content = f.read()
        new_init_content = re.sub(r'(__version__\s*=\s*")([^"]+)(")', rf"\g<1>{new_version}\g<3>", init_content)
        if init_content != new_init_content:
            with open(init_path, "w") as f:
                f.write(new_init_content)
            print(f"Updated {init_path}")
        else:
            print(f"No version found to update in {init_path}")
    else:
        print(f"Warning: {init_path} not found")

    # Update README.md badge
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            readme_content = f.read()
        # Look for the badge pattern: badge/version-X.Y.Z-blue
        new_readme_content = re.sub(r"(badge/version-)([^/-]+)(-blue)", rf"\g<1>{new_version}\g<3>", readme_content)
        if readme_content != new_readme_content:
            with open(readme_path, "w") as f:
                f.write(new_readme_content)
            print(f"Updated {readme_path}")
        else:
            print(f"No version badge found to update in {readme_path}")
    else:
        print(f"Warning: {readme_path} not found")


if __name__ == "__main__":
    update_version()
