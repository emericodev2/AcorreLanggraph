# Dependencies Directory

This directory contains the local dependencies that the LangGraph platform expects during deployment.

## Purpose

The LangGraph platform looks for a `/deps/*` directory during the build process to install local dependencies. This directory structure satisfies that requirement.

## Contents

- `__init__.py` - Makes this a Python package
- `setup.py` - Defines the package and its dependencies
- `README.md` - This file

## How it works

During deployment, the platform will:
1. Look for the `/deps/*` directory
2. Install packages from this directory using `pip install -e /deps/*`
3. Use the constraints.txt file for version resolution

This structure ensures that your custom code and dependencies are properly installed during the LangGraph platform deployment process.
