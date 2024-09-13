{
    "image": "mcr.microsoft.com/devcontainers/universal:2",
    "hostRequirements": {
      "cpus": 2
    },
    "waitFor": "onCreateCommand",
    "updateContentCommand": "python3 -m pip install ipythonblocks",
    "postCreateCommand": "",
    "customizations": {
      "codespaces": {
        "openFiles": []
      },
      "vscode": {
        "extensions": [
          "ms-toolsai.jupyter",
          "ms-python.python"
        ]
      }
    }
  }
