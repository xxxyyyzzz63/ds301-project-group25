from __future__ import annotations

import os

from src.prompt_framework import PromptEngineeringFramework


def main() -> None:
    os.makedirs("outputs", exist_ok=True)

    PromptEngineeringFramework.generate_report()
    PromptEngineeringFramework.save_framework_json("outputs/prompt_framework.json")

    print("Saved prompt framework to: outputs/prompt_framework.json")


if __name__ == "__main__":
    main()