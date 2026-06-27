# Module for different entry functions to the models.

import cmd
from typing import Iterable, Optional

from src.models import MODEL_REGISTRY
# from src.utils import load_config, load_model_from_artifact, seed_everything


def _print_options(options: Iterable[str]) -> None:
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")


def _parse_model_selection(selection: str) -> list[str]:
    model_names = list(MODEL_REGISTRY.keys())
    selected_names: list[str] = []
    invalid_tokens: list[str] = []
    for token in [token.strip() for token in selection.replace(",", " ").split() if token.strip()]:
        if token.isdigit():
            index = int(token) - 1
            if 0 <= index < len(model_names):
                selected_names.append(model_names[index])
            else:
                invalid_tokens.append(token)
        elif token in MODEL_REGISTRY:
            selected_names.append(token)
        else:
            invalid_tokens.append(token)

    if invalid_tokens:
        raise ValueError(f"Unknown selection: {', '.join(invalid_tokens)}")

    if not selected_names:
        raise ValueError("No valid models selected.")

    seen = set()
    return [name for name in selected_names if not (name in seen or seen.add(name))]


def _prompt_models() -> list[str]:
    model_names = list(MODEL_REGISTRY.keys())
    print("Available models:")
    _print_options(model_names)
    print("Enter one or more model numbers or names separated by commas. Example: '1,3' or 'resnet18, deit'")

    while True:
        selection = input("Select models: ").strip()
        if not selection:
            print("Please select at least one model.")
            continue

        try:
            return _parse_model_selection(selection)
        except ValueError as exc:
            print(exc)
            continue


def test_model() -> None:
    from src.utils import load_model_from_artifact
    mlp_mixer_20 = load_model_from_artifact(
        model_fn=MODEL_REGISTRY.get("mlp_mixer"),
        artifact_path="trash-busters/mlp_mixer_20epochs.pth:latest",
    )
    


def train_model() -> None:
    # lazy import to avoid long startup time
    from src.train import train_all
    from src.utils import load_config, seed_everything
    # Ask for reproducibility
    reproducibility = input("Do you want to ensure reproducibility? (y/n): ").strip().lower()
    if reproducibility == "y":
        # Set the seed for reproducibility
        config = load_config()
        seed_everything(config["project"]["random_seed"])
    # Select models to train
    selected_models = _prompt_models()
    # Train the selected models
    train_all(model_names=selected_models, epochs=config["training"]["epochs"])  


class MainShell(cmd.Cmd):
    intro = "Welcome to the program. Type help or ? to list commands."
    prompt = "(main) "

    def do_test(self, arg: str) -> None:
        """Run the test action."""
        test_model()

    def do_train(self, arg: str) -> None:
        """Run the train action."""
        if arg.strip():
            try:
                selected_models = _parse_model_selection(arg)
            except ValueError as exc:
                print(exc)
                return
            train_model(selected_models)
        else:
            train_model()

    def do_exit(self, arg: str) -> bool:
        """Exit the shell."""
        print("Exiting.")
        return True

    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl-D / EOF."""
        print()
        return True


if __name__ == "__main__":
    MainShell().cmdloop()

