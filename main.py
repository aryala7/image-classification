# Module for different entry functions to the models.

import cmd
from src.models import mlp_mixer
from src.utils import load_model_from_artifact


def test_model() -> None:
    mlp_mixer_20 = load_model_from_artifact(
        mlp_mixer,
        artifact_path="trash-busters/mlp_mixer_20epochs.pth:v3",
    )
    


def train_model() -> None:
    print("Train action executed.")


class MainShell(cmd.Cmd):
    intro = "Welcome to the program. Type help or ? to list commands."
    prompt = "(main) "

    def do_test(self, arg: str) -> None:
        """Run the test action."""
        test_model()

    def do_train(self, arg: str) -> None:
        """Run the train action."""
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

