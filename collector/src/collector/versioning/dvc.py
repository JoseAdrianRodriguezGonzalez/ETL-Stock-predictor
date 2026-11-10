import os
import subprocess
import sys


class DVCError(RuntimeError):
    """Raised when a DVC command fails."""


class DVCVersioner:
    def __init__(
        self,
        repo_root: str,
        remote: str | None = None,
    ) -> None:
        self.repo_root = os.path.abspath(repo_root)
        self.remote = remote
        self._dataset_file: str | None = None

        if not os.path.isdir(os.path.join(self.repo_root, ".dvc")):
            raise ValueError(f"No DVC repository found at {self.repo_root}")

    def _run(self, *args: str) -> None:
        command = [sys.executable, "-m", "dvc", *args]

        try:
            result = subprocess.run(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise DVCError(
                f"DVC command failed: {' '.join(command)}\n"
                f"Exit code: {exc.returncode}\n"
                f"Output: {exc.stdout or ''}\n"
                f"Error: {exc.stderr or ''}"
            ) from exc
        except OSError as exc:
            raise DVCError(f"Could not execute DVC command: {exc}") from exc

        if result.stdout.strip():
            print(result.stdout.strip())

    def _resolve_dataset_path(self, dataset_path: str) -> str:
        if os.path.isabs(dataset_path):
            absolute_path = os.path.abspath(dataset_path)
        else:
            absolute_path = os.path.abspath(os.path.join(self.repo_root, dataset_path))

        if os.path.commonpath([self.repo_root, absolute_path]) != self.repo_root:
            raise ValueError("Dataset path must be inside the DVC repository")

        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"Dataset path does not exist: {absolute_path}")

        return absolute_path

    def version_dataset(self, dataset_path: str) -> None:
        absolute_path = self._resolve_dataset_path(dataset_path)

        relative_path = os.path.relpath(
            absolute_path,
            self.repo_root,
        )

        self._run("add", relative_path)

        self._dataset_file = f"{relative_path}.dvc"

    def push(self) -> None:
        if self._dataset_file is None:
            raise DVCError(
                "No dataset has been versioned. Call version_dataset() before push()."
            )

        arguments = ["push", self._dataset_file]

        if self.remote is not None:
            arguments.extend(["-r", self.remote])

        self._run(*arguments)
