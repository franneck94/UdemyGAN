import os
import subprocess


directory = "."


def should_exclude_directory(dir_name: str) -> bool:
    exclude_dirs = [".mypy", ".ruff_cache", ".vscode"]
    return any(
        dir_name == exclude_dir
        or dir_name.startswith(exclude_dir + os.path.sep)
        for exclude_dir in exclude_dirs
    )


def main() -> None:
    for root, _, files in os.walk(directory, topdown=True):
        if should_exclude_directory(os.path.basename(root)):
            continue
        num_py_files = len([file for file in files if ".py" in file])
        if num_py_files > 0:
            print(f"dir: {root}, num py files: {num_py_files}")
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                if "checker" in file_path:
                    continue
                try:
                    print(f"\tRunning file: {file_path}")
                    with open(os.devnull, "w") as null_file:
                        process = subprocess.Popen(
                            ["python", file_path],
                            stderr=null_file,
                            stdout=null_file,
                        )
                        process.wait(timeout=10)
                        if process.returncode == 0:
                            print(f"\t\t{file_path} can be executed.")
                        else:
                            print(
                                f"\t\t{file_path} code {process.returncode}"
                            )
                except subprocess.TimeoutExpired:
                    process.terminate()  # pyright: ignore
                    print(f"\t\t{file_path} can be executed (interrupted).")
                except subprocess.CalledProcessError:
                    print(f"\t\t!!! {file_path} cannot be executed !!!")


if __name__ == "__main__":
    main()