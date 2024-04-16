from pathlib import Path
import shutil


def make_output_folder(output: Path, overwrite: bool):
    """Creates a folder (output), overwriting if specified

    :param output: Output folder to create
    :type output: Path
    :param overwrite: Remove pre-existing folder?
    :type overwrite: bool
    """
    def del_and_make():
        shutil.rmtree(output)
        output.mkdir(exist_ok=True, parents=True)

    if output.is_dir() and overwrite:
        del_and_make()
    elif output.is_dir() and not overwrite:
        response_str = f"Folder '{output}' exists. Are you sure you want to delete it? [Y,N]\n"
        if input(response_str).upper() == "Y":
            del_and_make()
        else:
            print('Stopping early.')
            exit()
    else:
        output.mkdir(exist_ok=True, parents=True)
