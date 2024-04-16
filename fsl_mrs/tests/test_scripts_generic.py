"""Test the general functions in the scripts folder
"""

import pytest

from fsl_mrs import scripts


def test_make_output(tmp_path, monkeypatch):

    scripts.make_output_folder(tmp_path / 'aa', overwrite=True)
    assert (tmp_path / 'aa').is_dir()

    scripts.make_output_folder(tmp_path / 'a', overwrite=False)
    assert (tmp_path / 'a').is_dir()

    (tmp_path / 'a' / 'b').mkdir()
    scripts.make_output_folder(tmp_path / 'a', overwrite=True)
    assert (tmp_path / 'a').is_dir()
    assert not (tmp_path / 'b').is_dir()

    (tmp_path / 'a' / 'c').mkdir()
    monkeypatch.setattr('builtins.input', lambda _: "Y")
    scripts.make_output_folder(tmp_path / 'a', overwrite=True)
    assert (tmp_path / 'a').is_dir()
    assert not (tmp_path / 'c').is_dir()

    monkeypatch.setattr('builtins.input', lambda _: "N")
    with pytest.raises(SystemExit) as e:
        scripts.make_output_folder(tmp_path / 'a', overwrite=False)
    assert e.type == SystemExit
