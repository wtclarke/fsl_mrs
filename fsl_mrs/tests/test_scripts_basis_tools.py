'''FSL-MRS test script

Test the basis tools script

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
import subprocess
from pathlib import Path
from shutil import copytree
import json
import numpy as np
# from unittest.mock import patch

# Files
testsPath = Path(__file__).parent

jmrui = testsPath / 'testdata/mrs_io/basisset_JMRUI'
lcm = testsPath / 'testdata/mrs_io/basisset_LCModel.BASIS'
fsl = testsPath / 'testdata/mrs_io/basisset_FSL'
mac = testsPath / 'testdata/basis_tools/macSeed.json'
diff1 = testsPath / 'testdata/basis_tools/low_res_off'
diff2 = testsPath / 'testdata/basis_tools/low_res_on'
raw = testsPath / 'testdata/basis_tools/RawBasis_for_PRESSGE_TE_35_BW_4000_NPts_2048'
jmrui_basis_path = testsPath / 'testdata' / 'mrs_io' / 'basisset_JMRUI'
osprey_2d = testsPath / 'testdata' / 'basis_tools' / 'basis_philips_slaser35.mat'
osprey_3d = testsPath / 'testdata' / 'basis_tools' / 'basis_philips_hercules-press.mat'


def test_info():
    subprocess.check_call(['basis_tools', 'info', str(jmrui)])
    subprocess.check_call(['basis_tools', 'info', str(lcm)])
    subprocess.check_call(['basis_tools', 'info', str(fsl)])


# To figure out one day
# @patch("matplotlib.pyplot.show")
# def test_vis(tmp_path):
#     subprocess.check_call(['basis_tools', 'vis',
#                            '--ppmlim', '0.2', '5.2',
#                            str(jmrui)])

def test_convert_lcmbasis(tmp_path):
    subprocess.check_call(['basis_tools', 'convert',
                           str(lcm),
                           str(tmp_path / 'new')])

    assert (tmp_path / 'new').is_dir()
    assert (tmp_path / 'new' / 'NAA.json').is_file()

    subprocess.check_call(['basis_tools', 'convert',
                           str(lcm),
                           str(tmp_path / 'new_nuc'),
                           '--nucleus', '31P'])

    assert (tmp_path / 'new_nuc').is_dir()
    assert (tmp_path / 'new_nuc' / 'NAA.json').is_file()


def test_convert_raw(tmp_path):
    subprocess.check_call(['basis_tools', 'convert',
                           str(raw),
                           '--bandwidth', '4000',
                           '--fieldstrength', '3.0',
                           str(tmp_path / 'new_raw')])

    assert (tmp_path / 'new_raw').is_dir()
    assert (tmp_path / 'new_raw' / 'NAA.json').is_file()


def test_convert_jmrui(tmp_path):
    subprocess.check_call(['basis_tools', 'convert',
                           str(jmrui_basis_path),
                           '--bandwidth', '4000',
                           '--fieldstrength', '3.0',
                           str(tmp_path / 'new_jmrui')])

    assert (tmp_path / 'new_jmrui').is_dir()
    assert (tmp_path / 'new_jmrui' / 'NAA.json').is_file()


def test_convert_with_remove(tmp_path):
    subprocess.check_call(['basis_tools', 'convert',
                           '--remove_reference',
                           str(lcm),
                           str(tmp_path / 'new')])

    assert (tmp_path / 'new').is_dir()
    assert (tmp_path / 'new' / 'NAA.json').is_file()


def test_convert_osprey_2d(tmp_path):
    out1 = tmp_path / 'osprey_out1'
    out2 = tmp_path / 'osprey_out2'

    subprocess.check_call(['basis_tools', 'convert', str(osprey_2d), str(out1)])
    subprocess.check_call([
        'basis_tools', 'convert', str(osprey_2d), str(out2),
        '--description', 'A', 'B', 'C', 'D', 'diff1', 'diff2', 'sum'
    ])
    assert out1.is_dir()
    assert out2.is_dir()
    assert (out1 / 'NAA.json').is_file()
    assert (out2 / 'NAA.json').is_file()

    with open(out1 / 'NAA.json') as f:
        ref_json = json.load(f)
    with open(out2 / 'NAA.json') as f:
        est_json = json.load(f)
    assert list(ref_json.keys()) == list(est_json.keys())
    assert np.allclose(ref_json['basis']['basis_re'], est_json['basis']['basis_re'], atol=0.0001)
    assert np.allclose(ref_json['basis']['basis_im'], est_json['basis']['basis_im'], atol=0.0001)
    assert np.allclose(ref_json['basis']['basis_dwell'], est_json['basis']['basis_dwell'], atol=0.0001)
    assert np.allclose(ref_json['basis']['basis_centre'], est_json['basis']['basis_centre'], atol=0.0001)
    assert np.allclose(ref_json['basis']['basis_width'], est_json['basis']['basis_width'], atol=0.0001)
    assert ref_json['basis']['basis_name'] == est_json['basis']['basis_name']
    assert np.allclose(ref_json['basis']['centralShift'], est_json['basis']['centralShift'], atol=0.0001)


def test_convert_osprey_3d(tmp_path):
    out1 = tmp_path / 'osprey_out1'
    out2 = tmp_path / 'osprey_out2'

    subprocess.check_call(['basis_tools', 'convert', str(osprey_3d), str(out1)])
    subprocess.check_call([
        'basis_tools', 'convert', str(osprey_3d), str(out2),
        '--description', 'A', 'B', 'C', 'D', 'diff1', 'diff2', 'sum'
    ])

    desc1 = ['FID_1', 'FID_2', 'FID_3', 'FID_4', 'FID_5', 'FID_6', 'FID_7']
    desc2 = ['A', 'B', 'C', 'D', 'diff1', 'diff2', 'sum']
    for i, j in zip(desc1, desc2):
        path1 = Path(str(out1) + '_' + i)
        path2 = Path(str(out2) + '_' + j)
        assert path1.is_dir()
        assert path2.is_dir()
        assert (path1 / 'NAA.json').is_file()
        assert (path2 / 'NAA.json').is_file()

        with open(path1 / 'NAA.json') as f:
            ref_json = json.load(f)
        with open(path2 / 'NAA.json') as f:
            est_json = json.load(f)
        assert list(ref_json.keys()) == list(est_json.keys())
        assert np.allclose(ref_json['basis']['basis_re'], est_json['basis']['basis_re'], atol=0.0001)
        assert np.allclose(ref_json['basis']['basis_im'], est_json['basis']['basis_im'], atol=0.0001)
        assert np.allclose(ref_json['basis']['basis_dwell'], est_json['basis']['basis_dwell'], atol=0.0001)
        assert np.allclose(ref_json['basis']['basis_centre'], est_json['basis']['basis_centre'], atol=0.0001)
        assert np.allclose(ref_json['basis']['basis_width'], est_json['basis']['basis_width'], atol=0.0001)
        assert ref_json['basis']['basis_name'] == est_json['basis']['basis_name']
        assert np.allclose(ref_json['basis']['centralShift'], est_json['basis']['centralShift'], atol=0.0001)


def test_add(tmp_path):
    out_loc = tmp_path / 'test_basis'
    copytree(fsl, out_loc)

    subprocess.check_call(['basis_tools', 'add',
                           '--info', 'some info',
                           '--scale',
                           '--conj',
                           '--name', 'new_basis',
                           str(mac),
                           str(out_loc)])

    assert (out_loc / 'new_basis.json').is_file()


def test_shift(tmp_path):
    out_loc = tmp_path / 'test_basis'

    subprocess.check_call(['basis_tools', 'shift',
                           str(fsl),
                           'NAA',
                           '1.0',
                           str(out_loc)])

    assert out_loc.is_dir()
    assert (out_loc / 'NAA.json').is_file()

    out_loc = tmp_path / 'test_basis2'

    subprocess.check_call(['basis_tools', 'shift',
                           str(fsl),
                           'all',
                           '1.0',
                           str(out_loc)])

    assert out_loc.is_dir()
    assert (out_loc / 'NAA.json').is_file()


def test_rescale(tmp_path):
    out_loc = tmp_path / 'test_basis1'

    subprocess.check_call(['basis_tools', 'scale',
                           str(fsl),
                           'NAA',
                           str(out_loc)])
    assert out_loc.is_dir()
    assert (out_loc / 'NAA.json').is_file()

    out_loc = tmp_path / 'test_basis2'
    subprocess.check_call(['basis_tools', 'scale',
                           str(fsl),
                           'NAA',
                           '--target_scale', '1.0',
                           str(out_loc)])

    assert out_loc.is_dir()
    assert (out_loc / 'NAA.json').is_file()


def test_diff(tmp_path):
    out_loc = tmp_path / 'test_basis'

    subprocess.check_call(['basis_tools', 'diff',
                           '--add_or_sub', 'sub',
                           str(diff1),
                           str(diff2),
                           str(out_loc)])

    assert out_loc.is_dir()
    assert (out_loc / 'NAA.json').is_file()


def test_conj(tmp_path):
    out_loc = tmp_path / 'test_basis'

    subprocess.check_call(['basis_tools', 'conj',
                           '--metabolite', 'NAA',
                           str(fsl),
                           str(out_loc)])

    assert out_loc.is_dir()
    assert (out_loc / 'NAA.json').is_file()


def test_remove(tmp_path):
    out_loc = tmp_path / 'test_basis'

    subprocess.check_call(['basis_tools', 'remove_peak',
                           '--all',
                           '--ppmlim', '1.8', '2.0',
                           str(fsl),
                           str(out_loc)])

    assert out_loc.is_dir()
    assert (out_loc / 'NAA.json').is_file()


def test_remove_hlsvd(tmp_path):
    out_loc = tmp_path / 'test_basis'

    subprocess.check_call(['basis_tools', 'remove_peak',
                           '--all',
                           '--ppmlim', '1.8', '2.0',
                           '--hlsvd',
                           str(fsl),
                           str(out_loc)])

    assert out_loc.is_dir()
    assert (out_loc / 'NAA.json').is_file()


# TO DO: Add tests fro shift_all
