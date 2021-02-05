'''FSL-MRS test script

Test the merge report script

Copyright Will Clarke, University of Oxford, 2021'''

# Imports
import subprocess
from pathlib import Path
import shutil

# Files
testsPath = Path(__file__).parent
report_dir = testsPath / 'testdata/merge_mrs_reports'


def test_merge_mrs_reports(tmp_path):

    for f in report_dir.glob('*.html'):
        shutil.copy(f, tmp_path)

    htmlfiles = list(tmp_path.glob('*.html'))

    subprocess.check_call(['merge_mrs_reports',
                           '-d', 'test',
                           '-f', 'test.html',
                           '-o', str(tmp_path),
                           '--delete'] + htmlfiles)

    assert (tmp_path / 'test.html').exists()
