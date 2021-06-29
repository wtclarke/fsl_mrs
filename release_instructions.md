Instructions for releasing a new version of FSL-MRS
===================================================
1. All tests passing
2. Identify new version number
3. Update CHANGELOG.rst
4. Commit to development fork master branch (e.g. wclarke/fsl_mrs:master)
5. Initiate merge request, and merge into fsl/fsl_mrs:master
6. Create new tag -tag name should be "x.x.x" with the message "Vx.x.x". New changelog contents are copied into release notes.
7. This will trigger doc build (published [here](https://open.win.ox.ac.uk/pages/fsl/fsl_mrs/)) and package build (currently broken).
8. Pull upstream master into development fork (with tags)  
    ```git pull upstream --tags```
    ```git push origin --tags```
9. Trigger manual package build using the [fsl/conda/fsl-mrs:pre-conda](https://git.fmrib.ox.ac.uk/fsl/conda/fsl-mrs:pre-conda) branch. Do this by updating the version number [here](https://git.fmrib.ox.ac.uk/fsl/conda/fsl-mrs/-/blob/mnt/pre-conda/meta.yaml#L6). This will trigger a pipeline from which the conda package can be downloaded (*Download build-noarch-conda-package:archive artifact*).
10. Send to MW for manual upload to the FSL conda channel.