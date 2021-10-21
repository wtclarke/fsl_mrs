Instructions for releasing a new version of FSL-MRS
===================================================
1. Establish feature branch in local copy of fork.  
```git fetch upstream```  
```git checkout -b feature upstream/master```
2. Make changes.
3. Update CHANGELOG.rst (establish new entry and version number if needed).
4. Commit and push feature branch to remote
5. Check all tests pass.
6. Merge to fsl/fsl_mrs:master.
7. Repeat with other features.
8. For new version create new tag -tag name should be "x.x.x" with the message "Vx.x.x". New changelog contents are copied into release notes.
9. This will trigger doc build (published [here](https://open.win.ox.ac.uk/pages/fsl/fsl_mrs/)) and package build (currently broken).
10. Update local master.  
    ```git fetch upstream --tags```  
    ```git checkout master```  
    ```git merge upstream/master```  
    ```git push origin --tags```
11. Trigger manual package build using the [fsl/conda/fsl-mrs:pre-conda](https://git.fmrib.ox.ac.uk/fsl/conda/fsl-mrs:pre-conda) branch. Do this by updating the version number [here](https://git.fmrib.ox.ac.uk/fsl/conda/fsl-mrs/-/blob/mnt/pre-conda/meta.yaml#L6). This will trigger a pipeline from which the conda package can be downloaded (*Download build-noarch-conda-package:archive artifact*).
12. Send to MW for manual upload to the FSL conda channel.

For local development installation run: ```pip install --no-deps -e .```