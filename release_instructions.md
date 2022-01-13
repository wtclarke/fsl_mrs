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
7. Delete local branch
```git branch -d localBranchName```
8. Repeat with other features.

Each time the fsl/fsl_mrs master branch is updated and the CI pipeline is run, the `trigger-staging-package` job triggers a pipeline on the conda recipe repository to build a package from the current state of master. This can be turned into a development release by running the `deploy-conda-package` on the latest job of `fsl/conda/fsl_mrs`. This will be published to the same channel and have a format `<last_tag>.YYMMDD`.

9. For new version create new tag -tag name should be "x.x.x" with the message "Vx.x.x". New changelog contents are copied into release notes.
10. This will trigger doc build (published [here](https://open.win.ox.ac.uk/pages/fsl/fsl_mrs/)) and package build.
11. Trigger the package build by accepting the relevant merge request on [fsl/conda/fsl_mrs](https://git.fmrib.ox.ac.uk/fsl/conda/fsl_mrs) branch. This will trigger a pipeline from which the conda package be released. Do this by manually triggering the `deploy-conda-package` job after the build has run.
12. New packages will be published to the [FSL conda channel](https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/).
13. To request a version to be included in the next FSL release manually trigger the `update-fsl-manifest` job. This opens a MR for the development team on the [manifest repository](https://git.fmrib.ox.ac.uk/fsl/conda/manifest/).

14. Optionally update local and fork master.  
    ```git fetch upstream --tags```  
    ```git checkout master```  
    ```git merge upstream/master```  
    ```git push origin --tags```

For local development installation run: ```pip install --no-deps -e .```