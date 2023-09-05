fMRS Analysis
=============

FSL-MRS includes the :code:`fmrs_stats` script to:

1. form contrasts and combine correlated peaks at the first-level of GLM analysis,
2. perform higher-level group analysis, and,
3. form contrasts on the higher-level. 

The higher-level analysis uses the *FLAMEO* tool packaged in FSL (which is also used for higher-level FSL FEAT analysis).

To understand the use of higher-level GLM analysis please see the `FSL documentation <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/GLM/CreatingDesignMatricesByHand>`_ and `FSL course <https://open.win.ox.ac.uk/pages/fslcourse/website/online_materials.html>`_, specifically the `FMRI2 E2 video <https://www.youtube.com/watch?v=-nf9Hcthnm8>`_.

Using fmrs_stats
----------------
On the command line :code:`fmrs_stats` can be called as shown in the example below. This example carries out all the steps described above. Steps, either forming first-level contrasts, or higher-level contrasts, can be omitted. This documentation will take you through this example, which performs a paired t-test on two groups of fMRS acquisitions acquired from the same subjects. This example is taken from the fMRS demo available from the FSL-MRS team.

.. code-block::

    fmrs_stats\
        --data results_list\
        --output group_stats\
        --fl-contrasts fl_contrasts.json\
        --combine NAA NAAG\
        --combine Cr PCr\
        --combine PCh GPC\
        --combine Glu Gln\
        --hl-design design.mat\
        --hl-contrasts design.con\
        --hl-contrast-names "STIM>CTRL" "CTRL>STIM"\
        --hl-covariance cov_split.mat\
        --hl-ftests design.fts\
        --overwrite

This example:

1. Forms a first-level contrast based on the contents of the :code:`fl_contrasts.json` file,
2. Sums (at the first-level) the NAA peaks, the creatine peaks, the choline peaks, and glutamine and glutamate,
3. Outputs the modified first level results to a new :code:`group_stats` result directory,
4. Then using the supplied design matrix (:code:`design.mat`), contrasts matrix (:code:`design.con`), and f-tests matrix (:code:`design.fts`) uses FLAMEO to perform the higher-level analysis, and,
5. The group level GLM statistics are then output to the :code:`group_stats` result directory.

To achieve this the user must provide a number of input files to the script.

Inputs to fmrs_stats
~~~~~~~~~~~~~~~~~~~~

:code:`--data results_list`
    A list of directories containing first-level results created using fsl_dynmrs. These can be listed directly on the command-line or as a list in a text file (with a directory on each separate line). e.g. for three subjects

    .. code-block::

        sub0_stim\
        sub1_stim\
        sub2_stim
        sub0_ctrl\
        sub1_ctrl\
        sub2_ctrl
        

:code:`--fl-contrasts fl_contrasts.json` 
    A JSON formatted file describing contrasts formed at the first level by linearly combining existing parameters.

    Here we will combine two GLM betas (which correspond to blocks of activation) to give a mean activation contrast. We assign a name and to take the mean rather than the sum we pass the scales :code:`[0.5, 0.5]`.

    .. code-block::

        [
            {
                "name": "mean_activation",
                "betas": ["beta0", "beta1"],
                "scale": [0.5, 0.5]
            }
        ]

    Multiple contrasts can be listed and created.

:code:`--combine NAA NAAG`
    The :code:`--combine` option sums the betas of the peaks listed after the command. In this example betas from NAA and NAAG will be combined. This command can be repeated for multiple combinations. The option works in concert with the :code:`--fl-contrasts` option, taking all parameter covariances into account.

:code:`--hl-design design.mat` 
    Pass the higher-level design matrix formatted as a `VEST <MRSpectroscopyStorage>`_ formatted file. This is created by forming a simple text file containing the design matrix for a three subject, two-case paired t-test. This is the equivalent input to the :code:`flameo --dm,--designfile` option. 

    .. code-block:: 

        1  1  0  0
        1  0  1  0
        1  0  0  1
        -1  1  0  0
        -1  0  1  0
        -1  0  0  1

    Subsequently this can be converted to the VEST format by running the packaged :code:`Text2Vest` tool.

    .. code-block::

        Text2Vest design.txt design.mat

:code:`--hl-contrasts design.con`
    As for :code:`--hl-design` but for the contrast matrix. I.e. equivalent to the :code:`flameo --tc,--tcf,--tcontrastsfile` option. 

    .. code-block:: 

        1 0 0 0
        -1 0 0 0

    This must also be formatted as a VEST file.

    .. code-block::

        Text2Vest contrast.txt design.con


:code:`--hl-contrast-names "STIM>CTRL" "CTRL>STIM"` 
    A name for each defined higher-level contrast (each row) can be defined. Here we name the two contrasts of the paired t-test.

:code:`--hl-covariance cov_split.mat`
    For groups with different variances, this file can be used to assign to different covariance groups. Equivalent to the :code:`flameo --cs,--csf,--covsplitfile` option. 

    Defaults to a single group for all first-level results.

    This must also be formatted as a VEST file.

    .. code-block::

        Text2Vest cov_split.txt cov_split.mat