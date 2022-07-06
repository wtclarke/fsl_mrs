.. _basis_tools:

Basis Spectra Manipulation
==========================

basis_tools
-----------
The :code:`basis_tools` script provides the user with tools for manipulating basis spectra in the FSL-MRS JSON format.

Provide one of the following subcommands with :code:`basis_tools` to convert, scale, add to, difference, or shift basis spectra.

info
****
| *Example* :code:`basis_tools info path/to/my/basis`
| Provides a short summary of the contents of the basis set.

vis
***
| *Example* :code:`basis_tools vis path/to/my/basis --ppmlim 0.2 4.2`
| Provides visualisation of the basis set.

convert
*******
| *Example* :code:`basis_tools convert path/to/my/lcmbasis.BASIS path/to/my/fslbasis`
| Convert LCModel (.Basis), LCModel (directory of .raw) or JMRUI format basis sets to FSL-MRS (.json) format.
| Note that the bandwidth and fieldstrength must be supplied manually to the CLI for the .raw format.

add
***
| *Example* :code:`basis_tools add --scale --name my_new_basis my_new_basis.json path/to/my/fslbasis`
| Add a json formatted basis spectrum to an existing basis set.

shift
*****
| *Example* :code:`basis_tools shift path/to/my/fslbasis NAA 1.0 path/to/my/edited_fslbasis`
| Shift a basis spectrum on the chemical shift axis.

scale
*****
| *Example* :code:`basis_tools shift path/to/my/fslbasis NAA path/to/my/scaled_fslbasis`
| Rescale a basis spectrum to the mean of all other basis spectra (or to specified target :code:`--target_scale`.

diff
****
| *Example* :code:`basis_tools diff --add_or_sub sub mega_on mega_off mega_diff`
| Form a basis set for a difference method using two other basis set. Add or subtract using :code:`--add_or_sub {'add'|'sub}`.

add_set
*******
| *Example* :code:`basis_tools add_set --add_MM basis_without_mm/ basis_with_default_mm/`
| Add a (predefined) set of Gaussian peaks to a basis set. Three defualt sets are defined:
| 1) The FSL 'default' with peaks at 0.9, 1.2, 1.4, 1.7 ppm and a linked set at 2.08 & 3.0 ppm
| 2) The (experimental) MEGA edited MM peaks at 0.915 & 3.0 ppm (ratio of 3.75:2.0).
| 3) A water peak at 4.65 ppm
| The other options (:code:`--gamma --sigma`) allow a custom set of peaks to be specified.
