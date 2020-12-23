# CoMTE
Counterfactual Explanations for Multivariate Time Series Data


Authors:
    Emre Ates (1), Burak Aksar (1), Vitus J. Leung (2), Ayse K. Coskun (1)

Affiliations:
    (1) Department of Electrical and Computer Engineering, Boston University
    (2) Sandia National Laboratories

This work has been partially funded by Sandia National Laboratories. Sandia
National Laboratories is a multimission laboratory managed and operated by
National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of
Energy’s National Nuclear Security Administration under Contract DENA0003525.

## Requirements
These files use data sets that are uploaded to Zenodo, at the [URL](https://doi.org/10.5281/zenodo.3760027)

The code assumes that the data is located at ./data

A python 3.x installation is required, as well as the packages inside
`requirements.txt` and the `fast_features` package. Instructions for `fast_features` package are inside the fast_features directory.

```
pip3 install --user -r requirements.txt
```

The jupyter notebooks as well as `faithfulness.py` and `robustness.py` are entry points for different
experiments in the paper, and the remaining files are shared code for data
loading, explainability methods, or machine learning methods.

## Usage
For faithfulness experiments, run `./faithfulness.py --dataset [hpas,
taxonomist, natops, test]`

For robustness experiments, first train using `./robustness.py --dataset [hpas,
taxonomist, natops, test] --method rf --train --outdir ./robustness`
and then calculate robustness using `./robustness.py --dataset [hpas,
taxonomist, natops, test] --method rf --outdir ./robustness --partition [0-50]`
Robustness typically takes a very long time (over 24 hours) even with partitions.

The partition is optional, and only calculates 1/50 of the results. The outdir
argument is used to specify where the trained classifier and results are placed.

The jupyter notebooks can also be executed. Only the generalizability notebook takes a
significant amount of time to execute however we provide pickle files so results can be easily regenerated.

