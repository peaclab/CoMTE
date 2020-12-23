# CoMTE
Counterfactual Explanations for Multivariate Time Series Data

Maintainer: 
* **Burak Aksar** - *baksar@bu.edu* 

Developers:  
* **Emre Ates** - *ates@bu.edu*  & **Burak Aksar** - *baksar@bu.edu* 


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

To be updated


## Authors

[Counterfactual Explanations for Machine Learning
on Multivariate Time Series Data](https://arxiv.org/pdf/2008.10781.pdf)

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

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details



