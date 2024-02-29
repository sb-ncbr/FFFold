[FFFold](https://fffold.biodata.ceitec.cz/) is a web application for the local optimisation of protein structures predicted by the [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2) algorithm and deposited in the [AlphaFoldDB](https://academic.oup.com/nar/article/50/D1/D439/6430488) database. The structures are optimised by the physics-based generic force field [GFN-FF](https://onlinelibrary.wiley.com/doi/full/10.1002/anie.202004239) accelerated by a divide-and-conquer approach which results are comparable to the optimisation of whole protein structure with constrained α-carbons. Thus, FFFold optimises in particular the bond lengths and angles and describes the interactions between nearby residues. Before computation of the charges, input protein structures are protonated by [PROPKA3](https://pubs.acs.org/doi/full/10.1021/ct100578z). The details about the methodology and usage are described in the [manual](https://github.com/sb-ncbr/FFFold/wiki). This website is free and open to all users and there is no login requirement.

## How to run

To run FFFold locally, you will need to have [Python 3.11](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installing/) installed. In addition, optimisation software [xtb](https://xtb-docs.readthedocs.io/en/latest/index.html) is needed, which can be obtained via [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/#).

Then, clone the project and install the project dependencies by running:

```bash
$ cd /opt
$ git clone --recurse-submodules --depth 1 https://github.com/sb-ncbr/FFFold
$ sudo python3.9 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```
Run the project by running the following command inside the virtual environment:

```bash
(venv) $ cd /opt/FFFold/app
(venv) $ export FLASK_APP=routes.py
(venv) $ flask run
```
and point your browser to localhost:5000/.

## License
MIT
