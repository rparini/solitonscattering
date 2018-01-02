A work in progress Python library to aid in the numerical examination of the scattering of solitons in classical integrable field theories with localised non-integrable deformations such as boundaries or impurities.  

The idea, as in [1], is that the time evolution of the soliton/deformation collision the soliton content of the field in the remaining integrable region can be found by computing part of the scattering data assocated with the inverse scattering method for the relevant integrable model.

Presently, the focus is on updating and then generalising the software used to produce many of the numerical results in [1]. 

### Installation
 [Download](https://github.com/rparini/solitonscattering/archive/master.zip), unzip and navigate to the folder `solitonscattering-master`.
 From there install from the command line with:
 ```bash
 pip install -r requirements.txt
 python install setup.py
 ``` 

### Examples
* [Sine-Gordon soliton colliding with a Robin boundary](examples/Robin.ipynb)

### References
[1] Robert Arthur, Patrick Dorey, Robert Parini ["Breaking integrability at the boundary: the sine-Gordon model with Robin boundary conditions"](https://doi.org/10.1088/1751-8113/49/16/165205), *Journal of Physics A*, Volume 49, Number 16, 2016,  [(ArXiv:1509.08448)](https://arxiv.org/abs/1509.08448)
