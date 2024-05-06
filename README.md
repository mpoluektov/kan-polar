# Kolmogorov-Arnold model for machine learning

_Accurate and fast regression model based on solid mathematical foundation._

## Summary

The code represents the implementation of the Kolmogorov-Arnold regression model and two methods of the model identification. This approach has recently gained popularity under name "KAN: Kolmogorov-Arnold Networks".

The code is written in MATLAB.

The code is a result of the collaborative project between myself and Andrew Polar conducted in 2019-2023. The basic concept of this approach has been published in Ref.[^1] in 2021. Our latest paper, Ref.[^2] published as a preprint in May 2023, has the most general formulation of the approach. The main highlights of the code are:
- the underlying functions of the representation can be either cubic splines or piecewise-linear, depending on the desired accuracy and the desired training time;
- two identification methods are implemented - the Gauss-Newton and the Newton-Kaczmarz methods;
- the code has minimalist design, it uses only built-in MATLAB functions;
- the core part of the code is less than 100 lines.

## Getting started

The user can run the code by executing the main file: `mainTriang;`

The computational example is a synthetic dataset - for each record, the inputs are the coordinates of three points in 2D and the output is the area of the triangle that is formed by the points. The points belong to unit square. The model is built and the RMSE as a function of the iteration number is plotted.

## Developer and acknowledgements

The code has been developed by Dr Michael Poluektov (University of Dundee, Department of Mathematical Sciences and Computational Physics). 

The author would like to acknowledge the great help of Dr Andrew Polar, who contributed equally to the research project and maintains a separate implementation of the approach in C# on his GitHub page. 

[^1]: A. Polar and M. Poluektov, _Eng. Appl. Artif. Intell._, 99:104137, 2021, [link](https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742).
[^2]: M. Poluektov and A. Polar, arXiv:2305.08194, 2023, [link](https://arxiv.org/abs/2305.08194).