# Kolmogorov-Arnold model for machine learning

_Accurate and fast regression model based on solid mathematical foundation._

## Summary

The code represents the implementation of the Kolmogorov-Arnold regression model and two model training algorithms. This approach has recently gained popularity under name "KAN: Kolmogorov-Arnold Networks".

The code is written in MATLAB.

The code is a result of the collaborative project between myself and Andrew Polar conducted in 2019-2023. The basic concept of this approach has been published in Ref.[^1] in 2021. Our latest paper, Ref.[^2] published as a preprint in May 2023 (first version), has the most general formulation of the approach. Updated Ref.[^2] published as a preprint in June 2024 (second version) also includes data-driven solution of partial differential equations. The research is summarised in the latest seminar presentation available on YouTube[^4].

The main highlights of the code are:
- the underlying functions of the representation can be either cubic splines or piecewise-linear functions;
- the Gauss-Newton (GN) and the Newton-Kaczmarz (NK) parameter estimation methods are implemented;
- the accelerated Newton-Kaczmarz method (vectorised form using some approximations) is implemented separately;
- the data-driven solution of partial differential equations is implemented;
- the code has minimalist design - it uses only built-in MATLAB functions;
- the core part of the code constitutes fewer than 100 lines.

## Getting started

Main script `mainTriang` runs the code. In the script, flag `modelMethod` selects the model type (splines or piecewise-linear) and the parameter estimation method (GN or NK). The latest addition is the accelerated NK method (see Ref.[^2], section on parallelisation/vectorisation). In the model building functions, flag `printProgr` switches the printout of the progress in the Command Window.

The code builds the model and plots `log(RMSE)` as a function of the number of passes through the data. The model constitutes two matrices with the parameters: `fnB` and `fnT`. 

The obtained model can be used to make a prediction on a new dataset. For the spline version of the model,\
`y = modelKA_basisC( x, xmin, xmax, ymin, ymax, fnB, fnT );`\
should be executed, where `x` is the input data in the same format as in script `mainTriang` and `y` will be the predicted output data. For the piecewise-linear version of the model, the function that makes the prediction is `modelKA_linear` (it has the same format as above).

The computational example is a synthetic dataset - for each record, the inputs are the coordinates of three points in 2D and the output is the area of the triangle that is formed by the points. The points belong to the unit square. The default example takes approximately 14 seconds on a laptop with 11th Gen Intel Core i5.

The example above demonstrates the classical Kolmogorov-Arnold architecture of two layers (corresponding to the inner and the outer functions). In some cases, a multi-layered version gives higher accuracy for the comparable number of parameters. Therefore, training of a three-layered version of the model is demonstrated in script `mainTriangDeep`. Both cubic-spline and piecewise-linear versions are implemented. The three-layered model constitutes three matrices with the parameters: `fnB`, `fnM`, and `fnT`. Having trained the model, a prediction on a new dataset can be obtained using functions `modelKAdeep_basisC` and `modelKAdeep_linear` for cubic-spline and piecewise-linear versions, respectively.

The latest addition (June 2025) is the pre-training algorithm for the three-layered version of the model, see corresponding section in the main scripts. Flag `modelPre` selects either no pre-training (random initialisation), pre-training for the spline basis functions, or pre-training for the piecewise-linear basis functions. The idea of the pre-training algorithm for the three-layered model is training a classical (two-layered) model first, then disregarding the top layer and taking the intermediate variable values as the new inputs, using which another classical (two-layered) model is trained. This gives the initial approximation for three layers, which are then trained in a standard way. Analogously, pre-training for an arbitrary multi-layered version of the model can be done (see video[^4], timestamp 18:30).

Training of a model for a vector output is demonstrated in script `mainMedians`. The computational example is similar to the example above with the triangles, where the inputs are the coordinates of three points in 2D, but in this case, there are three outputs - the lengths of the medians of the triangle that is formed by the points. At the moment, only piecewise-linear basis functions are implemented for this case.

There are three scripts with the units tests: `testBasis` for verifying the derivatives of the basis functions, `testDeriv` for verifying the derivatives of the model output by the inputs and by the parameters, and `testSpline` for comparing the implemented splines with the built-in MATLAB splines.

Data-driven solution of partial differential equations is implemented separately in script `solvePDE_NK`. The computational example is a second-order PDE; the details are given in Ref.[^2], the updated version from June 2024. The default example takes approximately 15 seconds on a laptop with 11th Gen Intel Core i5.

Script `mainMIT` runs the same example as in a recent Python implementation by other researchers (link in the file). It is added only for reproducing the latest benchmark results reported in Ref.[^2].

## Developer and acknowledgements

The code has been developed by Dr Michael Poluektov (University of Dundee, Department of Mathematical Sciences and Computational Physics).

The author would like to acknowledge the great help of Dr Andrew Polar, who contributed equally to the research project and maintains separate implementations of the approach in C# and C++ on his GitHub page[^3].

[^1]: A. Polar and M. Poluektov, _Eng. Appl. Artif. Intell._, 99:104137, 2021, [link](https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742).
[^2]: M. Poluektov and A. Polar, arXiv:2305.08194, 2023, [link](https://arxiv.org/abs/2305.08194).
[^3]: A. Polar, [link](https://github.com/andrewpolar).
[^4]: CRUNCH Group seminar recording, Brown University, 13 June 2025, [link](https://www.youtube.com/watch?v=V0EXlyv3TgI).