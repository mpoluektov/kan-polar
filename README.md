# Kolmogorov-Arnold model for machine learning

_Accurate and fast regression model based on solid mathematical foundation._

## Summary

The code represents the implementation of the Kolmogorov-Arnold regression model and two model training algorithms. This approach has recently gained popularity under name "KAN: Kolmogorov-Arnold Networks".

The code is written in MATLAB.

The code is a result of the collaborative project between me and Andrew Polar conducted in 2019-2024. The basic concept of this approach was published in 2021, Ref.[^1] (open-access preprint on arXiv since January 2020). The most general formulation, which includes data-driven solution of partial differential equations, was published in 2025, Ref.[^2] (open-access preprint on arXiv since May 2023). The research is summarised in the latest seminar presentation available on YouTube[^3].

The main highlights of the code are:
- the underlying functions of the model can be either cubic splines or piecewise-linear functions;
- the Gauss-Newton (GN) and the Newton-Kaczmarz (NK) parameter estimation methods are implemented;
- the accelerated Newton-Kaczmarz method (vectorised form using some approximations) is implemented separately;
- the data-driven solution of partial differential equations is implemented;
- the code has minimalist design - it uses only built-in MATLAB functions;
- the core part of the code constitutes fewer than 100 lines.

## Getting started

There are three main scripts that run the code - model training and inference. They address three separate cases.

Main script `mainTriang` constructs two-layer single-output "classical" model. In the script, flag `modelMethod` selects the model type (splines or piecewise-linear) and the training method (GN or NK). In the model building functions, flag `printProgr` switches the printout of the progress in the Command Window.

The code builds the model and plots `log(RMSE)` as a function of the number of passes through the data. The model constitutes two matrices with the parameters: `fnB` and `fnT`. 

The computational example is a synthetic dataset - for each record, the inputs are the coordinates of three points in 2D and the output is the area of the triangle that is formed by the points. The points belong to the unit square. The default example takes approximately 14 seconds on a laptop with 11th Gen Intel Core i5.

Main script `mainTriangDeep` constructs three-layer single-output model. This model constitutes three matrices with the parameters: `fnB`, `fnM`, and `fnT`.

The a pre-training can be used for the three-layer model. Flag `modelPre` selects either no pre-training (random initialisation), pre-training for the spline basis, or pre-training for the piecewise-linear basis. The pre-training algorithm consists in training a classical (two-layer) model first, disregarding the top layer, and taking the intermediate variables as the new inputs, using which another classical (two-layer) model is trained. This gives the initial approximation for the three layers.

Main script `mainMedians` constructs two-layer vector-output model. The computational example is similar to the above: the inputs are the coordinates of three points in 2D and the outputs are the lengths of the medians of the triangle that is formed by the points. At the moment, only piecewise-linear basis functions are implemented for this case.

There are three scripts with the units tests: `testBasis` for verifying the derivatives of the basis functions, `testDeriv` for verifying the derivatives of the model output by the inputs and by the parameters, and `testSpline` for comparing the implemented splines with the built-in MATLAB splines.

Data-driven solution of partial differential equations is implemented separately in script `solvePDE_NK`. The computational example is a second-order PDE (see latest paper for details). The default example takes approximately 15 seconds on a laptop with 11th Gen Intel Core i5.

## Developer and acknowledgements

The code has been developed by Dr Michael Poluektov (current affiliation - School of Computing and Mathematical Sciences, University of Greenwich, UK).

The author would like to acknowledge the great help of Dr Andrew Polar, who contributed equally to the research project and maintains separate implementations of the approach in C# and C++ on his GitHub page[^4] and website[^5].

[^1]: A. Polar and M. Poluektov, _Eng. Appl. Artif. Intell._, 99:104137, 2021, [link](https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742).
[^2]: M. Poluektov and A. Polar, _Mach. Learn._, 114:185, [link](https://link.springer.com/article/10.1007/s10994-025-06800-6).
[^3]: CRUNCH Group seminar recording, Brown University, 13 June 2025, [link](https://www.youtube.com/watch?v=V0EXlyv3TgI).
[^4]: A. Polar, GitHub page [link](https://github.com/andrewpolar).
[^5]: A. Polar, website [link](http://openkan.org).
