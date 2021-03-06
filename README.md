# Scalable Variational Inference for Stochastic Differential Equations

State and parameter estimation in dynamical systems based on sparse, discrete observations is a topical yet challenging problem. 
Traditional methods suffer from extremely high computational costs due to the need to carry out explicit numerical integration after parameter adaptation.
In contrast, the recently proposed gradient matching with Gaussian processes model is a promising tool.
It is a grid-free inference technique that also eliminates the dependency on numerical integration.
However, due to the intractability of the posterior, approximate inference techniques must be used.
Sampling-based solutions fall short in this case since most real-world dynamical systems are high-dimensional.
On the other hand, variational approaches have shown their potential both in terms of prediction accuracy and runtime performance, even in situations when the system is only partially observed.
Extending the state-of-art variational gradient matching framework, this work further improves the flexibility of the inference algorithm by relaxing the structural assumption on the dynamical systems.
Support for positivity constraints on the states and parameters that are common in many biochemical and physical applications is also introduced. 
Finally, a highly efficient parallel solution is devised to address problems involving stochastic differential equations.