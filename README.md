# Time Plan
For the majority of the following tasks, we plan on working together simultaneously. If the distribution
of tasks is required we will indicate this in the final report.
1. 7.12: Ensure assignment 3 is complete in order to build on top of the GPU implementation of
exercise 4.
2. 7.12: Setup repository in such a way that it can be easily run locally and on Google Colab
3. 14.12: Examine input and output format. Investigate if the output can be used as input in the
next step.
4. 14.12: Implement lower precision interpolation
5. 1.1: Run short tests to ensure functionality.
6. 1.1: Evaluate implementation and write scripts to evaluate performance over many different
simulations
7. 12.1: Submit Report and Present

TODO: 
1. Test converting between data types within kernel.
    Foreach variable, create half variant.
    Convert float -> respective variant
    do calculations
    convert half variant -> originial float
2. Test converting data types before data copying.
    Create alternative structs with halfs, convert the original structs
    alloc/memcopy
    launch
    memcopy()
    convert from half structs back to original float versions.
3. Evaluate the trade offs between faster conversion by parallelization, or faster memcopy but slower conversion.
4. Try if that trade off is significantly different without redundant allocs/memcopies.
    Create functions for alloc/memcopy
    Run once before mover loop in sputniPIC.cpp
    dealloc after.
    Is the time different? Observations?
5. Test if it makes a difference between 2D/3D.
6. Write report

