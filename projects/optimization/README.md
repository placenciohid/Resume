# Capacity-constrained scheduling

**Question:** which procedures should fit into two rooms with four hours available in each?

`python projects/optimization/demo.py`

The standard-library demo enumerates all `3^8 = 6,561` assignments of eight synthetic procedures. Each can be deferred or assigned to one room. The objective maximizes total priority subject to each room's four-hour capacity and each procedure being assigned at most once. It compares the exact optimum with a first-fit baseline using the same fixed input order. The scores are dimensionless toy priorities, not clinical outcomes or measured savings.

This is a capacity-allocation demonstration, not a deployable hospital scheduler. It excludes staff, equipment, patient dependencies, and within-day start times. Enumeration is exact for this small instance but will not scale to production sizes.

Original Julia constraint-programming and Pyomo notebooks are retained in `legacy/`. The original operating-room notebook needs a source workbook and GLPK installation absent from this repository. The portable demo illustrates the optimization decisions without asserting that it reproduces that notebook's results.

[Original Julia article](https://placenciohid.github.io/2023/07/13/Julia_Optimization.html)
