# Library for manipulating Matrix Product States.

## Features:
 - Canonicalisation, evs, etc.
 - Finite and infinite chain
 - Time evolution via TDVP
 - MPS lyapunov spectra

## Installation Instructions
Clone this repository somewhere and run 
`pip install -e .` from the project root.
Run tests with 
`python setup.py test`

## Conventions
An MPS object (either fMPS or iMPS) contains a list of rank-3 tensors (under mps.data).
Tensors are stored with their indices as (physical, virtual left, virtual right).
