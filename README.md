1D-Var librairy (data assimilation lab)
=======================================

Martin Deshaies-Jacques

[deshaies.martin@sca.uqam.ca](mailto:deshaies.martin@sca.uqam.ca)

[www.science.martn.info](http://www.science.martn.info)

Use, share, remix!
...and criticize

Test-it with [pyfKdV](https://github.com/martndj/pyfKdV)


About Data Assimilation concepts
--------------------------------


 * [F. Bouttier, P., Courtier, Data assimilation concepts and methods, ECMWF Meteorological Training Course Lecture Series, March 1999](http://msi.ttu.ee/~elken/Assim_concepts.pdf)

Installation informations
-------------------------

### Python Dependencies
 * pseudoSpec1D (found in [pyKdV](https://github.com/martndj/pyfKdV))
 * Numpy
 * Matplotlib
 * Scipy

### Instructions
 * ./ refer to the root of dVar installation;
 * [something] means optional arguments;
 * \<something\> means you must replace it with what is appropriate to your environment.

 1. On Linux OS Debian/Ubuntu, you can install it running

        [sudo] apt-get install python python-matplotlib python-numpy python-scipy

 2. Add the path to the python module dVar to your PYTHONPATH environment variable, in Linux bash environment you can do it running.
 
        export PYTHONPATH=${PYTHONPATH}:<path to ./>

    Adding this export line to your startup script (.bashrc or .profile) is a way to do it.

 3. Link the data assimilation lab with a 1+1D model deriving from the Launcher and TLMLaucher class from pseudoSpec1D (contained in [pyfKdV](https://github.com/martndj/pyfKdV)).
 Example scripts will come soon!
