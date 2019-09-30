# ForeCAT
Main file ForeCAT.py calls the other three files.   
1. CME_class - sets up and updates a CME object that holds all the information about the grid representing the CME as well as the properties of the CME  
2. ForceFields - uses positional information of CME to calculate magnetic fields/forces
3. ForeCAT_functions - various other programs for printing out information or modeling the solar wind density  
  
To minimize the effects of accidental tinkering, for most purposes ForeCAT.py is the only one that needs to be edited - the CME mass, velocity, and angular width profiles can all be changed here.  For consistency the same set of other CME_class/GPU_functions/ForeCAT_functions can be used for all runs and individual ForeCAT.py modified for each case.  
  
ForeCAT.py pulls in initial conditions from a .txt file in the format shown here
