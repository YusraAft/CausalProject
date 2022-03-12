# CausalProject

## Causal Effect of Class Size on Average Student Math Performance

In this project, I have tried to explore the causal question of whether or not class size which students learn effects the academic performance of students. The main focus is on the notion that smaller class sizes would cause better academic performance in math. Though, we will learn in this project that the data studied does not seem to support that causal relationship hypothesis at this time, but if we account for missing data and with further time, selection bias and such, then we may have better results.

### How to run the provided code?

Firstly, download the folder from GitHub and unzip it in your computer. Then, open (Anaconda) command prompt and create a virtual environnment. Here is a possible command conda create -n causal python=3.8. Then, cd into the directory .../classize/classize and run pip install -r requirements.txt to get the correct libraries donwloaded. Now, you may enter jupyter notebook to open a tab with a list of all the files in jupyter notebook. Finally, open the project_grade4.ipynb and project_grade5.ipynb to run the code. 

You can remove this virtual environment by typing conda env remove -n _______ where __________ is the name you gave to your virtual enviornment. You can see that by doing conda env list.  



### Below is the ReadMe for the data files used in this project. The ReadMe below does not belong to me. 

Angrist and Lavy (1999)

Using Maimonides' Rule to Estimate the Effect of Class Size on Student Achievement

Notes: These programs produce Tables II-V in the published paper. The program mmoulton_post.do implements a Moulton (1986) clustering adjustment for OLS and 2SLS and is used by the other .do files. 

These are STATA translations of the original SAS programs. The switch in software generates slightly different RMSEs.

Programs: 

    * AngristLavy_Table2.do creates Table II in the published paper
    * AngristLavy_Table3.do creates Table III in the published paper
    * AngristLavy_Table4.do creates Table IV in the published paper
    * AngristLavy_Table5.do creates Table V in the publilshed paper
    * mmoulton_post.do implements OLS Moulton corrections - this file should be run in conjunction with the files that create the tables

Data:

    * final4.dta contains data for 4th graders
    * final5.dta contains data for 5th graders 
