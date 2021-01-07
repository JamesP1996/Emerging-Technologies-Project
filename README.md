# Emerging-Technologies-Project

# Regression Model based on Power Production from Wind Turbines

The purpose of this project is to create a accurate data prediction model using jupyter notebook and tensorflows keras model that can work out the power outputted from a certain speed in wind turbines.

## Requirements
The requirements and tools used for this program to work are:

1. Jupyter Notebook [https://jupyter.org/install.html]
2. Anaconda / Python.
3. All of the libaries installed through pip or conda from **requirements.txt**
4. Docker with WSL2


## How to Run From Github

To run the Jupyter Notebook on Github, open the Power_Production.ipynb file and all of the code cells should automatically run on the webpage displayed. (The fitting of the regression Model may take a few seconds to complete.)

## How to run the Notebook locally to build the model
Clone this repository to a local directory on your machine and after you have jupyter notebook installed either through *Python Pip* or *Anaconda conda* package managers, run the command
``` jupyter notebook ```
This will display a menu of the items in this directory and open up the **Power_Production_Notebook.ipynb**
and run through each of the cells for the code to work and the Model to be created and saved as **myModel.h5**

## Running the Web Service Locally
If you want to run the web service locally (**webservice.py**) you will need to have Flask installed and run the commands ``set FLASK_APP=webservice.py`` which will set Flasks default application and then run the command ``flask run`` which will launch the web service under your machine's local ip address under port 5000 , eg. **127.168.1.1:5000** or **localhost:5000**

## Running the Web Service through Docker
To build the docker image and have your docker installed application run the web service program in a container. You need to run these two commands to make it work. This may take some time as docker will install it's own variants of the required libaries for python using this reqositories **requirements.txt** file
``` docker
docker build -t flask:latest .
docker run -d -p 5000:5000 flask
```
This will build a docker image onto your docker application.

*Note: Docker must be running before you run these commands and you will probably need a working build with WSL1 or WSL2 for your docker application to function properlly*


## Making Predictions on the Web Service
While the web service is running there are two ways to make predictions.

1. Using the front-end at your local ip as discussed above and entering in a number into the text field and pressing the "Predict Button"
2. Using curl or an application such as postman to send a request to the locally running web service with the body of the request sent through this application being 
``` json
{
    value: "2.5"
}
```
The value can be of any number between the quotations.
This will then send you a response with the predicted value.


**James Porter - G00327095 -<br> Student of Galway Mayo Institute of Technology**

