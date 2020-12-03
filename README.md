# Wine Quality Prediction Using Pyspark and Amazon Web Services (AWS)
This giude explains the procedure to use AWS services to train ML (Machine Learning) model on multiple parallel EC2(Elastic Compute Cloud) instances. The ML program is written in python using Apache Spark MLlib libraries. The training and prediction programs are configured to run inside a container. 

The Python programs used in this project can be found in *python_code* folder


The Dockerfile for containers can be found in *test_docker* and *train_docker* folders

The Datasets used can be found in *dataset* folder

The links for the Docker Container uploaded on Docker hub can be found in [Links](#links)
 
# Links
- **Github link** - https://github.com/Kruthika928/wine-quality-prediction-pa2 
 
- **Docker container for training** : [kruthika547nayak/winetrain:latest](https://hub.docker.com/repository/docker/kruthika547nayak/winetrain)

- **Docker container for testing** : [kruthika547nayak/winetest:latest](https://hub.docker.com/repository/docker/kruthika547nayak/winetest)

# Table of Contents
1) [Setting up EC2 Cluster on AWS](#setting-up-ec2-cluster-on-aws)
2) [Setting up Task Definitions and Tasks](#setting-up-task-definitions-and-tasks)
3) [Running the Prediction Application on AWS with Docker](#running-the-prediction-application-on-aws-with-docker)
4) [Running the Prediction Application without Docker](#running-the-prediction-application-without-docker)
5) [Using WinSCP to transfer data](#using-winscp-to-transfer-data)


## Setting up EC2 Cluster on AWS

To run the ML container application for training on multiple parallel EC2 instances, a cluster needs to be set up. The steps given below are followed to create a cluster with 4 instances.

- In AWS Management Console search for **Elastic Container Service (ECS)** and click on it. In ECS Console, select **Cluster** and click on *"Create Cluster"*

- In  **Select cluster template** click on *"EC2 Linux + Networking"* , because we will be using Amazon Linux 2 image for managing ECS tasks.
- In **Configure cluster** modify the following parameters and keep the rest as it is.

    __*Cluster name*__ : *wine-quality-train-cluster*
    
    __*Provisioning model*__ : *On-Demand Instance*
    
    __*EC2 instance type*__ : *t2.micro*
    
    __*Number of instances*__ : *4*
    
    __*Key Pair*__ : *Choose an appropriate key pair*
    
    __*Security group inbound rules (Port range)*__ : *22-80* 
- Click on *"Create"*  to create a cluster 


 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/aws-cluster1.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/aws-cluster2.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/aws-cluster4.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/aws-cluster3.JPG?raw=true" width="666">
 
 
Once the cluster is created 4 EC2 instances registered to the cluster are generated. This can be verified on EC2 dashboard and checking the number of EC2 instances running.

## Setting up Task Definitions and Tasks
We have successfully created a cluster but there is no task running on our instances. In order to run our ML container application we need to create a "Task Definition" which describes information regarding the containers to use, bind mounts, volumes etc. These are done as follows: 
- In **ECS console** select *"Task Definitions"*
- Click on *"Create New task Definition"*
- Choose **Select launch type compatibility** as "EC2"

This will open up **Configure task and container definitions** screen where we have to configure the parameters for our docker conatiner to run correctly. The ML conatiner application used for training outputs 2 files,  *Modelfile*  and *reults.txt*. The *Modelfile* is used in the prediction application and *results.txt* gives metrics of the training model against ValidationDataset.csv. A bind-mount is required between the Docker container and the host to access these files. Do the following configuration 
- __*Task Definition Name*__ : *wine-quality-train-task*
- __*Task Role*__ : *ecsTaskExecutionRole*

The following is done so that the files generated by the docker application is available to the host.
Under __*Volumes*__ click on *"Add volume"* 
- __*Name*__ : *host-path*
- __*Volume type*__ : *Bind Mount*
- __*Source path*__ : */home/ec2-user*

Now we will configure the ML container for training 
Under __*Container Definitions*__ click on *"Add container"* 
- __*Container Name*__ : *wine-train-container*
- __*Image*__ : *kruthika547nayak/winetrain:latest*
- __*Memory Limits*__ : *Soft Limit, 512

Under __*Mount points*__ ,  
- __*Source volume*__ : *host-path*
- __*Container path*__ : */job*
- Click on *"Add"* and then click on *"Create"*

 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/task-definition-1.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/task-definition-2.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/task-definition-3.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/task-definition-4.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/task-definition-5.JPG?raw=true" width="666">
 
We have successfully created our *"Task Definition"* , now we have create a Task which will initiate our docker container application on the EC2 instances. Do the following to run the task:
- On ECS console click on **Cluster**
- Select the cluster recently created (wine-train-quality-cluster)
- Select **Tasks** tab and click on *"Run new Task"*
- Select **Launch type** as *"EC2"* and select **Number of Task** as *"4"*
- Click on *"Run Task"*

This will download ML conatiner application for training on EC2 instances if not present and starts executing it.
Once the execution is completed two files named *"Modelfile"* and *"results.txt"* should be present in the home directory (/home/ec2-user) of all the running EC2 instances. This *"Modelfile"* can be downloaded from any one these instances which will be used in the prediction application. Instructions to download using WinSCP is given at ?????

## Running the Prediction Application on AWS with Docker
The ML container for prediction uses 2 files as input  *"Modelfile"*, *"TestDataset.csv"* .
The container application takes the *"TestDataset.csv"* as input and applies the model from *"Modelfile"* and generates a csv file with prediction output. The detailed steps are explained below.

### Launching an instance on AWS
Ubuntu instance is launched as follows :
- Go to **EC2 dashboard** and click on *"Launch instances"*
- In **Choose an Amazon Machine Image (AMI)** select *"Amazon Linux 2 AMI (HVM), SSD Volume Type"* 
- In ***Choose an Instance type*** select *"t2.micro"* and click on *"Review and Launch"*
- CLick on *"Launch"*
- Create a New key pair or choose an existing one and click on *"Launch"*

### Installing Docker and downloading the prediction container
In order to run the prediction conatiner docker package is required. To install docker in Amazon linux machine the following steps are done
- Run the following command
```Console
    $ sudo yum install docker -y && sudo systemctl start docker
```
- The docker is pulled with the following command 
```Console
    $ sudo docker pull kruthika547nayak/winetest:latest
```
- Run the following command to verify if the container has been installed
```Console
    $ sudo docker images
```

### Running the docker container 
After installing the image, two files must be present in order to run the container application properly (*Modelfile* and *Inputdataset.csv*). These two files can be uploaded on to the instances using WinSCP. To run the container use the following command
```Console
  $ sudo docker run -v /home/ec2-user/:job kruthika547nayak/winetest:latest TestDataset.csv
```
where ,
`/home/ec2-user` is the path to the home directory in the instance.
`/job` is the path mapped inside the conatiner
`kruthika547nayak/winetest:latest` is the name of prediction docker conatiner
`TestDataset.csv` is the name of the input file for prediction testing.

*Note: For the above command to work the Modelfile and TestDataset.csv must be present at /home/ec2-user*

## Running the Prediction Application without Docker
To run the prediction application without docker, the following packages are needed
- [Pyspark](https://pypi.org/project/pyspark/)
- [JAVA JDK](https://www.oracle.com/java/technologies/javase-jdk13-downloads.html)
- [numpy](https://pypi.org/project/numpy/)
- [Apache Spark(spark-3.0.1-bin-hadoop2.7.tgz)](https://spark.apache.org/downloads.html)

### Java Installation
- Download Java JDK from here [link](https://www.oracle.com/java/technologies/javase-jdk13-downloads.html)
- Go to downloads folder and run the following command on console
```Console
$ sudo dpkg -i jdk-13.0.2_linux-x64_bin.deb
```
- Run `java --version` on console to verify if java is installed. The following output is expected:

 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/java-version.JPG?raw=true" width="666">

- To setup environment variables for java, include the following in  `/etc/environemnt` file as shown below using any text editor (i.e nano, gedit)
```Console
 $ JAVA_HOME=/usr/lib/jvm/jdk-13.0.2
```
- Finally source the `/etc/environment` file, 

```Console
 $ source /etc/environment
```

### Installing Apache Spark
- Go to [Spark Website](https://spark.apache.org/downloads.html)
- Select Spark Release Version as 3.0.1 and download the .tgz file

 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/spark-tar.jpg?raw=true" width="666">
 
- Go to the downloads folder and extract the tgz file using the following command
```Console
  $sudo tar -xvzf spark-3.0.1-hadoop2.7.tgz 
```

- To setup environment variables for pyspark, include the following `~/.bashrc` file
```Console
 export SPARK_HOME=~/Downloads/spark-3.0.1-bin-hadoop2.7
 export PATH=$PATH:$SPARK_HOME/bin
 export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
 export PYSPARK_PYTHON=python3
 export PATH=$PATH:$JAVA_HOME/jre/bin
```
Image for reference:

<img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/bashrc.jpg?raw=true" width="666">

- Finally source `./bashrc` file or restart `console` for the variables to get updated

```Console
  $source ~./bashrc
 ```
- Run `pyspark` in console to verify the installation. You should get the following output. 

<img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/spark-console.jpg?raw=true" width="666">


### Running prediction application
- Make sure Modelfile is present before running the *wine_test_nodocker.py* file. In case it is not present run the *wine_train_nodocker.py* file to generate the same. This can be done by executing the following command
```Console
 $ python3 wine_train_nodocker.py
```
- Run the prediction app using this command
```Console
 $ python3 wine_test_nodocker.py TestDataset.csv
```
After the command is executed successfully, two files will be generated in the directory, `Results.txt` and `Resultdata` folder containing csv file.


## Using WinSCP to transfer data
In order to transfer data between the instance and the local PC (i.e Modelfile, results.txt, InputDataset.csv) we make use of program called [WinSCP](https://winscp.net/eng/download.php). The following are pre-requisites for using WinSCP
1. ppk key registered to the instance
2. WinSCP program itself

The following shows the step by step approach to configure and upload Modelfile to the instance. The same can also be used to download data from the instance.

 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/winSCP-login.JPG?raw=true" width="450">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/winSCP-login1.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/winSCP-login2.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/winSCP-login3.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/winSCP-login4.JPG?raw=true" width="666">
 
 <img src="https://github.com/Kruthika928/wine-quality-prediction-pa2/blob/main/images/winSCP-login5.JPG?raw=true" width="500">
 
 The above image is expected if all the configurations are done properly
 
 *Note : Ensure port 22 is set as an inbound rule for proper functioning of WinSCP*
 
 The left hand side denotes the directory structure of the local PC and the right hand side denotes the directory structure of the Amazon Linux instance.
 Files can be transferred by dragging and dropping from either side.







    
