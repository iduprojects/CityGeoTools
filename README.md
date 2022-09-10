## CityGeoTools
**CityGeoTools** is a python package of methods developed for analysis of the urban environment. These  methods could be useful for urban researchers and local governments for making informed decisions in the field of urban economy. At this moment the following methods are available: 
   
1.   Traffic calculation (between houses and stops)
2.   Visibility analysis (from certain point in the city)
3.   Weighted voronoi (to identify unserved zones, based on [research](https://www.sciencedirect.com/science/article/pii/S187705092032384X))
3.   Blocks clusterization (based on presented services)
4.   Service clusterization (based on services location)
5.   Spacematrix clusterization (based on [research](https://elibrary.ru/item.asp?id=45845752))
6. Availability isochrone (based on [research]())
7.  Diversity analysis (based on available services) - in progress for user mode
8.  Provision population with services - in progress for user mode
9.  Wellbeing (based on provision with services needed in living situations) - in progress for user mode

## How to use CityGeoTools
More detailed descriptions and examples of how to use the methods are provided as *jupyter notebooks* in [examples](https://github.com/iduprojects/CityGeoTools/tree/metrics-refactor/notebook_examples). In order to use these methods, it is not necessary to have a deep knowledge of python programming and machine learning. Simply follow the steps shown in the figure below. Two ways of using the presented methods are implemented - **general mode** and **user mode**.
  
![Image](https://github.com/iduprojects/CityGeoTools/blob/metrics-refactor/img/plot.png?raw=true)

It should be noted that we do not publish the package in a PyPI repository at this moment. Therefore, to work with this package, first you need to download the repository to the local machine:
```shell
git clone https://github.com/iduprojects/CityGeoTools
```

## Data preparation
We leave the responsibility for collecting and preparing data to the user. In the absence of reliable sources of information about the urban environment, it is recommended to use OpenStreetMap and [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) to get datasetes. Transport graphs needed for some methods are an exception. We provide the code for building a walk graph, a car travelling graph and a public transport graph separately as well as an intermodal graph (that includes walk routes, driveways and public transport routes) in the file [get_graphs.py](https://github.com/iduprojects/CityGeoTools/tree/metrics-refactor/data_collecting).

 To use implemented methods, the collected datasets **MUST** match specifications presented in the folder  [data_specification](https://github.com/iduprojects/CityGeoTools/tree/metrics-refactor/data_specification) regardless of the selected mode (general or user mode).
  
 ## Data imputation
The biggest problem in data preparation is the incompleteness of information about the urban environment  — the datasets often contain missing values. Before using the presented methods, it is necessary to deal with the missing values to obtain reliable results. Instead of deletion or filling the missing data with any point statistics, we suggest using our data imputer based on geospatial component of urban data. The data imputation algorithm and information about the accuracy of the imputed data are presented in the [paper](https://link.springer.com/chapter/10.1007/978-3-031-08757-8_21).
  
The figure below shows comparison of the accuracy of the imputed values obtained with the developed method (green line), a method from scikit-learn package (black line) and mean imputation (gray line) in building features such as number of storeys and population.  
![Image](https://github.com/iduprojects/CityGeoTools/blob/metrics-refactor/img/imputer.jpg?raw=true)  
  
Imputation options are set in three configuration files (more information about customizing options in the [examples](https://github.com/iduprojects/CityGeoTools/tree/metrics-refactor/notebook_examples)).
+ *config_imputation.json* contains general options (such as number of iterations over dataset features, number of imputations, initial imputation strategy and neighbors search parameters). 
+ *config_learning.json* sets a pipeline of data preprocessing and a grid of hyperparameters for certain machine learning methods that are going to be optimized inside the HalvingGridSearch algorithm.
+ *config_prediction.json* contains a path to earlier fitted models that can be used to predict missing values.

After customizing options in configuration files, you need to import DataImputer class, set path to a dataset which contains missing values, extend the dataset with aggregated neighbors features and call multiple imputation.   
```python
from data_imputer.imputer import DataImputer

dataset_path = "data_imputer/simulations/living_building_spb_130822_170502.geojson"
imputer = DataImputer(dataset_path)
imputer.add_neighbors_features()
full_data = imputer.multiple_imputation(save_models=True, save_logs=True)
```
If you already have fitted model, you can simply specify the path to the models and make multiple imputation the following way:
```python
full_data = imputer.impute_by_saved_models()
```

## General mode
General mode provides access to all data stored in databases through a data query interface and allows the use of all of the developed methods through API. This mode makes it possible to build an application for urban environment analysis or integrate the methods into existing one. For an example of the use, check out our [Digital Urban Platform]().  
  
![Image](https://github.com/iduprojects/CityGeoTools/blob/metrics-refactor/img/platform_example.png?raw=true)
  
To store urban data we recommend using PostgreSQL for structured tabular data and MongoDB for non-tabular data (e.g. for graphs in binary or XML format). The database structure for PostgreSQL and demo data (for Vasilyevsky Island, St. Petersburg, Russia) are placed in the [data_model](https://github.com/iduprojects/CityGeoTools/tree/metrics-refactor/data_model) folder. Following our best practice to organize and store urban data, you can consider and analyze urban data at the levels of:
1. atomic objects (such as parcels, buildings, services) that have their own properties;
2. infrastructure units (basic, social, transportation, service, leisure, tourism infrastructures);
3. territorial units  (e.g. districts, municipalities and blocks).  
  
CityGeoTools is built on top of the Pandas and Geopandas packages. So uploading from PostgreSQL and transformation SQL tables of massive datasets (e.g. buildings) to GeoDataFrame could take a lot of time. To achieve better performance of application development, RPYC server is used as a separate server with python objects (such as GeoDataFrame, NetworkX graph, etc.) stored in binary format. This allows to avoid long application startup and speed up methods' performance. To start the RPYC server, run the commands bellow with your connection data:
```shell
cd rpyc_server
pip install -r requirements.txt
POSTGRES=user:password@address/database_name MONGO=address python rpyc_server.py
```
After the RPYC server is available, run the application:
```shell
cd metrics
pip install -r requirements.txt
POSTGRES=user:password@address/database_name RPYC_SERVER=address uvicorn app.main:app --host 0.0.0.0 --port 5000
```
The documentation for using the methods can be found at **/docs**. Method call example:
```python
params = {
    'city': 'Saint_Petersburg',
    'x_from': '59.944',
    'y_from': '30.304',
    'view_distance': '700',
}

response = requests.get('http://localhost:5000/api/v2/visibility_analysis/visibility_analysis', params=params)
```
### Docker
The most hands-off way to start working with CityGeoTools in general mode is using docker. Put environment variables POSTGRES and MONGO into .env file and run the commands:
```shell
docker-compose build
docker-compose up
```

## User mode  
  
User mode is useful in case you are not interested in a full-scale solution and want to use only the certain methods for your city. To make this process simpler and faster:  
1. manually declare own City Information Model;  
2. load the data that is required by specification for the certain method;  
3. check if loaded data matches specification;  
4. call the method and get the result as a geojson file.
```python
from metrics.data import CityInformationModel as BaseModel
from metrics.calculations import CityMetricsMethods as CityMetrics

city_model = BaseModel.CityInformationModel(city_name="Saint-Petersburg", city_crs=32636, cwd="./CityGeoTools")
city_model.update_layer("MobilityGraph", "./data/graph.geojson")

if city_model.methods.if_method_available("accessibility_isochrones"):
	isochrone_calculator = CityMetrics.AccessibilityIsochrones(city_model)
	isochrone = isochrone_calculator.get_accessibility_isochrone(
    		travel_type="public_transport", 
    		x_from=349946.36, y_from=6647996.95, 
    		weight_type="time_min",
    		weight_value=15
    	)
```
See detailed information about declaring CityInformationModel and methods call in [examples](https://github.com/iduprojects/CityGeoTools/tree/metrics-refactor/notebook_examples).