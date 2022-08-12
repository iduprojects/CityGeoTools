## CityGeoTools
**CityGeoTools** is a python package of methods developed for analysis of the urban environment. These  methods could be useful for urban researchers and local governments for making informed decisions in the field of urban economy. At this moment the following methods are available: 
   
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
More detailed descriptions and examples of how to use the methods are provided as *jupyter notebooks* in [examples](). In order to use these methods, it is not necessary to have a deep knowledge of python programming and machine learning. Simply follow the steps shown in the figure below. Two ways of using the presented methods are implemented - **general mode** and **user mode**.
  
![Image](https://github.com/iduprojects/CityGeoTools/blob/metrics-refactor/img/plot.png?raw=true)

It should be noted that we do not publish the package in a PyPI repository at this moment. Therefore, to work with the this package, first you need to download the repository to the local machine:
```shell
https://github.com/iduprojects/CityGeoTools
```

## Data preparation  
  
## General mode
General mode provides access to all data stored in databases through a data query interface and allows the use of all of the developed methods through API. This mode makes it possible to build an application for urban environment analysis or integrate the methods into existing one. For an example of this use, check out our [Digital Urban Platform]().

To store urban data we recommend using PostgreSQL for structured tabular data and MongoDB for non-tabular data (e.g. for graphs in binary or XML format). The database structure for PostgreSQL and demo data (for St. Petersburg, Russia) are placed in the [data_model]() folder. Following our 'best practice' to orginize and store urban data, 
  
## User mode  
  
