## CityGeoTools
**CityGeoTools** is a set of method developed for analysis of the urban environment. These  methods could be useful for urban researches and local government for making informed decisions in the field of urban economy. At this moment the following methods are availbale: 
   
1.   Traffic calculation (between houses and stops)
2.   Visibility analysis (from certain point in the city)
3.   Weighted voronoi (to identify unserved zones, based on [research](https://www.sciencedirect.com/science/article/pii/S187705092032384X))
3.   Blocks clusterization (based on presented services)
4.   Service clusterization (based on services location)
5.   Spacematrix clusterization (based on [research](https://elibrary.ru/item.asp?id=45845752))
6. Availability isochrone (based on [research]() )
7.  Diversity analysis (based on available services) - in progress for *user mode*
8.  Provision population with services - in progress for *user mode*
9.  Wellbeing (based on provision with services needed in living situations) - in progress for *user mode*

## How to use CityGeoTools
More detailed descriptions and examples of how to use the methods are provided as *jupyter notebooks* in [examples](). In order to use these methods, it is not necessary to have deep knowledge of python programming and maching learning. Simply follow the steps shown in the figure below. Two ways of using the presented methods are implemented - *general mode* and *user mode*.
  
![Image](https://github.com/iduprojects/CityGeoTools/blob/metrics-refactor/plot.png?raw=true)

## General mode
General mode provides access to all data stored in databases through data query interface and allows to use all of the developed method trough API. This mode makes it possible to build an application for  urban environment analysis or integrate the methods into existed one. For an example of this use, check out our [Digital Urban Platform]().

