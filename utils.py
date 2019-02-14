# -*- coding: utf-8 -*-
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from glob import glob
from os.path import join
import pandas
import datetime
import numpy as np
import logging
import os.path


def load_data(dirname="../Data/training",f_geop="Geop_d02",f_meteo="meteoWRF",year=2016):
    """
    Load all files in memory for a given year
    """
    logger = logging.getLogger(__name__)
    
    ## sites
    ## Dataframe with columns "idPolair","nom_station","coord_x_l93","coord_y_l93","X_lamb2","Y_lamb2", "LON" ,"LAT", 
    #"Département","Zone_EPCI","typologie","NO2_influence", "NO2_2012", "NO2_2013","NO2_2014","NO2_2015", "NO2_2016","NO2_2017",
    #"O3_influence"   "O3_2012" ,"O3_2013", "O3_2014", "O3_2015","O3_2016" "PM10_2017","PM25_influence" "PM25_2012","PM25_2013","PM25_2014" 
    # "O3_2017","PM10_influence" "PM10_2012","PM10_2013","PM10_2014","PM10_2015","PM10_2016"    
    # "PM25_2015","PM25_2016","PM25_2017"
    logger.info("Loading sites files")
    ro.r['load'](join(dirname,'Description_Stations.RData'))
    sites = ro.r['Description_Stations']
    sites = pandas2ri.ri2py(sites)
    
    ## chimere 
    ## Dict on Pollutants
    ## for each pollutant
    ## Dataframe with columns 'date', 'val', 'idPolair', 'param'
    logger.info("Loading Chimere files")
    chimeres = dict({})
    for pol in ["PM10","PM25","O3","NO2"]:
        fname=join(dirname,"CHIMERE/CHIMERE_%s_%s.rds" %(pol,year))
        data = loadFile(fname)
        chimeres[pol]=data
    
    
    ## Geop
    ## Dict on sites
    ## Dataframe with columns 'date', 'idPolair', 'geop_p_500hPa', 'geop_p_850hPa'
    logger.info("Loading Geop files")
    geops = dict({})
    for site in sites.idPolair:
        fname = join(dirname,"WRF/Geop_02/Geop.%s.%s.d02.rds" %(site.lstrip('0'),year))
        if os.path.isfile(fname):
            data = loadFile(fname)
            geops[site] = data
        else:
            logger.info("Site %s does not have a geop file" %site)
    
    ## meteo
    ## Dataframe with columns "date", "idPolair", "T2", "Q2", "U10", "V10" "PSFC", "PBLH", "LH", "HFX", "ALBEDO", "SNOWC", "HR2", "VV10", "DV10", "PRECIP"
    logger.info("Loading Meteo files")
    fname = join(dirname,"WRF/%s_%s.RData"%(f_meteo,year))
    meteo = loadFile(fname,"wrfData")
    
    ## concentrations
    ## Dict on Pollutants
    ## for each pollutant, a dataframe with columns 'idPolair', 'Organisme', 'Station', 'Mesure', 'Date', 'Valeur'
    logger.info("Loading concentrations measure files")
    concentrations = dict({})
    for pol in ["PM10","PM25","O3","NO2"]:
        fname=join(dirname,"measures/Challenge_Data_%s_%s.rds" %(pol,year))
        data = loadFile(fname)
        concentrations[pol] = data
    return {"sites":sites,"chimeres":chimeres,"geops":geops,"meteo":meteo,"concentrations":concentrations}


def get_data_day(day,all_data,max_days=10,year=2016):
    """
    Given the loaded files (all_data) and a day of the year, get the data filtered between day-max_days and day+6hours
    """
    chimeres = all_data["chimeres"]
    geops = all_data["geops"]
    meteo = all_data["meteo"]
    concentrations = all_data["concentrations"]
    
    ## convert day number to datetime
    start_year_day = datetime.datetime(year, 1, 1).toordinal()
    d0 = datetime.datetime.fromordinal(start_year_day+day)
    past = datetime.datetime.fromordinal(start_year_day + np.maximum(day-max_days,0))
    
    
    ##Chimere
    ## Dict on Pollutants
    ## for each pollutant
    ## Dataframe with columns 'date', 'val', 'idPolair', 'param'
    chimeres_day = dict({})
    for pol in ["PM10","PM25","O3","NO2"]:
        data = filterByDay(chimeres[pol],d0,past,upto=3*24) ## for chimeres simulations get full 3 three days prediction
        chimeres_day[pol]=data
    
    ## Geop
    ## Dict on sites
    ## Dataframe with columns 'date', 'idPolair', 'geop_p_500hPa', 'geop_p_850hPa'
    geops_day=dict({})
    for site in geops.keys():
        data = filterByDay(geops[site],d0,past,upto=6)
        geops_day[site] = data
    
    ## meteo
    ## Dataframe with columns "date", "idPolair", "T2", "Q2", "U10", "V10" "PSFC", "PBLH", "LH", "HFX", "ALBEDO", "SNOWC", "HR2", "VV10", "DV10", "PRECIP"
    meteo_day = filterByDay(meteo,d0,past,upto=6)
    
    ## concentrations
    ## Dict on Pollutants
    ## for each pollutant, a dataframe with columns 'idPolair', 'Organisme', 'Station', 'Mesure', 'date', 'Valeur'
    concentrations_day = dict({})
    for pol in ["PM10","PM25","O3","NO2"]:
        data = filterByDay(concentrations[pol],d0,past)
        concentrations_day[pol] = data
    return chimeres_day,geops_day,meteo_day,concentrations_day

def loadFile(fname,varname=None):
    """
    fname :  rdata or rds filename to be loaded
    varname : variable name inside rdata
    """
    if varname is not None:
        ro.r['load'](fname)
        full_data =pandas2ri.ri2py(ro.r[varname])
    else: #assume it is in rds format
        full_data = pandas2ri.ri2py(ro.r['readRDS'](fname))
    if "date" in full_data.columns:
        full_data["date"] = pandas.to_datetime(full_data.date)
    if "idPolair" in full_data.columns:
        full_data["idPolair"] = full_data.idPolair.astype("category")
    return full_data

def filterByDay(full_data,d0,past,upto=6):
    """
    full_data :  full year dataframe
    d0 : datetime of day to filter on
    past : int number of past days to look into 
    upto : int number of hours allowed from d0 at 00:00 (6 hours for most measures, whole 3 days of prediction for Chimere simulations) 
    """
    
    end = (d0+datetime.timedelta(hours=upto)).strftime('%Y-%m-%d %H')
    start = past.strftime('%Y-%m-%d')
    data = full_data.loc[(full_data["date"] <= end) & (full_data["date"] >= start)]
    return data

def get_ground_truth_day(day,all_data,common_sites,year=2016):
    """
    Given the loaded files (all_data) and a day of the year, get the ground truth for the 3 following days
    """
    concentrations = all_data["concentrations"]
    sites = all_data["sites"]
    
    ## convert day number to datetime
    start_year_day = datetime.datetime(year, 1, 1).toordinal()
    d0 = datetime.datetime.fromordinal(start_year_day+day)
    
    ## concentrations
    ## Dict on Pollutants
    ## for each pollutant, a dataframe with columns 'idPolair', 'Organisme', 'Station', 'Mesure', 'date', 'Valeur'
    concentrations_day = dict({})
    for idPolair in common_sites:
        concentrations_day[idPolair] = np.zeros(4*24*3)
        for p,pol in enumerate(["PM10","PM25","O3","NO2"]):
            end = (d0+datetime.timedelta(hours=72)).strftime('%Y-%m-%d %H')
            start = d0.strftime('%Y-%m-%d')
            data = concentrations[pol].loc[(concentrations[pol]["date"] < end) & (concentrations[pol]["date"] >= start)]
            
            concentrations_day[idPolair][p::4] = data.Valeur[data.idPolair == idPolair] #first value is PM10 at H0, second value is PM25 at H0, 5th value is PM10 at H1 etc...
    return concentrations_day

def get_common_sites(data):
    common_sites = set(data["concentrations"]["PM10"].idPolair) 
    for pol in ["PM10","PM25","NO2","O3"]:
        common_sites = set(data["concentrations"][pol].idPolair).intersection(common_sites)
    
    for pol in ["PM10","PM25","NO2","O3"]: #remove sites with more than
        dat = data['concentrations'][pol]
        
        for site in common_sites.copy():
            if np.sum(np.isnan(dat.Valeur[dat.idPolair ==site]))/len(dat.Valeur[dat.idPolair ==site]) > 0.5:
                common_sites.remove(site)
    return common_sites