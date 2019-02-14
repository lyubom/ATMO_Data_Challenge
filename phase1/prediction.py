# -*- coding: utf-8 -*-
from utils import load_data,get_data_day
import numpy as np
import pickle
import datetime
import calendar
import time
import os
import sys
import pickle

from keras.models import model_from_json
from keras import optimizers


#### Predict function for a given day

dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, 'mean.npz')

# load model
# model_filename = os.path.join(dirname, "models/lstm_model")
model_filename = os.path.join(dirname, "models/randomforest2.sav")
mean_filename = os.path.join(dirname, "tmp/mean_coord.npz")

# RANDOM FOREST MODEL
mean = np.load(mean_filename)['a']
loaded_model = pickle.load(open(model_filename, 'rb'))

print("Loaded model from disk")

def predict(day,sites,chimeres_day,geops_day,meteo_day,concentrations_day,model=None):
    """
    day: day of the year (1-365)

    sites : Dataframe with columns "idPolair","nom_station","coord_x_l93","coord_y_l93","X_lamb2","Y_lamb2", "LON" ,"LAT",
"Département","Zone_EPCI","typologie","NO2_influence", "NO2_2012", "NO2_2013","NO2_2014","NO2_2015", "NO2_2016","NO2_2017","O3_influence","O3_2012",  "O3_2013", "O3_2014", "O3_2015","O3_2016" "PM10_2017","PM25_influence" "PM25_2012","PM25_2013","PM25_2014","O3_2017","PM10_influence" "PM10_2012","PM10_2013","PM10_2014","PM10_2015","PM10_2016","PM25_2015","PM25_2016","PM25_2017".
chimeres_day Dict on Pollutants, for each pollutant a Dataframe with columns 'date', 'val', 'idPolair', 'param'. Stopped at D0+72H

geops_day : Dict on sites, for each site a Dataframe with columns 'date', 'idPolair', 'geop_p_500hPa', 'geop_p_850hPa'. Stopped at D0+6H

meteo_day : Dataframe with columns "date", "idPolair", "T2", "Q2", "U10", "V10" "PSFC", "PBLH", "LH", "HFX", "ALBEDO", "SNOWC", "HR2", "VV10", "DV10", "PRECIP". Stopped at D0+6H

concentrations_day : Dict on Pollutants, for each pollutant, a dataframe with columns 'idPolair', 'Organisme', 'Station', 'Mesure', 'Date', 'Valeur'. Stopped at D0+6H
model : pretrained model data (e.g. saved learned sklearn model) if you have one. Change its default value with a relative path if you want to load a file
    """

    # Prediction step: up to you !



    results = dict({})
    for pol in ["PM10","PM25","O3","NO2"]:
        results[pol] = dict({})
        concentrations_pol = concentrations_day[pol]
        for idPolair in sites.idPolair:
            concentrations_pol_site = concentrations_pol[concentrations_pol.idPolair==idPolair]
            results[pol][idPolair] = dict({})

            if np.sum(~np.isnan(concentrations_pol_site.Valeur))!=0:


                tmp = concentrations_pol_site.Valeur
                # last_values = np.tile(tmp[72:], 3)[0:17]
                # tmp = np.concatenate((tmp, last_values))
                # previous_data = tmp.reshape([1, 4, 24])
                previous_data = np.array(tmp)
                if np.isnan(previous_data).any():
                    if idPolair != '33374':
                        chimeres_site = chimeres_day[pol].loc[chimeres_day[pol].idPolair == float(idPolair)]
                    else:
                        chimeres_site = chimeres_day[pol].loc[chimeres_day[pol].idPolair == 15114.]
                    inds = np.where(np.isnan(previous_data))
                    previous_data[inds] = chimeres_site['val'].iloc[inds]

                # IF DATA TYPE 2 IS USED
                if len(idPolair) == 4:
                    site = sites.loc[sites.idPolair == ('0'+idPolair)]
                else:
                    site = sites.loc[sites.idPolair == idPolair]

                tmp = np.array(site.loc[:, "coord_x_l93":"coord_y_l93"])[0] - mean
                tmp = np.concatenate(([convert_pol(pol)], tmp))
                previous_data = np.concatenate((tmp, previous_data))

                previous_data = np.expand_dims(previous_data, axis=0)


                result = loaded_model.predict(previous_data)

                for i, horizon in enumerate(["D0","D1","D2"]):
                    results[pol][idPolair][horizon] = result[0, i*24:(i+1)*24]

                results[pol][idPolair]["D0"][0:7] = previous_data[0, 72:79]
            else:
                for horizon in ["D0","D1","D2"]:
                    results[pol][idPolair][horizon] = np.zeros(24)
    return results

    # result format
    # dict[pol][site][horizon][array of size 24 (hourly prediction)]
    # results = dict({})
    # for pol in ["PM10","PM25","O3","NO2"]:
    #     results[pol] = dict({})
    #     concentrations_pol = concentrations_day[pol]
    #     for idPolair in sites.idPolair:
    #         concentrations_pol_site = concentrations_pol[concentrations_pol.idPolair==idPolair]
    #         results[pol][idPolair] = dict({})
    #         for horizon in ["D0","D1","D2"]:
    #
    #             ####### your prediction step
    #             if np.sum(~np.isnan(concentrations_pol_site.Valeur))!=0:
    #                 results[pol][idPolair][horizon] = np.ones(24)*np.nanmean(concentrations_pol_site.Valeur) #dummy example where we just copy for each hour the mean concentration of all previous available days
    #             else:
    #                 results[pol][idPolair][horizon] = np.zeros(24)
    #             ######
    #

def convert_pol(pol):
    if pol == "PM10":
        return 4
    elif pol == "PM25":
        return 3
    elif pol == "NO2":
        return 2
    elif pol == "O3":
        return 1


def convert_data_day(day, pol, idPolair, sites, chimeres_day, geops_day, meteo_day, concentrations_day):
    measures = 3*24 + 7
    ids = 3
    rows = 3*24 + 7
    columns = 18

    data = np.empty(shape=[rows, columns])
    # data = np.empty(shape=[ids+measures])
    if len(idPolair) == 4:
        site = sites.loc[sites.idPolair == ('0'+idPolair)]
    else:
        site = sites.loc[sites.idPolair == idPolair]
    if idPolair != '33374':
        meteo_site = meteo_day.loc[meteo_day.idPolair == float(idPolair)]
    else:
        meteo_site = meteo_day.loc[meteo_day.idPolair == 15114.]

    pol_ = convert_pol(pol)


    # if idPolair in geops_day.keys():
    #     geops_site = geops_day[idPolair]
    # else:
    #     mean_geop_p_500hPa = []
    #     mean_geop_p_850hPa = []
    #     for idPolair_ in geops_day.keys():
    #         mean_geop_p_500hPa.append(np.nanmean(geops_day[idPolair_].geop_p_500hPa))
    #         mean_geop_p_850hPa.append(np.nanmean(geops_day[idPolair_].geop_p_850hPa))

    if idPolair[0] == '0':
        concentrations_site = concentrations_day[pol].loc[concentrations_day[pol].idPolair == idPolair[1:]]
    else:
        concentrations_site = concentrations_day[pol].loc[concentrations_day[pol].idPolair == idPolair]

    if idPolair != '33374':
        chimeres_site = chimeres_day[pol].loc[chimeres_day[pol].idPolair == float(idPolair)]
    else:
        chimeres_site = chimeres_day[pol].loc[chimeres_day[pol].idPolair == 15114.]

    data[:, 0] = pol_*np.ones(shape=[rows])
    tmp = np.array(site.loc[:, "coord_x_l93":"coord_y_l93"])
    data[:, 1:3] = np.repeat(tmp, rows, axis=0)

    data[:, 3:17] = meteo_site.loc[:, 'T2':'PRECIP']
    # print(concentrations_day[pol])

    data[:, 17] = concentrations_site.Valeur
    # print(concentrations_site.Valeur)
    # print(concentrations_site.Valeur.iloc[6])
    # if np.isnan(data[3:]).any():
    #     for i in range(measures):
    #         chimeres_site_date = chimeres_site.iloc[i]
    #         data[i+3] = chimeres_site_date.val

    if np.isnan(data[:, 17]).any():
        inds = np.where(np.isnan(data[:, 17]))
        data[:, 17][inds] = chimeres_site.val.iloc[inds]

    return data
    # print(concentrations_site)
    # for i in range(measures):
    #     date = np.unique(meteo_day['date'])[i]
    #     # data[i, 1] = site.coord_x_l93
    #     # data[i, 2] = site["coord_y_l93"]
    #     # data[i, 3] = site["X_lamb2"]
    #     # data[i, 4] = site["Y_lamb2"]
    #     # data[i, 5] = site["LON"]
    #     # data[i, 6] = site["LAT"]
    #
    #     # date = meteo_day.date.iloc[i]
    #
    #
    #     # meteo_site_date = meteo_site.loc[meteo_site.date == date]
    #
    #
    #
    #     # data [i, 7:21] = meteo_site_date.loc[:, 'T2':'PRECIP']
    #     #
    #     # if idPolair in geops_day.keys():
    #     #
    #     #     geops_site_date = geops_site.loc[geops_site.date == date]
    #     #
    #     #     # data[i, 21] = geops_site_date.geop_p_500hPa
    #     #     # data[i, 22] = geops_site_date.geop_p_850hPa
    #     #     data[i, 21:23] = geops_site_date.loc[:, "geop_p_500hPa":"geop_p_850hPa"]
    #     # else:
    #     #
    #     #     data[i, 21] = np.mean(mean_geop_p_500hPa)
    #     #     data[i, 22] = np.mean(mean_geop_p_850hPa)
    #
    #
    #
    #     concentrations_site_date = concentrations_site.loc[concentrations_site.date == date]
    #
    #
    #     # chimeres_site = chimeres_day[pol].loc[chimeres_day[pol].idPolair == idPolair]
    #     # chimeres_site_date = chimeres_site.loc[chimeres_site.date == date]
    #     # print(concentrations_site_date.Valeur)
    #     # print('sep')
    #
    #
    #     # data[i + 3] = concentrations_site_date.Mesure
    #     if not np.isnan(concentrations_site_date.Valeur.iloc[0]):
    #         data[i + 3] = concentrations_site_date.Valeur
    #     else:
    #         chimeres_site_date = chimeres_site.loc[chimeres_site.date == date]
    #         data[i + 3] = chimeres_site_date.val
    #         # print(concentrations_site_date.Valeur)
    #         # print(chimeres_site_date.val)

        # data[i, 7] = meteo_day[]
        # sys.exit(1)





#### Main loop (no need to be changed)

def run_predict(year=2016,max_days=3,dirname="../Data/training",list_days=None):
    """
    year : year to be evaluated
    max_days: number of past days allowed to predict a given day (set to 10 on the platform)
    dirname: path to the dataset
    list_days: list of days to be evaluated (if None the full year is evaluated)
    """

    overall_start = time.time() # <== Mark starting time
    data = load_data(year=year,dirname=dirname) # load all data files
    sites = data["sites"] #get sites info
    day_results = dict({})
    if list_days is None:
        if calendar.isleap(year): # check if year is leap
            list_days = range(3, 366)
        else:
            list_days = range(3, 365)
    for day in list_days:
        print(day)
        chimeres_day,geops_day,meteo_day,concentrations_day = get_data_day(day,data,max_days=max_days,year=year) # you will get an extraction of the year datasets, limited to the past max_days for each day
        day_results[day] = predict(day,sites,chimeres_day,geops_day,meteo_day,concentrations_day) # do the prediction

    overall_time_spent = time.time() - overall_start # end computation time
    pickle.dump(day_results, open('submission/results.pk', 'wb')) #save results
    pickle.dump(overall_time_spent, open('submission/time.pk', 'wb')) #save computation time
