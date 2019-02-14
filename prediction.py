# -*- coding: utf-8 -*-
from utils import load_data,get_data_day, get_common_sites, get_ground_truth_day
import numpy as np
import pickle
import datetime
import calendar
import time
import os
from scipy.spatial.distance import mahalanobis
import pickle


#### Predict function for a given day

dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, 'mean.npz')

# load model
model_filename = os.path.join(dirname, "models/randomforest_full.sav")

# RANDOM FOREST MODEL
loaded_model = pickle.load(open(model_filename, 'rb'))

print("Loaded model from disk")

def predict(day,sites,common_sites,chimeres_day,geops_day,meteo_day,concentrations_day,model=None):
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
    #################
    # dummy example where we just average the first 6 hours and use them as variables for each pollutant. You can change everything but you need to return
    # the same format for results and results_covar
    #------------

    # You need to return a dictionary predictions that stores
    # for each site idPolair a vector v where the value of a pollutant pol at the day d at hour h is at v[pol + 4*(h+24*d)]
    # That is first value is PM10 at H0, second value is PM25 at H0, 5th value is PM10 at H1 etc...
    # And a results_covar dictionary that stores for each
    # site idPolair the covariance matrix corresponding to vector v

    results = dict({})
    results_covar = dict({})
    for idPolair in common_sites:

        # fist we create our regressor vector here of size 4 (nb pollutants) * 6 hours
        x = np.zeros(4*3*24)
        # then we build our regression matrix of size 288 (number of predictions) * 24 (number of regressors)
        # M = np.zeros((4*3*24,4*3*24))
        for p,pol in enumerate(["PM10","PM25","O3","NO2"]):
            concentrations_pol = concentrations_day[pol]
            concentrations_pol_site = concentrations_pol[concentrations_pol.idPolair==idPolair]
            tmp = concentrations_pol_site.Valeur
            previous_data = np.array(tmp)
            if np.isnan(previous_data).any():
                if idPolair != '33374':
                    chimeres_site = chimeres_day[pol].loc[chimeres_day[pol].idPolair == float(idPolair)]
                else:
                    chimeres_site = chimeres_day[pol].loc[chimeres_day[pol].idPolair == 15114.]
                inds = np.where(np.isnan(previous_data))
                previous_data[inds] = chimeres_site['val'].iloc[inds]

            previous_data = previous_data[-79:]
            previous_data = np.expand_dims(previous_data, axis=0)

            result = loaded_model.predict(previous_data)

            x[p::4] = result
            x[p::4][0:7] = previous_data[0, 72:79]

            # if np.sum(~(np.isnan(concentrations_pol_site.Valeur))) == 0:
            #     concentrations_pol_site.Valeur.fillna(0,inplace=True) # replace NaN by 0
            #     print(pol,idPolair)
            # else:
            #     concentrations_pol_site.Valeur.fillna(np.nanmean(concentrations_pol_site.Valeur),inplace=True) # replace NaN by mean
            # x[p::4] = concentrations_pol_site.Valeur[-6:] # Get only the last 6 values from the past data(6 first hours of current day). First value is PM10 at H0, second value is PM25 at H0, 5th value is PM10 at H1 etc...
            # # then we build our regression matrix
            # M[p::4,p::4] = 1./6 # for each pollutant for each hour we will just get the mean of the first 6 hours

        # then we get our prediction and covariance matrix . The covariance matrix must also follow the same pattern as the vector
        results[idPolair] = x # we get a vector where first value is PM10 at H0 (day0), second value is PM25 at H0, 5th value is PM10 at H1, 97th value is PM10 at H25(day2) etc...
        mean = np.repeat(np.mean(x), 4*3*24)
        x_ = x
        x_ = np.expand_dims(x_, axis=1)
        results_covar[idPolair] = np.dot(x_, x_.T) # here we simply take M.M' as covar example
        print(results_covar[idPolair][0:16, 0:16])
        np.fill_diagonal(results_covar[idPolair], 1.)
        print(results_covar[idPolair][0:16, 0:16])
        sys.exit()
        # print(np.isnan(results_covar[idPolair]).any())
        # print(np.isnan(results[idPolair]).any())

    ##########################
    # phase 1 result format
    # NO MODIFICATION NEEDED IN THIS PART
    # dict[pol][site][horizon][array of size 24 (hourly prediction)]
    results_oldformat = dict({})
    for p, pol in enumerate(["PM10","PM25","O3","NO2"]):
        results_oldformat[pol] = dict({})
        concentrations_pol = concentrations_day[pol]
        for idPolair in common_sites:
            concentrations_pol_site = concentrations_pol[concentrations_pol.idPolair==idPolair]
            results_oldformat[pol][idPolair] = dict({})
            for d, horizon in enumerate(["D0","D1","D2"]):
                results_oldformat[pol][idPolair][horizon] = np.zeros(24)
                for hour in range(24):
                    results_oldformat[pol][idPolair][horizon] = results[idPolair][p + 4*(hour+24*d)]

    return results_oldformat, results, results_covar

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

    data[:, 17] = concentrations_site.Valeur

    if np.isnan(data[:, 17]).any():
        inds = np.where(np.isnan(data[:, 17]))
        data[:, 17][inds] = chimeres_site.val.iloc[inds]

    return data

#### Main loop (no need to be changed)

def run_predict(year=2016,max_days=10,dirname="../Data/training",list_days=None):
    """
    year : year to be evaluated
    max_days: number of past days allowed to predict a given day (set to 10 on the platform)
    dirname: path to the dataset
    list_days: list of days to be evaluated (if None the full year is evaluated)
    """

    overall_start = time.time() # <== Mark starting time
    data = load_data(year=year,dirname=dirname) # load all data files
    sites = data["sites"] #get sites info

    # get common sites for all pollutants
    common_sites = get_common_sites(data)
    day_results = dict({})
    day_scores = []
    if list_days is None:
        if calendar.isleap(year): # check if year is leap
            list_days = range(366)
        else:
            list_days = range(365)
    for day in list_days:
        print(day)
        chimeres_day,geops_day,meteo_day,concentrations_day = get_data_day(day,data,max_days=max_days,year=year) # you will get an extraction of the year datasets, limited to the past max_days for each day
        results_oldformat, results, results_covar = predict(day,sites,common_sites,chimeres_day,geops_day,meteo_day,concentrations_day) # do the prediction
        ground_truth = get_ground_truth_day(day,data,common_sites,year=year)
        score = np.mean([mahalanobis(ground_truth[i][24:],results[i][24:],results_covar[i][24:,24:])**2 + np.log(np.linalg.det(results_covar[i][24:,24:])) for i in common_sites]) # mean over sites, remove the first 6 hours (24 = 6*4 pollutants) from score because they are known.
        day_scores.append(score)
        day_results[day] = results_oldformat
    tot_score = np.mean(day_scores) # mean over days
    print(tot_score)
    overall_time_spent = time.time() - overall_start # end computation time
    pickle.dump(day_results, open('submission/results.pk', 'wb')) #save results
    pickle.dump(tot_score, open('submission/score.pk', 'wb')) #save score
    pickle.dump(overall_time_spent, open('submission/time.pk', 'wb')) #save computation time
