import numpy as np
import sys, os, os.path
import calendar 
import rpy2.robjects as ro
import pickle
from rpy2.robjects import pandas2ri
import datetime
from pdb import set_trace as bp
import logging

logger = logging.getLogger(__name__)

online = False

if online:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    submit_dir = os.path.join(input_dir, 'res') 
    truth_dir = os.path.join(input_dir, 'ref')
    year = 2017

else: # adapt to your need, here we assume that res contains
    output_dir = "./res"
    truth_dir = "./ref"
    submit_dir = "./submission"
    year = 2016

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')
    
    err_pol = dict({})
    submission_file = os.path.join(submit_dir, "results.pk")
    time_file = os.path.join(submit_dir, "time.pk")
    tot_score_file = os.path.join(submit_dir, "score.pk")
    with open(time_file, 'rb') as f:
        exectime = pickle.load(f)
    with open(submission_file, 'rb') as f:
        measure = pickle.load(f)
    with open(tot_score_file, 'rb') as f:
        tot_score = pickle.load(f)
    ro.r['load'](os.path.join(truth_dir,'Description_Stations.RData'))
    sites = ro.r['Description_Stations']
    sites = pandas2ri.ri2py(sites)
    start_year_day = datetime.datetime(year, 1, 1).toordinal()
    error_global = []
    for pol in ["PM10","PM25","O3","NO2"]:
        print(pol)
        err_pol[pol] = dict({})
        truth_file =  os.path.join(truth_dir,"Challenge_Data_%s_%s.rds"%(pol,year))
        truth_data = pandas2ri.ri2py(ro.r['readRDS'](truth_file))
        truth_data["idPolair"] = truth_data.idPolair.astype("category")
        truth_data.set_index("date",inplace=True)
        std = np.nanstd(truth_data.Valeur)
        list_truth_site = dict({})
        for site in sites.idPolair:    
            list_truth_site[site]=truth_data[(truth_data.idPolair == site)]
        for h,horizon in enumerate(["D0","D1","D2"]):
            err_pol[pol][horizon] = []
            err_pol_day_site =[]
            for day in measure.keys():
                d0 = datetime.datetime.fromordinal(start_year_day+day+h)    
                d0_str = d0.strftime('%Y-%m-%d')
                d0_str_end = (d0+datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                for site in sites.idPolair:    
                    truth_site=list_truth_site[site]
                    if len(truth_site) !=0:
                        truth_day_site = truth_site.loc[d0_str:d0_str_end] #get vector of measurements for day+horizon and site
                        truth_day_site = truth_day_site.Valeur
                        if (len(truth_day_site)!=0):
                            non_nan= ~(np.isnan(truth_day_site))
                            nan_sol = np.isnan(measure[day][pol][site][horizon])
                            measure[day][pol][site][horizon][nan_sol] = 0 # nan values in proposed solution are replaced by 0 (if we skipped them it would be to easy to have a good score...)
                            
                            if (np.sum(non_nan) != 0): #check if at least one value in ground truth not a nan 
                                D = truth_day_site[non_nan]-measure[day][pol][site][horizon][non_nan]
                                W = np.abs(np.log(np.maximum(measure[day][pol][site][horizon][non_nan],10**(-4))/np.maximum(truth_day_site[non_nan],10**(-4))))
                                tmp = np.mean(D**2 / std**2 * W) #sum over hours
                                if np.isnan(tmp):
                                    tmp=0.
                                    logger.info("Nan error for site %s, day %s and pol %s " %(site,day,pol))
                                    print("NanError", D,"h",W,"h",std,tmp)
                                
                            else: # ref is only nan, put 0
                                logger.info("Only nan values in ground truth for site %s, day %s and pol %s " %(site,day,pol))
                                tmp=0.
                            err_pol_day_site.append(tmp)
                err_pol_day = np.mean(err_pol_day_site) #mean over sites
                err_pol[pol][horizon].append(err_pol_day) 
            output_file.write("%s_%s: %f \n" % (pol.lower(),horizon.lower(),np.mean(err_pol[pol][horizon]))) #mean over days
    #tot = np.sum([(3-h)*p*np.mean(err_pol[pol][horizon]) for pol,p in zip(["O3","NO2","PM10","PM25"],range(1,5)) for h,horizon in enumerate(["D0","D1","D2"])])
    #tot = tot/np.sum([(3-h)*p for p in range(1,5) for h in range(3)])
    
    output_file.write("ave_score: %f \n" % tot_score)
    output_file.write("Duration: %f \n" % exectime)
    output_file.close()
    