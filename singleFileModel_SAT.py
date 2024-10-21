#%%
import sys
import os
import glob
import numpy as np
import mat73
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr 
import scipy
from numpy.polynomial.polynomial import Polynomial
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.interpolate import RegularGridInterpolator, interp1d, make_interp_spline, splrep, BSpline
from torch.utils.data import random_split
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from sklearn.cluster import MiniBatchKMeans
import matplotlib.colors as mcolors
import gsw
import seaborn as sns
from scipy.stats import linregress
from scipy.spatial.distance import cdist
import cmocean.cm as ccm
from collections import Counter
from tqdm import tqdm  # For the progress bar
from pandas import DataFrame as df
import time
import pandas as pd
import calendar
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from scipy.stats import pearsonr
from matplotlib.ticker import FormatStrFormatter

# from matplotlib.dates import date2num, num2date
# from sklearn.cluster import MiniBatchKMeans
sys.path.append("/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/eoas_pyutils")
from io_utils.coaps_io_data import get_aviso_by_date, get_sst_by_date, get_sss_by_date
sys.path.append("/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/")

plt.rcParams.update({'font.size': 18})
# Set the seed for reproducibility
# load_trained_model = False
load_trained_model = True
ensemble_models = False
# load_dataset_file = False
load_dataset_file = True
gen_paula_profiles = False
global debug
debug = False  # Set to False to disable debugging
seed = 42
n_runs = 1# number of model runs
nn_repeat_time = 10 # number of nespreso runs for generation timing 
gem_repeat_time = 1 # number of GEM runs for generation timing

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

coolwhitewarm = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue', 
                                                 colors =[(0, 0, 1), 
                                                          (1, 1., 1), 
                                                          (1, 0, 0)]
                                                 )

def datenum_to_datetime(matlab_datenum):
    # MATLAB's datenum (1) is equivalent to January 1, year 0000, but Python's datetime minimal year is 1
    # There are 366 days for year 0 in MATLAB (it's a leap year in proleptic ISO calendar)
    days_from_year_0_to_year_1 = 366
    python_datetime = datetime.fromordinal(int(matlab_datenum) - days_from_year_0_to_year_1) + timedelta(days=matlab_datenum % 1)
    return python_datetime

def rmse(predictions, targets):
    return np.sqrt(np.nanmean(((predictions - targets) ** 2)))

def bias(predictions, targets):
    return np.nanmean((predictions - targets))

def mad(x):
    '''
    median absolute deviation
    '''
    return np.nanmedian(np.absolute(x - np.nanmedian(x, axis = 1, keepdims=True)), axis = 1)

def load_satellite_data(TIME, LAT, LON):
    """
    New method to load SST and SSH data
    """
    aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
    sss_folder = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
    min_lat = 18.0
    max_lat = 31.0
    min_lon = -98.0
    max_lon = -81.0
    ex_lon = -88.0
    ex_lat = 23.0
    unique_dates = sorted(list(set(TIME)))
    sss_data = np.nan * np.ones(len(TIME))
    sst_data = np.nan * np.ones(len(TIME))
    aviso_data = np.nan * np.ones(len(TIME))
    bbox=(min_lat, max_lat, min_lon, max_lon)
    
    # # Convert serialized date numbers to date objects
    # base_date = datetime(1, 1, 1)

    # for idx, serialized_date in enumerate(unique_dates):
    for idx, c_date in enumerate(unique_dates):
        print(f"Querying satellite data:  {c_date.date()}.")
        # c_date = base_date + timedelta(days=float(serialized_date))
        date_idx = np.array([date_obj == c_date for date_obj in TIME])  # Ensure both sides of the comparison are datetime objects
        coordinates = np.array([LAT[date_idx], LON[date_idx]]).T
        
        # TODO: get errors
        
        try:
            sss_datapoint, lats, lons = get_sss_by_date(sss_folder, c_date, bbox)
            interpolator = RegularGridInterpolator((lats, lons), sss_datapoint.sss_smap_40km.values, bounds_error=False, fill_value=None)
            sss_data[date_idx] = interpolator(coordinates)
            if (sss_data[date_idx] < 0).any() or (sss_data[date_idx] > 45).any():
                sss_data[date_idx] = np.nan
                print(f"Invalid SSS on date {c_date}, value: {sss_data[date_idx]}, coordinates: {coordinates}, lats: {lats}, lons: {lons}")
        except Exception as e:
            print(f"SSS not found on date {c_date}. Error: {e}")
        
        try:
            aviso_adt, aviso_lats, aviso_lons = get_aviso_by_date(aviso_folder, c_date, bbox)
            # Generate 2D arrays of latitudes and longitudes for each point in the grid
            lons, lats = np.meshgrid(aviso_lons, aviso_lats)

            # Create the mask based on inclusion criteria (bbox) and exclusion criteria
            inclusion_mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
            exclusion_mask = (lats < ex_lat) & (lons > ex_lon)

            # Combine masks to exclude the specified area
            combined_mask = inclusion_mask & ~exclusion_mask

            # Apply the combined mask to filter the data before calculating the mean
            daily_avg = np.nanmean(aviso_adt.adt.values[combined_mask])
                        
            interpolator_ssh = RegularGridInterpolator((aviso_lats, aviso_lons), aviso_adt.adt.values, bounds_error=False, fill_value=None)
            aviso_data[date_idx] = interpolator_ssh(coordinates) - daily_avg
            
        except Exception as e:
            print("AVISO not found for date ", c_date, "Error: ", str(e))
            continue

        try:
            sst_date, sst_lats, sst_lons = get_sst_by_date(sst_folder, c_date, bbox)
            interpolator_sst = RegularGridInterpolator((sst_lats, sst_lons), sst_date.analysed_sst.values[0], bounds_error=False, fill_value=None)
            sst_data[date_idx] = interpolator_sst(coordinates)
            if (sst_data[date_idx] < 0).any() or (sst_data[date_idx] > 350).any():
                sst_data[date_idx] = np.nan
                print(f"Invalid SST on date {c_date}, value: {sst_data[date_idx]}, coordinates: {coordinates}, lats: {sst_lats}, lons: {sst_lons}")
        except Exception as e:
            print("SST not found for date ", c_date, "Error: ", str(e))
            continue

        # Check if data was actually filled
        if np.isnan(aviso_data[date_idx]).all():
            print(f"No AVISO data for date {c_date}")
        if np.isnan(sst_data[date_idx]).all():
            print(f"No SST data for date {c_date}")

    return sss_data, sst_data, aviso_data

def prepare_inputs(time, lat, lon, sss, sst, ssh, input_params):
    """
    Transforms the individual data arrays into the format expected by the model.

    Args:
    - time (array): Time data.
    - lat (array): Latitude data.
    - lon (array): Longitude data.
    - sss (array): Sea Surface Salinity data.
    - sst (array): Sea Surface Temperature data.
    - ssh (array): Sea Surface Height data.
    - input_params (dict): Dictionary indicating which features to include.

    Returns:
    - torch.Tensor: Tensor of transformed input data.
    """
    try:
        num_samples = len(time)  # Assuming all arrays have the same length
    except:
        num_samples = 1
        
    inputs = []

    # Iterate over each sample and create input features
    for i in range(num_samples):
        sample_inputs = []
        
        if input_params.get("timecos", False):
            sample_inputs.append(np.cos(2 * np.pi * (time[i] % 365) / 365))
        
        if input_params.get("timesin", False):
            sample_inputs.append(np.sin(2 * np.pi * (time[i] % 365) / 365))
        
        if input_params.get("latcos", False):
            sample_inputs.append(np.cos(2 * np.pi * (lat[i] / 180)))
        
        if input_params.get("latsin", False):
            sample_inputs.append(np.sin(2 * np.pi * (lat[i] / 180)))
        
        if input_params.get("loncos", False):
            sample_inputs.append(np.cos(2 * np.pi * (lon[i] / 360)))
        
        if input_params.get("lonsin", False):
            sample_inputs.append(np.sin(2 * np.pi * (lon[i] / 360)))

        if input_params.get("sat", False):
            if input_params.get("sss", False):
                sample_inputs.append(sss[i])
            if input_params.get("sst", False):
                sample_inputs.append(sst[i] - 273.15)
            if input_params.get("ssh", False):
                sample_inputs.append(ssh[i])
                
        # Convert the list of inputs for this sample to a tensor and add to the main list
        inputs.append(torch.tensor(sample_inputs, dtype=torch.float32))

    # Convert the list of tensors to a single tensor
    inputs_tensor = torch.stack(inputs)

    return inputs_tensor

class TemperatureSalinityDataset(torch.utils.data.Dataset):
    """
    Custom dataset for temperature and salinity profiles.
    
    Attributes:
    - TEMP: Temperature profiles matrix.
    - SAL: Salinity profiles matrix.
    - SSH: Sea Surface Height vector.
    - pca_temp: PCA model for temperature profiles.
    - pca_sal: PCA model for salinity profiles.
    - temp_pcs: Transformed temperature profiles using PCA.
    - sal_pcs: Transformed salinity profiles using PCA.
    """
    def __init__(self, n_components=15, input_params=None, max_depth = 2000, min_depth = 20):
        """
        
        Args:
        - path (str): File path to the dataset.
        - n_components (int): Number of PCA components to retain.
        """
        self.n_components = n_components
        self.data_path =  "/unity/g2/jmiranda/SubsurfaceFields/Data/ARGO_GoM_20220920.mat"
        self.aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
        self.sst_folder   = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
        self.sss_folder   = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
        
        self.max_depth = max_depth
        self.min_depth = min_depth # data quality is poor above 20m
        
        self.data = mat73.loadmat(self.data_path)
        self.TIME = [datenum_to_datetime(datenum) for datenum in self.data['TIME']]
        
        # self.TIME = data['TIME']
        self.LAT = self.data['LAT']
        self.LON = self.data['LON']
        self.SH1950 = self.data['SH1950']
        self.min_lat = 18.0
        self.max_lat = 31.0
        self.min_lon = -98.0
        self.max_lon = -81.0
        
        self.input_params = input_params
        
        self.SSS, self.SST, self.AVISO_ADT = self._load_satellite_data()
        self.satSSS, self.satSST, self.sat_ADT = np.copy(self.SSS), np.copy(self.SST), np.copy(self.AVISO_ADT) #backup
        
        # self.adjust_ADT()
        
        self.valid_mask = self._get_valid_mask(self.data)
        valid_mask = self.valid_mask
        self.TEMP, self.SAL, self.AVISO_ADT, self.SST, self.SSS, self.TIME, self.LAT, self.LON, self.SH1950 = self._filter_and_fill_data(self.data, valid_mask)
        
        # Applying PCA
        self.temp_pcs, self.pca_temp = self._apply_pca(self.TEMP, self.n_components)
        self.sal_pcs, self.pca_sal = self._apply_pca(self.SAL, self.n_components)
        
    # def adjust_ADT(self):
    #     #  Remove the daily mean ADT from the AVISO_ADT
        
    #     gom_mean = xr.load_dataset('/unity/g2/jmiranda/SubsurfaceFields/Data/gom_mean_adt_2013_2022.nc')
    #     self.mean_adt = gom_mean.gom_mean_adt.values
    #     self.time_mean_adt = np.floor(gom_mean.time.values)

    #     for i, t in enumerate(self.data['TIME']):
    #         t = np.floor(t)
    #         i_match = self.time_mean_adt == t
    #         if np.sum(i_match) == 0:
    #             print(f"No match: idx: {i}\t .nc time: {t}")
    #         elif np.sum(i_match) > 1:
    #             print(f"Multiple matches at dx: {i}\t .nc time: {t}")
    #         # else:
    #         #     print(f"Single match: {i} \t {t} \t {mean_adt[i_match]} ")
                
    #         self.AVISO_ADT[i] = self.AVISO_ADT[i] - self.mean_adt[i_match]
    
    def reload(self):
        # in case we want to change parameters...
        self.SSS, self.SST, self.AVISO_ADT = np.copy(self.satSSS), np.copy(self.satSST), np.copy(self.sat_ADT)
        self.data = mat73.loadmat(self.data_path)
        # self.adjust_ADT()

        valid_mask = self._get_valid_mask(self.data)
        self.TEMP, self.SAL, self.AVISO_ADT, self.SST, self.SSS, self.TIME, self.LAT, self.LON, self.SH1950 = self._filter_and_fill_data(self.data, valid_mask)
        
        # Applying PCA
        self.temp_pcs, self.pca_temp = self._apply_pca(self.TEMP, self.n_components)
        self.sal_pcs, self.pca_sal = self._apply_pca(self.SAL, self.n_components)

    def _load_satellite_data(self):
        """
        Load Sea Surface Temperature (SST), Sea Surface Salinity (SSS), and Sea Surface Height (SSH) data.
        
        This method loads and interpolates satellite data for SST, SSS, and SSH within a specified geographic bounding box
        and time range. It also includes an optional debugging mode that logs data loading failures.

        Returns:
            tuple: Tuple containing arrays for SSS, SST, and AVISO data.
        """
        aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
        sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
        sss_folder = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
        min_lat = 18.0
        max_lat = 31.0
        min_lon = -98.0
        max_lon = -81.0
        ex_lon = -90.0
        ex_lat = 23.0
        bbox=(min_lat, max_lat, min_lon, max_lon)
        sss_data, sst_data, aviso_data = np.nan * np.ones(len(self.TIME)), np.nan * np.ones(len(self.TIME)), np.nan * np.ones(len(self.TIME))
        error_log = []

        for idx, c_date in tqdm(enumerate(sorted(set(self.TIME))), total=len(set(self.TIME)), desc="Loading Satellite Data"):
            date_idx = np.array([date_obj == c_date for date_obj in self.TIME])
            coordinates = np.array([self.LAT[date_idx], self.LON[date_idx]]).T
            
            # SSS data loading
            try:
                sss_datapoint, lats, lons = get_sss_by_date(sss_folder, c_date, bbox)
                interpolator = RegularGridInterpolator((lats, lons), sss_datapoint.sss_smap_40km.values, bounds_error=False, fill_value=None)
                sss_data[date_idx] = interpolator(coordinates)
                if (sss_data[date_idx] < 0).any() or (sss_data[date_idx] > 45).any():
                    sss_data[date_idx] = np.nan
                    if debug:
                        error_log.append({'Date': c_date, 'Parameter': 'SSS', 'Filename': sss_datapoint.filename, 'Reason': 'Invalid SSS values'})
            except Exception as e:
                if debug:
                    error_log.append({'Date': c_date, 'Parameter': 'SSS', 'Filename': None, 'Reason': str(e)})

            # AVISO data loading
            try:
                aviso_adt, aviso_lats, aviso_lons = get_aviso_by_date(aviso_folder, c_date, bbox)
                lons, lats = np.meshgrid(aviso_lons, aviso_lats)
                inclusion_mask = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
                exclusion_mask = (lats < ex_lat) & (lons > ex_lon)
                combined_mask = inclusion_mask & ~exclusion_mask
                daily_avg = np.nanmean(aviso_adt.adt.values[combined_mask])
                interpolator_ssh = RegularGridInterpolator((aviso_lats, aviso_lons), aviso_adt.adt.values, bounds_error=False, fill_value=None)
                aviso_data[date_idx] = interpolator_ssh(coordinates) - daily_avg
            except Exception as e:
                if debug:
                    error_log.append({'Date': c_date, 'Parameter': 'AVISO', 'Filename': None, 'Reason': str(e)})

            # SST data loading
            try:
                sst_date, sst_lats, sst_lons = get_sst_by_date(sst_folder, c_date, bbox)
                interpolator_sst = RegularGridInterpolator((sst_lats, sst_lons), sst_date.analysed_sst.values[0], bounds_error=False, fill_value=None)
                sst_data[date_idx] = interpolator_sst(coordinates)
            except Exception as e:
                if debug:
                    error_log.append({'Date': c_date, 'Parameter': 'SST', 'Filename': None, 'Reason': str(e)})

        if debug and error_log:
            # You can choose to save this as a CSV or any other format
            print("Error log:", error_log)

        return sss_data, sst_data, aviso_data

    def __getitem__(self, idx):
        """
        Args:
        - idx (int): Index of the profile.

        Returns:
        - tuple: input values and concatenated PCA components for temperature and salinity.
        """
        
        inputs = []
        
        if self.input_params["timecos"]:
            inputs.append(np.cos(2*np.pi*(self.TIME[idx]%365)/365)) 
            
        if self.input_params["timesin"]:
            inputs.append(np.sin(2*np.pi*(self.TIME[idx]%365)/365))  
        
        if self.input_params["latcos"]:
            inputs.append(np.cos(2*np.pi*(self.LAT[idx]/180)))

        if self.input_params["latsin"]:
            inputs.append(np.sin(2*np.pi*(self.LAT[idx]/180)))  

        if self.input_params["loncos"]:
            inputs.append(np.cos(2*np.pi*(self.LON[idx]/360)))  
            
        if self.input_params["loncos"]:
            inputs.append(np.sin(2*np.pi*(self.LON[idx]/360)))
            
        if self.input_params["sat"]:                
            if self.input_params["sss"]:
                # inputs.append(self.SAL[0, idx])
                inputs.append(self.SSS[idx])

            if self.input_params["sst"]:
                inputs.append(self.SST[idx] - 273.15) # convert from Kelvin to Celsius
                
            if self.input_params["ssh"]:
                # inputs.append(self.SH1950[idx]) #Uses profile SSH
                inputs.append(self.AVISO_ADT[idx]) #Uses satellite SSH
        else:
            if self.input_params["sss"]:
                inputs.append(self.SAL[0, idx])
                # inputs.append(self.SSS[idx])

            if self.input_params["sst"]:
                inputs.append(self.TEMP[0, idx])  # First value of temperature profile
                # inputs.append(self.SST[idx])
                
            if self.input_params["ssh"]:
                inputs.append(self.SH1950[idx]) #Uses profile SSH
                # inputs.append(self.AVISO_ADT[idx]) #Uses satellite SSH
            
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        profiles = torch.tensor(np.hstack([self.temp_pcs[:, idx], self.sal_pcs[:, idx]]), dtype=torch.float32)
        return inputs_tensor, profiles
    
    def _get_valid_mask(self, data):
        """Internal method to get mask of valid profiles based on missing values."""
        
        ssh_mask = ~np.isnan(self.AVISO_ADT)
        sst_mask = ~np.isnan(self.SST)
        sss_mask = ~np.isnan(self.SSS)
        
        combined_mask = np.logical_and(sst_mask, ssh_mask)
        print(f"Filtered dataset (sal/temp/ssh/sst) contains {np.sum(combined_mask)} profiles.")
        combined_mask = np.logical_and(combined_mask, sss_mask)
        print(f"Filtered dataset (sal/temp/ssh/sst/sss) contains {np.sum(combined_mask)} profiles.")
        
        return combined_mask
    
    def _filter_and_fill_data(self, data, valid_mask):
        """Internal method to filter data using the mask and fill missing values."""
        TEMP = data['TEMP'][self.min_depth:self.max_depth+1, valid_mask]
        SAL =  data['SAL'][self.min_depth:self.max_depth+1, valid_mask]
        LAT =  data['LAT'][valid_mask]
        LON =  data['LON'][valid_mask]
        ADT =  data['SH1950'][valid_mask]
        TIME = data['TIME'][valid_mask]
        
        SSH =  self.AVISO_ADT[valid_mask]
        SST =  self.SST[valid_mask]
        SSS =  self.SSS[valid_mask]
        
        # Fill missing values using interpolation
        for i in range(TEMP.shape[1]):
            # let's fill nans at the beginning and end of the profiles
            first_nan = np.where(~np.isnan(TEMP[:, i]))[0][0]
            last_nan = np.where(~np.isnan(TEMP[:, i]))[0][-1]
            if np.isnan(TEMP[0, i]):
                TEMP[:first_nan, i] = TEMP[first_nan, i]
            if np.isnan(TEMP[-1, i]):
                TEMP[last_nan:, i] = TEMP[last_nan, i]
            
            valid_temp_idx = np.where(~np.isnan(TEMP[:, i]))[0]
            TEMP[:, i] = np.interp(range(TEMP.shape[0]), valid_temp_idx, TEMP[valid_temp_idx, i])
            valid_sal_idx = np.where(~np.isnan(SAL[:, i]))[0]
            SAL[:, i] = np.interp(range(SAL.shape[0]), valid_sal_idx, SAL[valid_sal_idx, i])
        
        return TEMP, SAL, SSH, SST, SSS, TIME, LAT, LON, ADT
    
    def _apply_pca(self, data, n_components):
        """Internal method to apply PCA transformation to the data."""
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(data.T).T
        return pcs, pca
    
    def __len__(self):
        """Returns number of profiles in the dataset."""
        return self.TEMP.shape[1]
    
    def inverse_transform(self, pcs):
        """
        Inverse the PCA transformation.

        Args:
        - pcs (numpy.ndarray): Concatenated PCA components for temperature and salinity.

        Returns:
        - tuple: Inversed temperature and salinity profiles.
        """
        temp_profiles = self.pca_temp.inverse_transform(pcs[:, :self.n_components]).T
        sal_profiles = self.pca_sal.inverse_transform(pcs[:, self.n_components:]).T
        return temp_profiles, sal_profiles
    
    def get_profiles(self, indices, pca_approx=False):
        """
        Returns temperature and salinity profiles for the given indices.

        Args:
        - indices (list or numpy.ndarray): List of indices for which profiles are needed.
        - pca_approx (bool): Flag to return PCA approximated profiles if True, 
                             or original profiles if False.

        Returns:
        - numpy.ndarray: concatenated temperature and salinity profiles in the required format for visualization.
        """
        indices = np.atleast_1d(indices)
        if pca_approx:
            # Concatenate temp and sal PCA components for the given indices
            concatenated_pcs = np.hstack([self.temp_pcs[:, indices].T, self.sal_pcs[:, indices].T])
            # Obtain PCA approximations using the concatenated components
            temp_profiles, sal_profiles = self.inverse_transform(concatenated_pcs)
        else:
            temp_profiles = self.TEMP[:, indices]
            sal_profiles = self.SAL[:, indices]

        # Stack along the third dimension
        profiles_array = np.stack([temp_profiles, sal_profiles], axis=1)

        return profiles_array
    
    def calc_gem(self, ignore_indices, degree=7, sat_ssh=False):
        """
        Calculates this dataset's polyfits for the GEM profiles for each month.

        Args:
        - degree: Degree of the polynomial fit. Default is 7.
        - sat_ssh: Flag to use satellite SSH instead of profile SSH. Uses measured SSH as default.

        Returns:
        - nothing, but saves the polyfits in the attributes `self.gem_T_polyfits` and `self.gem_S_polyfits` for each month.
        """
        
        self.pressure_grid = np.arange(self.min_depth, self.max_depth + 1)
        # Initialize dictionaries to hold polyfits for each month
        self.gem_T_polyfits = {}
        self.gem_S_polyfits = {}
        
        mask = np.ones(len(self.SH1950), dtype=bool)
        mask[ignore_indices] = False
        
        steric_height = self.SH1950[mask]
        SSH = self.AVISO_ADT[mask]
        TEMP = self.TEMP[:, mask]
        SAL = self.SAL[:, mask]
        TIME = np.array(self.TIME)[mask]  # Apply mask to TIME
        
        self.gem_slope, self.gem_intercept, _, _, _ = linregress(steric_height, SSH)
                
        sort_idx = np.argsort(steric_height)
            
        sh_sorted = steric_height[sort_idx] + np.arange(len(steric_height)) * 1e-10  # add small number to avoid duplicate values
        temp_sorted = TEMP[:, sort_idx]
        sal_sorted = SAL[:, sort_idx]
        time_sorted = TIME[sort_idx]  # Sort TIME based on steric_height sorting
        
        # Convert sorted TIME to months
        months_sorted = [int((datenum_to_datetime(datenum).month -1)/3) for datenum in time_sorted]
        
        # Start time
        start_time = time.perf_counter()
        
        # Iterate over each month
        for month in set(months_sorted):
            self.gem_T_polyfits[month] = []
            self.gem_S_polyfits[month] = []
            
            # Indices for the current month
            month_indices = [i for i, m in enumerate(months_sorted) if m == month]
            
            # For each pressure level
            for i, p in enumerate(self.pressure_grid):
                # Filter data for the current month
                TEMP_at_p = temp_sorted[i, month_indices]
                SAL_at_p = sal_sorted[i, month_indices]
                sh_at_p = sh_sorted[month_indices]
                
                # Polynomial fit for the current month
                TEMP_polyfit = Polynomial.fit(sh_at_p, TEMP_at_p, degree)
                SAL_polyfit = Polynomial.fit(sh_at_p, SAL_at_p, degree)
                
                # Append the polynomial fit to the lists for the current month
                self.gem_T_polyfits[month].append(TEMP_polyfit)
                self.gem_S_polyfits[month].append(SAL_polyfit)
                
        end_time = time.perf_counter()
        # Calculate elapsed time
        elapsed_time = (end_time - start_time)
        print(f"GEM fit: {elapsed_time:.2f} seconds.")
                
        return
    
    def get_gem_profiles(self, indices, sat_ssh=True):
        '''
        Generates the GEM profiles for the given indices, month by month.

        Args:
        - indices (list or numpy.ndarray): Indices for which profiles are needed.
        - sat_ssh (bool): Flag to use satellite SSH instead of profile SSH. Uses measured SSH as default.

        Returns:
        - numpy.ndarray: concatenated temperature and salinity profiles in the required format for visualization.
        '''
        
        # Initialize arrays to hold GEM profiles
        temp_GEM = np.empty((len(indices), self.max_depth+1-self.min_depth))
        sal_GEM = np.empty((len(indices), self.max_depth+1-self.min_depth))
        temp_GEM[:] = np.nan  # Initialize with NaNs
        sal_GEM[:] = np.nan

        for idx, index in enumerate(indices):
            # Determine the month for the current index
            month = int((datenum_to_datetime(self.TIME[index]).month - 1)/3)

            # Check if there are polyfits for this month
            if month not in self.gem_T_polyfits:
                continue  # Skip if no polyfits available for the month

            # Select SSH based on the sat_ssh flag
            if sat_ssh:
                ssh = (self.AVISO_ADT[index] - self.gem_intercept) / self.gem_slope
            else:
                ssh = self.SH1950[index]

            # For each pressure level
            for i, p in enumerate(self.pressure_grid):
                # Evaluate the fitted polynomials at the given SSH value
                temp_GEM[idx, i] = self.gem_T_polyfits[month][i](ssh)
                sal_GEM[idx, i] = self.gem_S_polyfits[month][i](ssh)

        # Interpolate missing values in temp_GEM and sal_GEM (same as before)
        for array in [temp_GEM, sal_GEM]:
            for row in range(array.shape[0]):
                valid_mask = ~np.isnan(array[row])
                if not valid_mask.any():  # skip rows with only NaNs
                    continue

                array[row] = np.interp(np.arange(array.shape[1]), np.where(valid_mask)[0], array[row, valid_mask])

                # If NaNs at the start, fill with the first non-NaN value
                first_valid_idx = valid_mask.argmax()
                array[row, :first_valid_idx] = array[row, first_valid_idx]
                
                # If NaNs at the end, fill with the last non-NaN value
                last_valid_idx = len(array[row]) - valid_mask[::-1].argmax() - 1
                array[row, last_valid_idx+1:] = array[row, last_valid_idx]

        return temp_GEM, sal_GEM

    def get_inputs(self, idx):
        sst_inputs = self.TEMP[0, idx]
        ssh_inputs = self.AVISO_ADT[idx]
        return sst_inputs, ssh_inputs
    
    def get_lat_lon_date(self, idx):
        lat = self.LAT[idx]
        lon = self.LON[idx]
        date = self.TIME[idx]
        return lat, lon, date
    
    def get_pca_weights(self):
        """
        Get the concatenated vector of the variance represented by each PC of the temperature and salinity datasets,
        divided by the total variance of each PCS.

        Returns:
        - numpy.ndarray: Concatenated vector of variances for temperature and salinity PCs.
        """
        temp_variance = self.pca_temp.explained_variance_ratio_ / self.temp_pcs.var(axis=1)
        sal_variance = self.pca_sal.explained_variance_ratio_ / self.sal_pcs.var(axis=1)
        concatenated_variance = np.concatenate([temp_variance, sal_variance])
        return concatenated_variance

class PredictionModel(nn.Module):
    """
    Neural Network model for predicting temperature and salinity profiles based on sea surface height (SSH).

    Attributes:
    - model (nn.Sequential): Sequential model containing layers defined by `layers_config`.

    Parameters:
    - input_dim (int): Dimension of the input feature(s). Default is 1 (for SSH).
    - layers_config (list of int): List where each element represents the number of neurons in 
                                   a respective layer. Default is [512, 256].
    - output_dim (int): Dimension of the output. Default is 30 (15 components for TEMP and 15 for SAL).

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the model.
    """

    def __init__(self, input_dim=1, layers_config=[512, 256], output_dim=30, dropout_prob = 0.5):
        super(PredictionModel, self).__init__()
        
        # Construct layers based on the given configuration
        layers = []
        prev_dim = input_dim
        for neurons in layers_config:
            layers.append(nn.Linear(prev_dim, neurons))
            layers.append(nn.ReLU())
            # layers.append(nn.Tanh())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob)) # added dropout
            prev_dim = neurons
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)      
    
    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
        - torch.Tensor: Model's predictions of shape (batch_size, output_dim).
        """
        # print(f"x shape: {x.shape}")
        return self.model(x)
    
def split_dataset(dataset, train_size, val_size, test_size, batch_size=32, use_batches=True):
    """
    Splits the dataset into training, validation, and test sets.
    
    Parameters:
    - dataset: The entire dataset to be split.
    - train_size, val_size, test_size: Proportions for splitting. They should sum to 1.
    
    Returns:
    - train_dataset, val_dataset, test_dataset: Split datasets.
    """
    total_size = len(dataset)
    train_len = int(total_size * train_size)
    val_len = int(total_size * val_size)
    test_len = total_size - train_len - val_len

    return random_split(dataset, [train_len, val_len, test_len])

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=10):
    """
    Train the model with early stopping and CUDA support.

    Parameters:
    - model: the PyTorch model.
    - train_loader: the DataLoader for training data.
    - val_loader: the DataLoader for validation data.
    - criterion: the loss function.
    - optimizer: the optimizer.
    - device: device to which data and model should be moved before training.
    - epochs: number of total epochs to train.
    - patience: number of epochs to wait for improvement before stopping.

    Returns:
    - model: trained model.
    """
    model.to(device)
    best_val_loss = float('inf')
    best_weights = None  # To store best model weights
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        
        # Validation loss
        avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        
        if epoch == 0 or epoch%10==9:
            print(f"Epoch [{(epoch + 1):4.0f}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Patience left: {(100*(patience - no_improve_count)/patience):3.0f}% | Best: {best_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = model.state_dict()  # Save the model weights
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at Epoch {epoch + 1}")
                break

    model.load_state_dict(best_weights)  # Load the best model weights
    return model

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on the provided data with CUDA support.

    Parameters:
    - model: the PyTorch model.
    - dataloader: the DataLoader for evaluation data.
    - criterion: the loss function.
    - device: device to which data and model should be moved before evaluation.

    Returns:
    - avg_loss: average loss over the dataset.
    """
    model.to(device)
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def get_predictions(model, dataloader, device):
    """
    Get model's predictions on the provided data with CUDA support.

    Parameters:
    - model: the PyTorch model.
    - dataloader: the DataLoader for the data.
    - device: device to which data and model should be moved before getting predictions.

    Returns:
    - predictions: model's predictions.
    """
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in dataloader:
            #inputs here is a list
            # print(type(inputs))
            inputs = inputs[0].to(device)  # Getting only the input features, ignoring labels
            # now it has shape (batch_size, input_dim)
            # print(f"inputs shape: {inputs.shape}")
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)

def get_inputs(dataloader, device):
    """
    Get inputs from the provided dataloader with CUDA support.

    Parameters:
    - dataloader: the DataLoader for the data.
    - device: device to which data should be moved.

    Returns:
    - all_inputs: list of inputs from the dataloader.
    """
    all_inputs = []

    for inputs in dataloader:
        inputs = inputs[0].to(device)  # Getting only the input features, ignoring labels
        all_inputs.extend(inputs.cpu().numpy())

    return np.array(all_inputs)

def predict_with_numpy(model, numpy_input, device=DEVICE):
    # Convert numpy array to tensor
    tensor_input = torch.tensor(numpy_input, dtype=torch.float32)
    
    # Check if CUDA is available and move tensor to the appropriate device
    if device == "cuda" and torch.cuda.is_available():
        tensor_input = tensor_input.cuda()
        model = model.cuda()
    
    # Make sure the model is in evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(tensor_input)
    
    # Convert predictions back to numpy (if on GPU, move to CPU first)
    numpy_predictions = predictions.cpu().numpy()
    
    return numpy_predictions

def inverse_transform(pcs, pca_temp, pca_sal, n_components):
    """
    Inverse the PCA transformation.

    Args:
    - pcs (numpy.ndarray): Concatenated PCA components for temperature and salinity.
    - pca_temp, pca_sal: PCA models for temperature and salinity respectively.
    - n_components (int): Number of PCA components.

    Returns:
    - tuple: Inversed temperature and salinity profiles.
    """
    temp_profiles = pca_temp.inverse_transform(pcs[:, :n_components]).T
    sal_profiles = pca_sal.inverse_transform(pcs[:, n_components:]).T
    return temp_profiles, sal_profiles

## Custom Loss
class WeightedMSELoss(nn.Module):
    """
    The code defines several loss functions for use in a PCA-based model, including a weighted MSE loss
    and a combined PCA loss.
    
    @param n_components The parameter `n_components` represents the number of principal components to
    consider in the PCA loss. It determines the dimensionality of the PCA space for both temperature and
    salinity profiles.
    @param device The "device" parameter in the code refers to the device on which the computations will
    be performed. It can be either "cuda" for GPU acceleration or "cpu" for CPU computation.
    @param weights The "weights" parameter is a list of weights that are used to assign different
    importance to each element in the loss calculation. These weights are used in the WeightedMSELoss
    class to multiply the squared differences between the predicted and true values. The weights are
    normalized so that they sum up to
    
    @return The `forward` method of the `CombinedPCALoss` class returns the combined loss, which is the
    sum of the PCA loss and the weighted MSE loss.
    """
    def __init__(self, weights, device):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, input, target):
        squared_diff = (input - target) ** 2
        weighted_squared_diff = self.weights * squared_diff
        loss = weighted_squared_diff.mean()
        return loss

def genWeightedMSELoss(n_components, device, weights): # min test loss ~ 2
    # Normalizing weights so they sum up to 1
    normalized_weights = weights / np.sum(weights)
    return WeightedMSELoss(normalized_weights, device)
   
class PCALoss(nn.Module): #min test loss ~ 13
    def __init__(self, temp_pca, sal_pca, n_components):
        super(PCALoss, self).__init__()
        self.n_components = n_components
        self.n_samples = len(temp_pca)
        # convert PCS to tensors
        self.temp_pca_components = torch.nn.Parameter(torch.from_numpy(np.asarray(temp_pca.temp_pcs)).float().to(DEVICE), requires_grad=False)
        self.sal_pca_components = torch.nn.Parameter(torch.from_numpy(np.asarray(sal_pca.sal_pcs)).float().to(DEVICE), requires_grad=False)

    def inverse_transform(self, pcs, pca_components):
        # Perform the inverse transform using PyTorch operations
        return torch.mm(pcs, pca_components) # + pca_mean

    def forward(self, pcs, targets):
        # Split the predicted and true pcs for temp and sal
        pred_temp_pcs, pred_sal_pcs = pcs[:, :self.n_components], pcs[:, self.n_components:]
        true_temp_pcs, true_sal_pcs = targets[:, :self.n_components], targets[:, self.n_components:]
        
        # Inverse transform the PCA components to get the profiles
        pred_temp_profiles = self.inverse_transform(pred_temp_pcs, self.temp_pca_components)
        pred_sal_profiles = self.inverse_transform(pred_sal_pcs, self.sal_pca_components)
        true_temp_profiles = self.inverse_transform(true_temp_pcs, self.temp_pca_components)
        true_sal_profiles = self.inverse_transform(true_sal_pcs, self.sal_pca_components)
        
        # Calculate the Avg Squared Error between the predicted and true profiles
        mse_temp = nn.functional.mse_loss(pred_temp_profiles, true_temp_profiles)
        mse_sal = nn.functional.mse_loss(pred_sal_profiles, true_sal_profiles)
        
        # Combine the MSE for temperature and salinity
        # total_mse = (mse_temp/(8.6) + mse_sal/(35.2))/self.n_samples # divide by the mean T and S values
        total_mse = (mse_temp/(37.86) + mse_sal/(0.28))/self.n_samples # divide by the T and S variances
        # total_mse = (mse_temp/(2.01) + mse_sal/(0.03))/self.n_samples # divide by the mean T and S values
        return total_mse
    
class CombinedPCALoss(nn.Module):
    def __init__(self, temp_pca, sal_pca, n_components, weights, device):
        super(CombinedPCALoss, self).__init__()
        self.pca_loss = PCALoss(temp_pca, sal_pca, n_components)
        self.weighted_mse_loss = genWeightedMSELoss(n_components, device, weights)

    def forward(self, pcs, targets):
        # Calculate the PCA loss
        pca_loss = self.pca_loss(pcs, targets)

        # Calculate the weighted MSE loss
        weighted_mse_loss = self.weighted_mse_loss(pcs, targets)

        # Combine the losses - Choose scaling factor
        combined_loss = (pca_loss/2.8294 + weighted_mse_loss/0.0255)/2 #here I divide by the individual minimum loss, and divide by two
        return combined_loss

def visualize_combined_results(true_values, gem_temp, gem_sal, predicted_values, sst_values, ssh_values, min_depth = 20, max_depth=2000, num_samples=5):
    # TODO: add date to plot
    """
    Visualize the true vs. predicted vs. GEM approximated values for a sample of profiles and their differences.

    Parameters:
    - true_values: ground truth temperature and salinity profiles.
    - gem_temp: GEM approximated temperature profiles.
    - gem_sal: GEM approximated salinity profiles.
    - predicted_values: model's predicted temperature and salinity profiles.
    - sst_values: Sea Surface Temperature values for each profile.
    - ssh_values: Sea Surface Height (adt) values for each profile.
    - num_samples: number of random profiles to visualize.

    Returns:
    - None (plots the results).
    """
    n_depths = max_depth + 1
    depth_levels = np.arange(min_depth, n_depths)
    population_size = true_values.shape[2]

    if num_samples == population_size:
        indices = np.arange(num_samples)
    else:
        indices = np.random.choice(int(population_size), num_samples, replace=False)

    for idx in indices:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # First row: Actual Profiles
        # Temperature profile
        axs[0][0].plot(gem_temp[idx], depth_levels, 'g', label="GEM Profile", alpha = 0.75)
        axs[0][0].plot(predicted_values[0][:, idx], depth_levels, 'r', label="NeSPReSO Profile", alpha = 0.75)
        axs[0][0].plot(true_values[:,0, idx], depth_levels, 'k', label="Target", linewidth = 0.7)
        axs[0][0].invert_yaxis()
        axs[0][0].set_title(f"Temperature Profile")
        axs[0][0].set_ylabel("Depth")
        axs[0][0].set_xlabel("Temperature")
        axs[0][0].legend(loc='lower right')
        axs[0][0].grid(color='gray', linestyle='--', linewidth=0.5)

        # Salinity profile
        axs[0][1].plot(gem_sal[idx], depth_levels, 'g', label="GEM Profile", alpha = 0.75)
        axs[0][1].plot(predicted_values[1][:, idx], depth_levels, 'r', label="NeSPReSO Profile", alpha = 0.75)
        axs[0][1].plot(true_values[:,1, idx], depth_levels, 'k', label="Target", linewidth = 0.7)
        axs[0][1].invert_yaxis()
        axs[0][1].set_title(f"Salinity Profile")
        axs[0][1].set_ylabel("Depth")
        axs[0][1].set_xlabel("Salinity")
        axs[0][1].legend(loc='lower right')
        axs[0][1].grid(color='gray', linestyle='--', linewidth=0.5)

        # Second row: Differences
        gem_temp_dif = gem_temp[idx]-true_values[:,0, idx]
        gem_sal_dif = gem_sal[idx]-true_values[:,1, idx]
        nn_temp_dif = predicted_values[0][:, idx]-true_values[:,0, idx]
        nn_sal_dif = predicted_values[1][:, idx]-true_values[:,1, idx]
        
        axs[1][0].plot(np.abs(gem_temp_dif), depth_levels, 'g', label="GEM Profile", alpha = 0.75)
        axs[1][0].plot(np.abs(nn_temp_dif), depth_levels, 'r', label="NeSPReSO Profile", alpha = 0.75)
        axs[1][0].axvline(0, color='k', linestyle='--', linewidth=0.5)
        axs[1][0].invert_yaxis()
        axs[1][0].set_title(f"Temperature Differences")
        axs[1][0].set_ylabel("Depth")
        axs[1][0].set_xlabel("Absolute difference [°C]")
        axs[1][0].legend(loc='best')
        axs[1][0].grid(color='gray', linestyle='--', linewidth=0.5)

        # Salinity difference
        axs[1][1].plot(np.abs(gem_sal_dif), depth_levels, 'g', label="GEM Profile", alpha = 0.75)
        axs[1][1].plot(np.abs(nn_sal_dif), depth_levels, 'r', label="NeSPReSO Profile", alpha = 0.75)
        axs[1][1].axvline(0, color='k', linestyle='--', linewidth=0.5)
        axs[1][1].invert_yaxis()
        axs[1][1].set_title(f"Salinity Differences")
        axs[1][1].set_ylabel("Depth")
        axs[1][1].set_xlabel("Absolute difference [PSU]")
        axs[1][1].legend(loc='best')
        axs[1][1].grid(color='gray', linestyle='--', linewidth=0.5)

        gem_temp_se_individual = np.sqrt(np.mean(gem_temp_dif**2))
        gem_sal_se_individual = np.sqrt(np.mean(gem_sal_dif**2))
        nn_temp_se_individual = np.sqrt(np.mean(nn_temp_dif**2))
        nn_sal_se_individual = np.sqrt(np.mean(nn_sal_dif**2))

        accuracy_gain_temp = 100*(gem_temp_se_individual - nn_temp_se_individual) / gem_temp_se_individual
        accuracy_gain_sal = 100*(gem_sal_se_individual - nn_sal_se_individual) / gem_sal_se_individual

        # Add sst, ssh and accuracy gain information to the suptitle
        plt.suptitle(f"Profile {idx} - SST: {sst_values[idx]:.2f}, SSH (adt): {ssh_values[idx]:.2f}\n"
                     f"T prediction improvement: {accuracy_gain_temp:.2f}%, S prediction improvement: {accuracy_gain_sal:.2f}%", fontsize=16)

        plt.tight_layout()
        plt.show()

    # RMSE Calculations and Accuracy Gain
    gem_temp_errors = (gem_temp.T - true_values[:, 0, :]) ** 2
    gem_sal_errors = (gem_sal.T - true_values[:, 1, :]) ** 2

    nn_temp_errors = (predicted_values[0][:, :] - true_values[:, 0, :]) ** 2
    nn_sal_errors = (predicted_values[1][:, :] - true_values[:, 1, :]) ** 2
        
    gem_temp_se = np.sqrt(np.mean(gem_temp_errors))
    gem_sal_se = np.sqrt(np.mean(gem_sal_errors))

    nn_temp_se = np.sqrt(np.mean(nn_temp_errors))
    nn_sal_se = np.sqrt(np.mean(nn_sal_errors))

    accuracy_gain_temp = 100*(gem_temp_se-nn_temp_se)/gem_temp_se
    accuracy_gain_sal = 100*(gem_sal_se-nn_sal_se)/gem_sal_se
    
    print(f"NeSPReSO Average temperature RMSE: {nn_temp_se:.3f}°C")
    print(f"NeSPReSO Average salinity RMSE: {nn_sal_se:.3f} PSU")
    print(f"GEM Average temperature RMSE: {gem_temp_se:.3f}°C")
    print(f"GEM Average salinity RMSE: {gem_sal_se:.3f} PSU")
    
    gem_temp_errors = (gem_temp.T[150:,:] - true_values[150:, 0, :]) ** 2
    gem_sal_errors = (gem_sal.T[150:,:] - true_values[150:, 1, :]) ** 2

    nn_temp_errors = (predicted_values[0][150:, :] - true_values[150:, 0, :]) ** 2
    nn_sal_errors = (predicted_values[1][150:, :] - true_values[150:, 1, :]) ** 2

    gem_temp_se = np.sqrt(np.mean(gem_temp_errors))
    gem_sal_se = np.sqrt(np.mean(gem_sal_errors))

    nn_temp_se = np.sqrt(np.mean(nn_temp_errors))
    nn_sal_se = np.sqrt(np.mean(nn_sal_errors))

    accuracy_gain_temp = 100*(gem_temp_se-nn_temp_se)/gem_temp_se
    accuracy_gain_sal = 100*(gem_sal_se-nn_sal_se)/gem_sal_se

def matlab2datetime(matlab_datenum):
    python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days=366)
    return python_datetime

def filter_by_season(data, dates, season):
    SEASONS = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Fall": [9, 10, 11]
    }
    months = SEASONS[season]
    indices = [i for i, date in enumerate(dates) if matlab2datetime(date).month in months]
    return [data[i] for i in indices]

def seasonal_plots(lat_val, lon_val, dates_val, original_profiles, gem_temp, gem_sal, val_predictions, sst_inputs, ssh_inputs, max_depth, num_samples):
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    total_samples = len(lat_val)
    indexes = np.arange(total_samples)
    for season in seasons:
        idx = np.array(filter_by_season(indexes, dates_val, season))
        print(season)
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 10))
        ax.set_global()
        ax.coastlines()
        # Setting plot limits to the Gulf of Mexico region
        ax.set_extent([-98, -80, 18, 31])
        scatter = ax.scatter(lon_val[idx], lat_val[idx], c=ssh_inputs[idx], cmap='viridis', edgecolors='k', linewidth=0.5, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.02, shrink=1)
        cbar.set_label("SSH Value")

        ax.set_title(f"{season} profiles in validation", fontsize=16)
        plt.show()
        
        # Now plot some samples from this season
        sliced_val_pred = [array[:, idx] for array in val_predictions]
        visualize_combined_results(original_profiles[:,:, idx], gem_temp[idx], gem_sal[idx], sliced_val_pred, sst_inputs[idx], ssh_inputs[idx], max_depth = max_depth, num_samples=num_samples)
     
def calculate_bias(true_values, predicted_values, gem_temp, gem_sal):
    
    depths = np.arange(min_depth, max_depth+1)
    gem_temp_bias = (gem_temp.T - true_values[:, 0, :])
    gem_sal_bias = (gem_sal.T - true_values[:, 1, :])

    nn_t_bias = (predicted_values[0][:, :] - true_values[:, 0, :])
    nn_s_bias = (predicted_values[1][:, :] - true_values[:, 1, :])
    
    return nn_t_bias, nn_s_bias, gem_temp_bias, gem_sal_bias
    
def calculate_average_in_bin(lon_bins, lat_bins, lon_val, lat_val, bias_values, dpt_range = np.arange(0, 1801), is_rmse=True):
    # dpt_min, dpt_max = dpt_range
    avg_rmse_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    num_prof_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    
    if is_rmse:
        input_vals = bias_values**2
    else:
        input_vals = bias_values

    for i in range(len(lon_bins)-1):
        for j in range(len(lat_bins)-1):
            # Find points that fall into the current bin
            # in_bin = (lon_val == lon_bins[i]) & (lat_val == lat_bins[j])
            in_bin = (lon_val >= lon_bins[i]) & (lon_val < lon_bins[i+1]) & (lat_val >= lat_bins[j]) & (lat_val < lat_bins[j+1])
            # Calculate average RMSE for points in the bin
            rmses = input_vals[dpt_range,:]
            rmses = rmses[:,in_bin]
            avg_rmse_grid[j, i] = np.mean(rmses)
            num_prof_grid[j, i] = np.sum(in_bin)
    
    if is_rmse:
        return np.sqrt(avg_rmse_grid), num_prof_grid

    else:
        return avg_rmse_grid, num_prof_grid

def plot_bin_map(lon_bins, lat_bins, avg_rmse_nn, num_prof, title_prefix, variable_plotted):
    # Calculate centers of the bins
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    
    vmin = 0

    # Set up color maps and limits
    if title_prefix.startswith("Temperature"):
        units = "[°C]"
        if variable_plotted == "Bias":
            cmap = "coolwarm"
            vmax = 1
            vmin = -1
        
        else:
            cmap = "YlOrRd"
            vmax = 2
            vmin = 0.3
            
    else:
        units = "[PSU]"
        if variable_plotted == "Bias":
            cmap = "PiYG_r"
            vmax = 0.2
            vmin = -0.2
            
        else:
            cmap = "PuBuGn"
            vmax = 0.35
            vmin = 0            
        
    # Create subplot grid
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the maps
    plot_rmse_on_ax(ax1, lon_centers, lat_centers, avg_rmse_nn, num_prof, f"NeSPReSO Average {variable_plotted} - {title_prefix}")

    pcm = ax1.pcolormesh(lon_centers, lat_centers, avg_rmse_nn, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax1, orientation="vertical", pad=0.04, fraction=0.465*(1/15), label=f"Average {variable_plotted} {units}")
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    # Set x and y ticks, 
    ax1.set_xticks(np.arange(-99, -81, 1))
    ax1.set_yticks(np.arange(18, 30, 1))
    #add grid
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()
    
def plot_rmse_on_ax(ax, lon_centers, lat_centers, avg_rmse_grid, num_prof, title):
    ax.set_extent([-99, -81, 18, 30])  # Set to your area of interest
    ax.coastlines()

    pcm = ax.pcolormesh(lon_centers, lat_centers, avg_rmse_grid, cmap='coolwarm', vmin=-3, vmax=3)
    ax.set_title(title, fontsize=18)

    # Annotate each cell with the average RMSE value
    for i, lon in enumerate(lon_centers):
        for j, lat in enumerate(lat_centers):
            value = avg_rmse_grid[j, i]
            number = num_prof[j, i]
            if not np.isnan(value):  # Check if the value is not NaN, and if there are more than 2 profiles in the bin
                ax.text(lon, lat+0.2, f'{number:.0f}', color='gray', ha='center', va='center', fontsize=12, transform=ccrs.PlateCarree())
                ax.text(lon, lat-0.2, f'{value:.2f}', color='black', ha='center', va='center', fontsize=12, transform=ccrs.PlateCarree())

def plot_comparison_maps(lon_centers, lat_centers, avg_var_nn, avg_var_compare, title_prefix, name_compare, variable_name = "RMSE"):
    # Calculate the difference
    avg_var_diff = np.abs(avg_var_nn) - np.abs(avg_var_compare)

    # Set up color maps and limits
    if title_prefix == "temperature":
        units = "[°C]"
        if variable_name == "Bias":
            cmap = "coolwarm"
            n_plots = 3
            vmax = 1
            vmin = -1
        else:
            cmap = "YlOrRd"
            n_plots = 3
            vmax = 2
            vmin = 0.3
            
    else:
        units = "[PSU]"
        if variable_name == "Bias":
            cmap = "PiYG_r"
            n_plots = 3
            vmax = 0.2
            vmin = -0.2
        else:
            cmap = "PuBuGn"
            n_plots = 3
            vmax = 0.35
            vmin = 0

    # Custom colormap for difference plot
    diff_cmap = "bwr"
    norm_diff = plt.Normalize(-vmax, vmax)

    # Create subplot grid
    fig, axes = plt.subplots(1, n_plots, figsize=(n_plots*10, 15), subplot_kw={'projection': ccrs.PlateCarree()})

    # Titles for each subplot
    if variable_name == "Bias":
        dif_name = "Difference of magnitude"
    else:
        dif_name = "Difference"
    titles = [f"NeSPReSO", f"{name_compare}", f"{dif_name} (lower is better)"]

    # Function to add values to bins
    def annotate_bins(ax, data):
        for i, lon in enumerate(lon_centers):
            for j, lat in enumerate(lat_centers):
                value = data[j, i]
                if not np.isnan(value):
                    ax.text(lon, lat, f'{value:.2f}', color='black', ha='center', va='center', fontsize=9, transform=ccrs.PlateCarree())

    # Plotting NN RMSE, ISOP RMSE, and Difference
    for i, (data, title) in enumerate(zip([avg_var_nn, avg_var_compare, avg_var_diff], titles)):
        if i<2:
            pcm = axes[i].pcolormesh(lon_centers, lat_centers, data, cmap=cmap, vmin=vmin, vmax=vmax)
        elif i >= 2 and n_plots == 3:  # For the difference plot
            pcm_diff = axes[i].pcolormesh(lon_centers, lat_centers, data, cmap=diff_cmap, norm=norm_diff)
        # if i < n_plots:
        #     annotate_bins(axes[i], data)
        
    for i in range(n_plots):
        axes[i].set_title(titles[i], weight='bold')
        axes[i].coastlines()
        axes[i].set_xticks(np.arange(-99, -81, 2))
        axes[i].set_yticks(np.arange(18, 32, 2))
        axes[i].grid(color='gray', linestyle='--', linewidth=0.5)

    # Adding colorbar for the first two plots
    fig.colorbar(pcm, ax=axes[1], orientation="vertical", pad=0.04, fraction=0.0315)
    fig.suptitle(f"Average {title_prefix} {variable_name} by region", fontsize=28, y=0.705, fontweight="bold")

    if n_plots == 3:
        # Adding colorbar for the difference plot
        fig.colorbar(pcm_diff, ax=axes[2], orientation="vertical", pad=0.04, fraction=0.0305).set_label(label=f"{dif_name} {units}", size=14)
    
    plt.show()

def plot_residual_profiles_for_top_bins(lon_bins, lat_bins, lon_val, lat_val, nn_profiles, avg_rmse_grid, num_prof_grid, param, min_depth, max_depth, top_n=9):
    """
    Plots residual profiles for the top bins with the highest number of profiles.

    Parameters:
    - lon_bins, lat_bins: Arrays of longitude and latitude bin edges.
    - lon_val, lat_val: Arrays of longitude and latitude values for each profile.
    - nn_profiles, target_profiles: Arrays of neural network predicted and target profiles.
    - avg_rmse_grid, num_prof_grid: Grids of average RMSE and number of profiles per bin.
    - top_n: Number of top bins to plot. Default is 9 (for a 3x3 grid).
    """
    # Flatten the grid and sort bins by the number of profiles
    num_profiles_flat = num_prof_grid.flatten()
    sorted_indices = np.argsort(num_profiles_flat)[::-1][:top_n]

    # Set up the 3x3 subplot
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()

    for idx, ax in enumerate(axs):
        if idx >= len(sorted_indices):
            ax.axis('off')
            continue
        
        # Get the bin index
        bin_index = np.unravel_index(sorted_indices[idx], num_prof_grid.shape)
        j, i = bin_index

        # Find profiles in this bin
        in_bin = (lon_val >= lon_bins[i]) & (lon_val < lon_bins[i+1]) & (lat_val >= lat_bins[j]) & (lat_val < lat_bins[j+1])
        
        # Check if there are any profiles in the bin
        if np.any(in_bin):
            residuals = nn_profiles[:, in_bin].T

            # Plotting each residual profile
            for residual in residuals:
                ax.plot(residual, np.arange(min_depth,max_depth+1, 1), label=f'Lat: {lat_bins[j]}-{lat_bins[j+1]}, Lon: {lon_bins[i]}-{lon_bins[i+1]}', color='gray', linewidth=0.5)
        else:
            ax.axis('off')  # No data for this bin
            
        ax.axvline(x=0, color='k', linewidth=0.5)
        # Set title with lat/lon, number of profiles, and average RMSE
        ax.set_title(f'Bin: Lat: {lat_bins[j]:.0f} ~ {lat_bins[j+1]:.0f}, Lon: {lon_bins[i]:.0f} ~ {lon_bins[i+1]:.0f}\n'
                        f'Profiles: {num_prof_grid[j, i]}, Avg RMSE: {avg_rmse_grid[j, i]:.2f}')
        fig.suptitle(f'Residual profiles for {param} bins with the most profiles\n', fontsize=16, fontweight="bold")
        ax.set_xlabel('Residual')
        ax.set_ylabel('Depth')
        ax.invert_yaxis()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
def load_all_models(models_dir, device, input_dim, layers_config, n_components, dropout_prob):
    model_paths = sorted(glob.glob(os.path.join(models_dir, 'model_Test Loss: *.pth')))
    models = []
    for model_path in model_paths:
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize the model architecture
        model = PredictionModel(
            input_dim=input_dim, 
            layers_config=layers_config, 
            output_dim=n_components * 2, 
            dropout_prob=dropout_prob
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode
        models.append(model)
    return models
    
#%%
if __name__ == "__main__":
    # Configurable parameters
    bin_size = 1  # bin size in degrees
    n_components = 15
    layers_config = [512, 512]
    batch_size = 512
    min_depth = 0
    max_depth = 1800
    dropout_prob = 0.2
    epochs = 8000
    patience = 500
    learning_rate = 0.001
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    input_params = {
        "timecos": True,
        "timesin": True,
        "latcos": True,
        "latsin": True,
        "loncos": True,
        "lonsin": True,
        "sat": True,  # Use satellite data?
        "sst": True,  # First value of temperature if sat = false, OISST if sat = true
        "sss": True,  # similar to above
        "ssh": True   # similar to above
    }
    num_samples = 1  # profiles that will be plotted
    dataset_pickle_file = '/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/config_dataset_full.pkl'

    # Load or create the dataset
    if os.path.exists(dataset_pickle_file) and load_dataset_file:
        # Load data from the pickle file
        with open(dataset_pickle_file, 'rb') as file:
            data = pickle.load(file)
            full_dataset = data['full_dataset']
            full_dataset.n_components = n_components
            full_dataset.min_depth = min_depth
            full_dataset.max_depth = max_depth
            full_dataset.input_params = input_params
            if not load_trained_model:
                full_dataset.reload()
    else:
        # Load and split data
        full_dataset = TemperatureSalinityDataset(n_components=n_components, input_params=input_params, min_depth=min_depth, max_depth=max_depth)

        # Save data to the pickle file
        with open(dataset_pickle_file, 'wb') as file:
            data = {
                'min_depth': min_depth,
                'max_depth': max_depth,
                'epochs': epochs,
                'patience': patience,
                'n_components': n_components,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'dropout_prob': dropout_prob,
                'layers_config': layers_config,
                'input_params': input_params,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
                'full_dataset': full_dataset
            }
            pickle.dump(data, file)

    # Split the dataset
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, train_size, val_size, test_size)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    subset_indices = val_loader.dataset.indices
    full_dataset.calc_gem(subset_indices)

    # Compute the input dimension dynamically
    input_dim = sum(val for val in input_params.values()) - 1 * input_params['sat']

    # Check CUDA availability
    device = DEVICE
    
    # # Use multiple GPUs if available
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     multi_gpu = True
    # else:
    #     multi_gpu = False

    # Loss function using the variance of the PCA components divided by the varance of each PC as weights
    weights = full_dataset.get_pca_weights()

    # Print explained variance
    print(f"Explained Variance - T: {(sum(full_dataset.pca_temp.explained_variance_ratio_) * 100):.1f}% - S: {(100 * sum(full_dataset.pca_sal.explained_variance_ratio_)):.1f}%")

    # Set the appropriate loss
    # criterion = genWeightedMSELoss(n_components=n_components,
    #                                weights=weights,
    #                                device=device)
    # criterion = PCALoss(temp_pca=train_dataset.dataset, #best so far
    #                             sal_pca=train_dataset.dataset, 
    #                             n_components=n_components) 
    criterion = CombinedPCALoss(temp_pca=train_dataset.dataset, 
                                sal_pca=train_dataset.dataset, 
                                n_components=n_components, 
                                weights=weights, 
                                device=device)

    # Print parameters and dataset size
    true_params = [param for param, value in input_params.items() if value]

    def printParams():
        print(f"\nNumber of profiles: {len(full_dataset)}")
        print("Parameters used:", ", ".join(true_params))
        print(f"Min depth: {min_depth}, Max depth: {max_depth}")
        print(f"Number of components used: {n_components} x2")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Dropout probability: {dropout_prob}")
        print(f'Train/test/validation split: {train_size}/{test_size}/{val_size}')
        print(f"Layer configuration: {layers_config}\n")

    printParams()

    if load_trained_model:
        # Load model and PCA components
        model_path = '/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/model_Test Loss: 0.8945_2024-10-09 20:35:59_sat.pth'
        
        checkpoint = torch.load(model_path, map_location=torch.device(DEVICE))

        # Load model
        trained_model = PredictionModel(input_dim=input_dim, layers_config=layers_config, output_dim=n_components * 2, dropout_prob=dropout_prob)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        trained_model.to(DEVICE)
        
        # # Use multiple GPUs if available
        # if multi_gpu:
        #     model = nn.DataParallel(model)

        # Load PCA components
        pca_temp = checkpoint['pca_temp']
        pca_sal = checkpoint['pca_sal']
        input_params = checkpoint['input_params']
    else:
        for run in range(n_runs):
            print(f"Run {run + 1}/{n_runs}")

            # Create the model
            model = PredictionModel(input_dim=input_dim, layers_config=layers_config, output_dim=n_components * 2, dropout_prob=dropout_prob)
            model.to(DEVICE)

            # # Use multiple GPUs if available
            # if multi_gpu:
            #     model = nn.DataParallel(model)
                
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Start timer
            start_time = time.perf_counter()
            
            # Train the model
            trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience)

            end_time = time.perf_counter()
            # Calculate elapsed time
            elapsed_time = (end_time - start_time)
            print(f"NeSPReSO train: {elapsed_time:.2f} seconds.")
            
            # Test evaluation
            test_loss = evaluate_model(trained_model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f}")

            # Save the model and PCA components
            save_model_path = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/model_"
            save_model_path += f"Test Loss: {test_loss:.4f}" + "_"
            suffix = ".pth"
            if input_params['sat']:
                suffix = "_sat.pth"

            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            save_model_path += now_str + suffix

            pca_temp = full_dataset.pca_temp
            pca_sal = full_dataset.pca_sal
            
            # Save checkpoint with model state and PCA components
            checkpoint = {
                'model_state_dict': trained_model.state_dict(),
                'pca_temp': full_dataset.pca_temp,
                'pca_sal': full_dataset.pca_sal,
                'input_params': input_params
            }
            torch.save(checkpoint, save_model_path)
            print(f"Saved model and PCA components to {save_model_path}")
            
    print("Statistics from the last run:")
    print("Method & # profiles & Time (ms) & Time per profile (µs)")
    n_val = len(val_dataset)
    # Start time
    start_time = time.perf_counter()
    # Get predictions for the validation dataset
    for i in range(nn_repeat_time):
        val_predictions_pcs = get_predictions(trained_model, val_loader, device)
        # Accessing the original dataset for inverse_transform
        val_predictions = val_dataset.dataset.inverse_transform(val_predictions_pcs)
    # print(f"val_predictions_pcs: {type(val_predictions_pcs)} {val_predictions_pcs.shape}")
    # End time
    end_time = time.perf_counter()
    # Calculate elapsed time
    elapsed_time = (end_time - start_time)/nn_repeat_time
    print(f"NeSPReSO ({device}) & {n_val} & {elapsed_time*1e3:.2f} & {((elapsed_time*1e6)/n_val):.2f}")
    
    # repeat time calculation for other device, if available
     # Check if CUDA is available
    if torch.cuda.is_available():
        if next(trained_model.parameters()).is_cuda:
            cuda_device = torch.device("cpu")
        else:
            cuda_device = torch.device("cuda")
        
        # Move model to CUDA device
        trained_model = trained_model.to(cuda_device)
        
        # Create new DataLoader with CUDA device
        cuda_val_loader = DataLoader(val_dataset, batch_size=val_loader.batch_size, sampler=val_loader.sampler)
        
        # Start time
        start_time = time.perf_counter()
        
        # Get predictions for the validation dataset on CUDA
        for i in range(nn_repeat_time):
            cuda_val_predictions_pcs = get_predictions(trained_model, cuda_val_loader, cuda_device)
            # Accessing the original dataset for inverse_transform
            cuda_val_predictions = val_dataset.dataset.inverse_transform(cuda_val_predictions_pcs)
        
        # End time
        end_time = time.perf_counter()
        
        # Calculate elapsed time
        cuda_elapsed_time = (end_time - start_time) / nn_repeat_time
        print(f"NeSPReSO ({cuda_device}) & {n_val} & {cuda_elapsed_time*1e3:.2f} & {((cuda_elapsed_time*1e6)/n_val):.2f}")
        
        # Move model back to original device
        trained_model = trained_model.to(device)
    else:
        print("CUDA is not available. Skipping GPU time calculation.")
    
    # load ISOP results
    file_path_new = '/unity/g2/jmiranda/SubsurfaceFields/Data/ISOP1_rmse_bias_1deg_maps.nc'
    data_ISOP = xr.open_dataset(file_path_new)

    # Create bins for longitude and latitude
    lon_bins = np.arange(np.min(data_ISOP.lon) - 0.5, np.max(data_ISOP.lon) + 1.5, 1)
    lat_bins = np.arange(np.min(data_ISOP.lat) - 0.5, np.max(data_ISOP.lat) + 1.5, 1)

    # Calculate centers of the bins
    lon_centers = lon_bins + bin_size/2
    lat_centers = lat_bins + bin_size/2

    # Initialize a NaN array for the number of profiles
    num_prof = np.full((len(lat_centers), len(lon_centers)), np.nan)

    # Extracting RMSE data and ensuring it matches the dimensions of the bins
    avg_rmse_isop_t = data_ISOP['t_rmse_syn']
    avg_rmse_isop_s = data_ISOP['s_rmse_syn']
    
    avg_rmse_gdem_t = data_ISOP['t_rmse_gdem']
    avg_rmse_gdem_s = data_ISOP['s_rmse_gdem']
    
    avg_bias_isop_t = data_ISOP['t_bias_syn']
    avg_bias_isop_s = data_ISOP['s_bias_syn']
    
    subset_indices = val_loader.dataset.indices

    # For original profiles
    original_profiles = val_dataset.dataset.get_profiles(subset_indices, pca_approx=False)

    # For PCA approximated profiles
    pca_approx_profiles = val_dataset.dataset.get_profiles(subset_indices, pca_approx=True)
    
    if ensemble_models:
        # Directory where models are saved
        models_dir = '/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/'
        
        start_time = time.perf_counter()
        
        # Load all models
        models = load_all_models(
            models_dir=models_dir,
            device=device,
            input_dim=input_dim,
            layers_config=layers_config,
            n_components=n_components,
            dropout_prob=dropout_prob
        )
        # print(f"Loaded {len(models)} models.")
        
        # Initialize accumulators for predictions
        accumulated_pred_T = None
        accumulated_pred_S = None
        
        # Iterate over each model and accumulate predictions
        for model in models:
            # print(f"Generating predictions with model: {model}")
            # Get predictions for the validation dataset
            val_predictions_pcs = get_predictions(model, val_loader, device)
            # Inverse transform to get actual predictions
            val_predictions = val_dataset.dataset.inverse_transform(val_predictions_pcs)
            
            # Split predictions into T and S
            pred_T_current = val_predictions[0]  # Assuming index 0 for T
            pred_S_current = val_predictions[1]  # Assuming index 1 for S
            
            # Accumulate predictions
            if accumulated_pred_T is None:
                accumulated_pred_T = pred_T_current
                accumulated_pred_S = pred_S_current
            else:
                accumulated_pred_T += pred_T_current
                accumulated_pred_S += pred_S_current
        
        # Compute the average predictions
        avg_pred_T = accumulated_pred_T / len(models)
        avg_pred_S = accumulated_pred_S / len(models)
        
        # End time
        end_time = time.perf_counter()
        
        # Calculate elapsed time
        cuda_elapsed_time = (end_time - start_time)
        
        # Assign averaged predictions to final variables
        pred_T = avg_pred_T
        pred_S = avg_pred_S
        
        print(f"NeSPReSO ensamble 15x & {cuda_elapsed_time*1e3:.2f} & {((cuda_elapsed_time*1e6)/n_val):.2f}")
            
    else:
        # if not ensamble, get predictions from model
        pred_T = val_predictions[0]
        pred_S = val_predictions[1]
    
    orig_T = original_profiles[:, 0, :]
    orig_S = original_profiles[:, 1, :]
    
    # Start time
    start_time = time.perf_counter()
    # Get predictions for the validation dataset
    for i in range(gem_repeat_time):
        gem_temp, gem_sal = val_dataset.dataset.get_gem_profiles(subset_indices)
    # End time
    end_time = time.perf_counter()
    # Calculate elapsed time
    elapsed_time = (end_time - start_time)/gem_repeat_time
    print(f"GEM (cpu) & {n_val} & {elapsed_time*1e3:.2f} & {((elapsed_time*1e6)/n_val):.2f}")
    # print(f"GEM predictions {n_val} - Elapsed time: {elapsed_time:.6f} seconds, ran {gem_repeat_time} times.")
    gems_T = gem_temp.T
    gems_S = gem_sal.T
    
    pred_T_resid = pred_T - orig_T
    pred_S_resid = pred_S - orig_S
    gems_T_resid = gems_T - orig_T
    gems_S_resid = gems_S - orig_S
    
    sst_inputs, ssh_inputs = val_dataset.dataset.get_inputs(subset_indices)
    
    gem_temp, gem_sal = val_dataset.dataset.get_gem_profiles(subset_indices)
    
    lat_val, lon_val, dates_val = val_dataset.dataset.get_lat_lon_date(subset_indices)
    lat_val = np.floor(lat_val)+bin_size/2
    lon_val = np.floor(lon_val)+bin_size/2
    
    # visualize_combined_results(pca_approx_profiles, gem_temp, gem_sal, val_predictions, sst_inputs, ssh_inputs, min_depth=min_depth, max_depth = max_depth, num_samples=num_samples)
        
    printParams()
    
    print("Let's investigate how the method compares against vanilla GEM with in-situ SSH")
    
    gem_temp_se = gems_T_resid**2
    gem_sal_se = gems_S_resid**2

    nn_temp_se = pred_T_resid**2
    nn_sal_se = pred_S_resid**2
    
    ist = xr.open_dataset('/unity/g2/jmiranda/SubsurfaceFields/Data/isop1_stats_temp.nc')
    iss = xr.open_dataset('/unity/g2/jmiranda/SubsurfaceFields/Data/isop1_stats_salt.nc')

    our_depths = np.arange(0,1801)
    isop_depths = ist.depth.values
    avg_gem_temp_rmse = np.sqrt(np.mean(gem_temp_se, axis = 1))
    avg_nn_temp_rmse = np.sqrt(np.mean(nn_temp_se, axis = 1))

    avg_gem_sal_rmse = np.sqrt(np.mean(gem_sal_se, axis = 1))
    avg_nn_sal_rmse = np.sqrt(np.mean(nn_sal_se, axis = 1))

    avg_gem_temp_bias = np.mean(gems_T_resid, axis = 1)
    avg_nn_temp_bias = np.mean(pred_T_resid, axis = 1)

    avg_gem_sal_bias = np.mean(gems_S_resid, axis = 1)
    avg_nn_sal_bias = np.mean(pred_S_resid, axis = 1)

    # Identify indices for the training, validation, and testing datasets
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    
    ## Multiple linear regression
    

    def prepare_features(inputs_array, max_degree=3):
        """
        Prepare the feature matrix for regression by including polynomial terms.

        Args:
        - inputs_array (numpy.ndarray): Array of input features, shape (n_samples, n_features).
        - max_degree (int): Maximum degree of polynomial features.

        Returns:
        - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features_expanded).
        """
        # Generate polynomial features up to the specified degree
        poly = PolynomialFeatures(degree=max_degree, include_bias=False)
        X = poly.fit_transform(inputs_array)
        return X

    def fit_pcs_regression_exact_gpu(X, pcs):
        """
        Fit regression models to predict principal component scores from features using exact least squares on GPU.

        Args:
        - X (numpy.ndarray): Feature matrix, shape (n_samples, n_features_expanded).
        - pcs (numpy.ndarray): Principal component scores, shape (n_samples, n_components).

        Returns:
        - beta (torch.Tensor): Coefficient matrix, shape (n_features_expanded, n_components).
        """
        # Convert data to torch tensors and move to GPU
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        pcs_tensor = torch.tensor(pcs, dtype=torch.float32).to(DEVICE)

        # Compute the pseudoinverse of X
        # Note: For large matrices, torch.linalg.lstsq may be more efficient
        X_pinv = torch.pinverse(X_tensor)
        
        print(f"{X_pinv.shape=}")

        # Compute the coefficients (beta) analytically
        beta = X_pinv @ pcs_tensor

        return beta

    def predict_pcs_exact_gpu(beta, X_new):
        """
        Predict principal component scores using the exact coefficients on GPU.

        Args:
        - beta (torch.Tensor): Coefficient matrix, shape (n_features_expanded, n_components).
        - X_new (numpy.ndarray): New feature matrix, shape (n_samples_new, n_features_expanded).

        Returns:
        - pcs_pred (numpy.ndarray): Predicted principal component scores, shape (n_samples_new, n_components).
        """
        with torch.no_grad():
            X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(DEVICE)
            pcs_pred_tensor = X_new_tensor @ beta
            pcs_pred = pcs_pred_tensor.cpu().numpy()
        return pcs_pred

    def inverse_transform(pcs, pca_temp, pca_sal, n_components):
        """
        Inverse the PCA transformation to reconstruct temperature and salinity profiles.

        Args:
        - pcs (numpy.ndarray): Concatenated PCA components for temperature and salinity.
        - pca_temp, pca_sal: PCA models for temperature and salinity respectively.
        - n_components (int): Number of PCA components for each.

        Returns:
        - temp_profiles (numpy.ndarray): Reconstructed temperature profiles.
        - sal_profiles (numpy.ndarray): Reconstructed salinity profiles.
        """
        temp_profiles = pca_temp.inverse_transform(pcs[:, :n_components]).T
        sal_profiles = pca_sal.inverse_transform(pcs[:, n_components:]).T
        return temp_profiles, sal_profiles

    # define polynomial degree
    MLR_degree = 1
    # Collect inputs and PCS from the training dataset
    MLR_indices = train_indices + val_indices
    inputs_list_train, pcs_list_train = full_dataset.__getitem__(MLR_indices)

    # Convert lists to numpy arrays
    inputs_array_train = np.array(inputs_list_train.T)  # Shape: (n_samples_train, n_features)
    pcs_array_train = np.array(pcs_list_train)
    pcs_T_train, pcs_S_train = np.hsplit(pcs_list_train, 2)
    pcs_T_train = pcs_T_train.T        # Shape: (n_samples_train, n_components)
    pcs_S_train = pcs_S_train.T        # Shape: (n_samples_train, n_components)
    
    # Prepare features for training
    X_train = prepare_features(inputs_array_train, max_degree=MLR_degree)
    # Initialize the StandardScaler
    scaler = StandardScaler()
    X_avgs = X_train.mean(axis=0)
    # Fit the scaler on the training data
    X_train_scaled = scaler.fit_transform(X_train) + 1

    # Fit regression model on GPU
    print("Beginning multiple linear regression using PyTorch on GPU")
    start_time = time.perf_counter()
    beta_T = fit_pcs_regression_exact_gpu(X_train, pcs_T_train)
    beta_S = fit_pcs_regression_exact_gpu(X_train, pcs_S_train)
    
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time)
    print(f"MLR fit completed in {elapsed_time:.2f} seconds.")

    # Prepare inputs for the validation dataset
    inputs_list_val, _ = full_dataset.__getitem__(val_indices)
    inputs_array_val = np.array(inputs_list_val.T)  # Shape: (n_samples_val, n_features)
    # Prepare features for validation
    X_val = prepare_features(inputs_array_val, max_degree=MLR_degree)
    X_val_scaled = scaler.fit_transform(X_val) + 1
    
    # Predict the principal component scores for the validation data
    pcs_pred_val_T = predict_pcs_exact_gpu(beta_T, X_val)
    pcs_pred_val_S = predict_pcs_exact_gpu(beta_S, X_val)
    pcs_pred_val = np.hstack([pcs_pred_val_T, pcs_pred_val_S])

    # Reconstruct the temperature and salinity profiles using the inverse PCA transformation
    temp_MLR_profiles, sal_MLR_profiles = inverse_transform(pcs_pred_val, pca_temp, pca_sal, n_components)
    
    # Extract the original temperature and salinity profiles
    original_temp_profiles = original_profiles[:, 0, :]  # Shape: (n_samples, depth_levels)
    original_sal_profiles = original_profiles[:, 1, :]   # Shape: (n_samples, depth_levels)

    # Calculate residuals
    mlr_T_resid = temp_MLR_profiles - original_temp_profiles  # Shape: (n_samples, depth_levels)
    mlr_S_resid = sal_MLR_profiles - original_sal_profiles    # Shape: (n_samples, depth_levels)

    # Compute squared errors
    mlr_temp_se = mlr_T_resid**2  # Shape: (depth_levels, n_samples)
    mlr_sal_se = mlr_S_resid**2   # Shape: (depth_levels, n_samples)

    # Compute average RMSE
    avg_mlr_temp_rmse = np.sqrt(np.mean(mlr_temp_se, axis=1))  # Shape: (depth_levels,)
    avg_mlr_sal_rmse = np.sqrt(np.mean(mlr_sal_se, axis=1))    # Shape: (depth_levels,)

    # Compute average bias
    avg_mlr_temp_bias = np.mean(mlr_T_resid, axis=1)  # Shape: (depth_levels,)
    avg_mlr_sal_bias = np.mean(mlr_S_resid, axis=1)   # Shape: (depth_levels,)

    fig = plt.figure(figsize=(18,18))

    # Temperature RMSE Plot
    ax = fig.add_subplot(2,2,1)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(ist.rmse.values, ist.depth.values, linewidth=3, label='ISOP', color='xkcd:blue')
    plt.plot(avg_gem_temp_rmse, np.arange(0,1801), linewidth=3, label='GEM', color='xkcd:orange')
    plt.plot(avg_mlr_temp_rmse, np.arange(0,1801), linewidth=3, label='MLR', color='xkcd:green')
    plt.plot(avg_nn_temp_rmse, np.arange(0,1801), linewidth=3, label='NeSPReSO', color='xkcd:gray')
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature RMSE [°C]")
    plt.ylabel("Depth [m]")
    plt.title("Average Temperature RMSE")

    # Salinity RMSE Plot
    ax = fig.add_subplot(2,2,2)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(iss.rmse.values, iss.depth.values, linewidth=3, label='ISOP', color='xkcd:blue')
    plt.plot(avg_gem_sal_rmse, np.arange(0,1801), linewidth=3, label='GEM', color='xkcd:orange')
    plt.plot(avg_mlr_sal_rmse, np.arange(0,1801), linewidth=3, label='MLR', color='xkcd:green')
    plt.plot(avg_nn_sal_rmse, np.arange(0,1801), linewidth=3, label='NeSPReSO', color='xkcd:gray')
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Salinity RMSE [PSU]")
    plt.title("Average Salinity RMSE")

    # Temperature Bias Plot
    ax = fig.add_subplot(2,2,3)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(ist.bias.values, ist.depth.values, linewidth=3, label='ISOP', color='xkcd:blue')
    plt.plot(avg_gem_temp_bias, np.arange(0,1801), linewidth=3, label='GEM', color='xkcd:orange')
    plt.plot(avg_mlr_temp_bias, np.arange(0,1801), linewidth=3, label='MLR', color='xkcd:green')
    plt.plot(avg_nn_temp_bias, np.arange(0,1801), linewidth=3, label='NeSPReSO', color='xkcd:gray')
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Temperature Bias [°C]")
    plt.ylabel("Depth [m]")
    plt.title("Average Temperature Bias")

    # Salinity Bias Plot
    ax = fig.add_subplot(2,2,4)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.plot(iss.bias.values, iss.depth.values, linewidth=3, label='ISOP', color='xkcd:blue')
    plt.plot(avg_gem_sal_bias, np.arange(0,1801), linewidth=3, label='GEM', color='xkcd:orange')
    plt.plot(avg_mlr_sal_bias, np.arange(0,1801), linewidth=3, label='MLR', color='xkcd:green')
    plt.plot(avg_nn_sal_bias, np.arange(0,1801), linewidth=3, label='NeSPReSO', color='xkcd:gray')
    ax.invert_yaxis()
    plt.legend()
    plt.xlabel("Salinity Bias [PSU]")
    plt.title("Average Salinity Bias")

    plt.tight_layout()
    plt.show()
    
    # visualize features    
    feature_names = [
        "timecos", "timesin",
        "latcos", "latsin",
        "loncos", "lonsin",
        "sst", "sss", "ssh"
    ]

    def plot_coefficients_heatmap(beta, feature_names, title, normalize=True, threshold=1e-4):
        """
        Plots a heatmap of regression coefficients with optional per-PC normalization and thresholding.

        Args:
        - beta (torch.Tensor): Coefficient matrix, shape (n_features, n_components).
        - feature_names (list): List of feature names corresponding to the rows of beta.
        - title (str): Title of the heatmap.
        - normalize (bool): Whether to apply per-PC Max-Abs normalization. Default is True.
        - threshold (float): Threshold below which coefficients are set to zero. Default is 1e-4.
        """
        # Convert to NumPy
        beta_np = beta.cpu().numpy()

        if normalize:
            # Apply Max-Abs normalization per PC (column-wise)
            max_abs_per_pc = np.max(np.abs(beta_np), axis=0)  # Shape: (n_components,)
            
            # Handle cases where the maximum is zero to avoid division by zero
            max_abs_per_pc[max_abs_per_pc == 0] = 1
            
            # Normalize each column (PC) by its maximum absolute value
            beta_np = beta_np / max_abs_per_pc  # Broadcasting division

        # Thresholding: Set coefficients with abs < threshold to zero
        beta_np_thresholded = np.where(np.abs(beta_np) < threshold, 0, beta_np)

        # Create a DataFrame for seaborn
        df_beta = pd.DataFrame(
            beta_np_thresholded,
            index=feature_names,
            columns=[f'PC{i+1}' for i in range(beta_np_thresholded.shape[1])]
        )

        plt.figure(figsize=(20, 10))
        sns.heatmap(
            df_beta,
            cmap='coolwarm',
            center=0,
            annot=False,
            fmt=".2f",
            vmin=-1,  # Since normalization scales coefficients between -1 and 1
            vmax=1,
            linewidths=0.5,
            linecolor='gray'
        )
        plt.title(title, fontsize=16)
        plt.xlabel('Principal Components', fontsize=14)
        plt.ylabel('Input Features', fontsize=14)
        plt.tight_layout()
        plt.show()

    beta_T_scaled = beta_T.cpu() / X_avgs[:,None]
    beta_S_scaled = beta_S.cpu() / X_avgs[:,None]
    beta_T_dropped = torch.cat((beta_T_scaled[:2], beta_T_scaled[6:]), dim=0)
    beta_S_dropped = torch.cat((beta_S_scaled[:2], beta_S_scaled[6:]), dim=0)
    feature_names_dropped = feature_names[:2] + feature_names[6:]
    
    # # Plot Heatmap for Temperature PCs with Normalization
    # plot_coefficients_heatmap(
    #     beta_T_dropped,
    #     feature_names_dropped,
    #     "Normalized Regression Coefficients for Temperature PCs",
    #     normalize=True
    # )

    # # Plot Heatmap for Salinity PCs with Normalization
    # plot_coefficients_heatmap(
    #     beta_S_dropped,
    #     feature_names_dropped,
    #     "Normalized Regression Coefficients for Salinity PCs",
    #     normalize=True
    # )

    # Heatmaps of the RMSE
    # dpt_range = np.arange(0,201)
    upper_limit = 0
    lower_limit = 1800
    dpt_range = isop_depths[(isop_depths <= lower_limit) & (isop_depths >= upper_limit)].astype(int)

    print(f"Statistics/Plots for Depth range: [{lower_limit}, {upper_limit}]")
    
    # Calculate average temperature RMSE for NN and GEM !
    grid_avg_temp_rmse_nn, num_prof_nn = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, pred_T_resid, dpt_range, is_rmse=True)  
    grid_avg_temp_rmse_gem, num_prof_gem = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, gems_T_resid, dpt_range, is_rmse=True)  
    grid_avg_temp_rmse_gain = grid_avg_temp_rmse_nn - grid_avg_temp_rmse_gem
    # same for MLR
    grid_avg_temp_rmse_mlr, num_prof_mlr = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, mlr_T_resid, dpt_range, is_rmse=True)
    grid_avg_temp_rmse_gain_mlr = grid_avg_temp_rmse_nn - grid_avg_temp_rmse_mlr
    
    plot_bin_map(lon_bins, lat_bins, grid_avg_temp_rmse_nn, num_prof_nn, "Temperature", "RMSE")
    
    #now let's do the same for salinity
    # Calculate average temperature RMSE for NN and GEM
    grid_avg_sal_rmse_nn, num_prof_nn = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, pred_S_resid, dpt_range, is_rmse=True)
    grid_avg_sal_rmse_gem, num_prof_gem = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, gems_S_resid, dpt_range, is_rmse=True)
    grid_avg_sal_rmse_gain = grid_avg_sal_rmse_nn - grid_avg_sal_rmse_gem
    # same for MLR
    grid_avg_sal_rmse_mlr, num_prof_mlr = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, mlr_S_resid, dpt_range, is_rmse=True)
    grid_avg_sal_rmse_gain_mlr = grid_avg_sal_rmse_nn - grid_avg_sal_rmse_mlr
    
    plot_bin_map(lon_bins, lat_bins, grid_avg_sal_rmse_nn, num_prof_nn, "Salinity", "RMSE")
    
    # same maps, but bias    
    
    avg_nn_t_bias, num_prof_nn = calculate_average_in_bin(  lon_centers, lat_centers, lon_val, lat_val, pred_T_resid, dpt_range, is_rmse=False)
    avg_nn_s_bias, num_prof_nn = calculate_average_in_bin(  lon_centers, lat_centers, lon_val, lat_val, pred_S_resid, dpt_range, is_rmse=False)
    avg_gem_t_bias, num_prof_gem = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, gems_T_resid, dpt_range, is_rmse=False)
    avg_gem_s_bias, num_prof_gem = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, gems_S_resid, dpt_range, is_rmse=False)
    # same for MLR
    avg_mlr_t_bias, num_prof_mlr = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, mlr_T_resid, dpt_range, is_rmse=False)
    avg_mlr_s_bias, num_prof_mlr = calculate_average_in_bin(lon_centers, lat_centers, lon_val, lat_val, mlr_S_resid, dpt_range, is_rmse=False)
    
    #TODO: fix the bias color scale (negative values are not being shown properly)
    plot_bin_map(lon_bins, lat_bins, avg_nn_t_bias, num_prof_nn, "Temperature", "Bias")
    plot_bin_map(lon_bins, lat_bins, avg_nn_s_bias, num_prof_nn, "Salinity", "Bias")
    
    # now let's redo these maps, but for the different seasons (months spring: MAM, summer: JJA, fall: SON, winter: DJF)
    # Convert matlab dates to Python datetime objects
    python_dates = [datetime.fromordinal(int(d)) + timedelta(days=d%1) - timedelta(days=366) for d in dates_val]

    # Define seasons
    def get_season(date):
        month = date.month
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Winter'

    seasons = [get_season(date) for date in python_dates]
    
    # Calculate and plot statistics for each season
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    
    # Initialize lists to store data for all seasons
    nn_temp_rmse_all = []
    nn_sal_rmse_all = []
    nn_temp_bias_all = []
    nn_sal_bias_all = []
    gem_temp_rmse_all = []
    gem_sal_rmse_all = []
    gem_temp_bias_all = []
    gem_sal_bias_all = []
    mlr_temp_rmse_all = []
    mlr_sal_rmse_all = []
    mlr_temp_bias_all = []
    mlr_sal_bias_all = []

    for season in seasons:
        season_mask = np.array([get_season(date) for date in python_dates]) == season
        
        # Calculate RMSE and bias by depth for each season for both NN and GEM
        nn_temp_rmse = np.sqrt(np.nanmean((pred_T_resid[:,season_mask])**2, axis=1))
        nn_sal_rmse = np.sqrt(np.nanmean((pred_S_resid[:,season_mask])**2, axis=1))
        nn_temp_bias = np.nanmean(pred_T_resid[:,season_mask], axis=1)
        nn_sal_bias = np.nanmean(pred_S_resid[:,season_mask], axis=1)
        gem_temp_rmse = np.sqrt(np.nanmean((gems_T_resid[:,season_mask])**2, axis=1))
        gem_sal_rmse = np.sqrt(np.nanmean((gems_S_resid[:,season_mask])**2, axis=1))
        gem_temp_bias = np.nanmean(gems_T_resid[:,season_mask], axis=1)
        gem_sal_bias = np.nanmean(gems_S_resid[:,season_mask], axis=1)
        mlr_temp_rmse = np.sqrt(np.nanmean((mlr_T_resid[:,season_mask])**2, axis=1))
        mlr_sal_rmse = np.sqrt(np.nanmean((mlr_S_resid[:,season_mask])**2, axis=1))
        mlr_temp_bias = np.nanmean(mlr_T_resid[:,season_mask], axis=1)
        mlr_sal_bias = np.nanmean(mlr_S_resid[:,season_mask], axis=1)
        
        # Append data to lists
        nn_temp_rmse_all.append(nn_temp_rmse)
        nn_sal_rmse_all.append(nn_sal_rmse)
        nn_temp_bias_all.append(nn_temp_bias)
        nn_sal_bias_all.append(nn_sal_bias)
        gem_temp_rmse_all.append(gem_temp_rmse)
        gem_sal_rmse_all.append(gem_sal_rmse)
        gem_temp_bias_all.append(gem_temp_bias)
        gem_sal_bias_all.append(gem_sal_bias)
        mlr_temp_rmse_all.append(mlr_temp_rmse)
        mlr_sal_rmse_all.append(mlr_sal_rmse)
        mlr_temp_bias_all.append(mlr_temp_bias)
        mlr_sal_bias_all.append(mlr_sal_bias)
        
    # Create the figures
    fig_rmse, axs_rmse = plt.subplots(4, 2, figsize=(20, 30))
    fig_bias, axs_bias = plt.subplots(4, 2, figsize=(20, 30))

    fig_rmse.suptitle("RMSE by Depth for Each Season", fontsize=20)
    fig_bias.suptitle("Bias by Depth for Each Season", fontsize=20)

    # Find max values for consistent x-axis scales
    max_temp_rmse = max(np.max(nn_temp_rmse_all), np.max(gem_temp_rmse_all))
    max_sal_rmse = max(np.max(nn_sal_rmse_all), np.max(gem_sal_rmse_all))
    max_temp_bias = max(np.max(np.abs(nn_temp_bias_all)), np.max(np.abs(gem_temp_bias_all)))
    max_sal_bias = max(np.max(np.abs(nn_sal_bias_all)), np.max(np.abs(gem_sal_bias_all)))

    for i, season in enumerate(seasons):
        # RMSE plots
        axs_rmse[i, 0].plot(gem_temp_rmse_all[i], np.arange(0,1801), linewidth=3, label='GEM', color='xkcd:orange')
        axs_rmse[i, 0].plot(mlr_temp_rmse_all[i], np.arange(0,1801), linewidth=3, label='MLR', color='xkcd:green')
        axs_rmse[i, 0].plot(nn_temp_rmse_all[i], np.arange(0,1801), linewidth=3, label='NeSPReSO', color='xkcd:gray')
        axs_rmse[i, 0].invert_yaxis()
        if i==3:
            axs_rmse[i, 0].set_xlabel("Temperature RMSE [°C]")
        axs_rmse[i, 0].set_ylabel("Depth [m]")
        axs_rmse[i, 0].set_title(f"{season} - Temperature RMSE")
        axs_rmse[i, 0].legend()
        axs_rmse[i, 0].grid(True)
        axs_rmse[i, 0].set_xlim(0, max_temp_rmse)

        axs_rmse[i, 1].plot(gem_sal_rmse_all[i], np.arange(0,1801), linewidth=3, label='GEM', color='xkcd:orange')
        axs_rmse[i, 1].plot(mlr_sal_rmse_all[i], np.arange(0,1801), linewidth=3, label='MLR', color='xkcd:green')
        axs_rmse[i, 1].plot(nn_sal_rmse_all[i], np.arange(0,1801), linewidth=3, label='NeSPReSO', color='xkcd:gray')
        axs_rmse[i, 1].invert_yaxis()
        if i==3:
            axs_rmse[i, 1].set_xlabel("Salinity RMSE [PSU]")
        axs_rmse[i, 1].set_title(f"{season} - Salinity RMSE")
        axs_rmse[i, 1].legend()
        axs_rmse[i, 1].grid(True)
        axs_rmse[i, 1].set_xlim(0, max_sal_rmse)

        # Bias plots
        axs_bias[i, 0].plot(gem_temp_bias_all[i], np.arange(0,1801), linewidth=3, label='GEM', color='xkcd:orange')
        axs_bias[i, 0].plot(mlr_temp_bias_all[i], np.arange(0,1801), linewidth=3, label='MLR', color='xkcd:green')
        axs_bias[i, 0].plot(nn_temp_bias_all[i], np.arange(0,1801), linewidth=3, label='NeSPReSO', color='xkcd:gray')
        axs_bias[i, 0].invert_yaxis()
        if i==3:
            axs_bias[i, 0].set_xlabel("Temperature Bias [°C]")
        axs_bias[i, 0].set_ylabel("Depth [m]")
        axs_bias[i, 0].set_title(f"{season} - Temperature Bias")
        axs_bias[i, 0].legend()
        axs_bias[i, 0].grid(True)
        axs_bias[i, 0].set_xlim(-max_temp_bias, max_temp_bias)

        axs_bias[i, 1].plot(gem_sal_bias_all[i], np.arange(0,1801), linewidth=3, label='GEM', color='xkcd:orange')
        axs_bias[i, 1].plot(mlr_sal_bias_all[i], np.arange(0,1801), linewidth=3, label='MLR', color='xkcd:green')
        axs_bias[i, 1].plot(nn_sal_bias_all[i], np.arange(0,1801), linewidth=3, label='NeSPReSO', color='xkcd:gray')
        axs_bias[i, 1].invert_yaxis()
        if i==3:
            axs_bias[i, 1].set_xlabel("Salinity Bias [PSU]")
        axs_bias[i, 1].set_title(f"{season} - Salinity Bias")
        axs_bias[i, 1].legend()
        axs_bias[i, 1].grid(True)
        axs_bias[i, 1].set_xlim(-max_sal_bias, max_sal_bias)

    plt.tight_layout()
    plt.show()
        # # Temperature RMSE
        # grid_avg_temp_rmse_nn_season, num_prof_nn_season = calculate_average_in_bin(
        #     lon_centers, lat_centers, lon_val[season_mask], lat_val[season_mask], 
        #     pred_T_resid[:,season_mask], dpt_range, is_rmse=True
        # )
        
        
        # plot_bin_map(lon_bins, lat_bins, grid_avg_temp_rmse_nn_season, num_prof_nn_season, 
        #              f"Temperature - {season}", "RMSE")
        
        # # Salinity RMSE
        # grid_avg_sal_rmse_nn_season, _ = calculate_average_in_bin(
        #     lon_centers, lat_centers, lon_val[season_mask], lat_val[season_mask], 
        #     pred_S_resid[:,season_mask], dpt_range, is_rmse=True
        # )
        # plot_bin_map(lon_bins, lat_bins, grid_avg_sal_rmse_nn_season, num_prof_nn_season, 
        #              f"Salinity - {season}", "RMSE")
        
        # # Temperature Bias
        # avg_nn_t_bias_season, _ = calculate_average_in_bin(
        #     lon_centers, lat_centers, lon_val[season_mask], lat_val[season_mask], 
        #     pred_T_resid[:,season_mask], dpt_range, is_rmse=False
        # )
        # plot_bin_map(lon_bins, lat_bins, avg_nn_t_bias_season, num_prof_nn_season, 
        #              f"Temperature - {season}", "Bias")
        
        # # Salinity Bias
        # avg_nn_s_bias_season, _ = calculate_average_in_bin(
        #     lon_centers, lat_centers, lon_val[season_mask], lat_val[season_mask], 
        #     pred_S_resid[:,season_mask], dpt_range, is_rmse=False
        # )
        # plot_bin_map(lon_bins, lat_bins, avg_nn_s_bias_season, num_prof_nn_season, 
        #              f"Salinity - {season}", "Bias")

    # Comparison maps    
    avg_rmse_nn_t  = grid_avg_temp_rmse_nn
    avg_rmse_nn_s  = grid_avg_sal_rmse_nn
    avg_bias_nn_t  = avg_nn_t_bias
    avg_bias_nn_s  = avg_nn_s_bias
    avg_rmse_gem_t = grid_avg_temp_rmse_gem
    avg_rmse_gem_s = grid_avg_sal_rmse_gem
    avg_bias_gem_t = avg_gem_t_bias
    avg_bias_gem_s = avg_gem_s_bias
    avg_rmse_mlr_t = grid_avg_temp_rmse_mlr
    avg_rmse_mlr_s = grid_avg_sal_rmse_mlr
    avg_bias_mlr_t = avg_mlr_t_bias
    avg_bias_mlr_s = avg_mlr_s_bias
    
    lon_centr = lon_centers[:-1]
    lat_centr = lat_centers[:-1]
        
    #GEM
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_t, avg_rmse_gem_t, "temperature", "GEM")
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_s, avg_rmse_gem_s, "salinity", "GEM")
    
    #ISOP
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_t, avg_rmse_isop_t, "temperature", "ISOP")
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_s, avg_rmse_isop_s, "salinity", "ISOP")
    
    #MLR
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_t, avg_rmse_mlr_t, "temperature", "MLR")
    plot_comparison_maps(lon_centr, lat_centr, avg_rmse_nn_s, avg_rmse_mlr_s, "salinity", "MLR")
    
    print("the following are bias plots")
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_t, avg_bias_isop_t, "temperature", "ISOP", "Bias")
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_s, avg_bias_isop_s, "salinity", "ISOP", "Bias")
    
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_t, avg_bias_gem_t, "temperature", "GEM", "Bias")
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_s, avg_bias_gem_s, "salinity", "GEM", "Bias")
    
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_t, avg_bias_mlr_t, "temperature", "MLR", "Bias")
    plot_comparison_maps(lon_centr, lat_centr, avg_bias_nn_s, avg_bias_mlr_s, "salinity", "MLR", "Bias")
    
    # Residual calculations
    nn_temp_residuals = pred_T - original_profiles[:, 0, :]
    nn_sal_residuals = pred_S - original_profiles[:, 1, :]
    
    ## GLIDER: Load the MATLAB file
    file_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/Glider_binned_data_for_heat_content_IA_mission_lowpass_LCE_Campeche_cyclone.mat'
    gl_data = scipy.io.loadmat(file_path)

    # Display the keys to understand the structure of the data
    print(gl_data.keys())

    # Variable	Shape   	nan_min	    nan_max	    nan_avg	    nan_count
    # T1	    (201,240)	5.043881	40.509893	12.093195	387
    # S1	    (201,240)	32.740974	36.657226	35.476389	387
    # lon1	    (1,	240)	-94.817717	-94.405167	-94.690244	0
    # lat1	    (1,	240)	24.02255	27.0873	    25.840945	0
    # t1	    (1,	240)	736863.0151	736883.0033	736873.3337	0
    # T2	    (201,238)	4.708326	32.524226	12.129467	426
    # S2	    (201,238)	30.45545	36.783535	35.478705	426
    # lon2	    (1,	238)	-95.757333	-93.456633	-94.694365	0
    # lat2	    (1,	238)	22.9795	    26.09425	24.404746	0
    # t2	    (1,	238)	736892.9703	736914.0272	736902.8855	0
    # T3	    (201,223)	4.648858	27.38052	10.070962	271
    # S3	    (201,223)	34.902126	36.851436	35.290017	271
    # lon3	    (1,	223)	-95.99222	-93.557308	-94.752642	0
    # lat3	    (1,	223)	19.702925	21.887858	20.131646	0
    # t3	    (1,	223)	737142.9568	737180.9581	737161.8109	0
    # T4	    (201,301)	5.099238	31.526334	13.782459	4914
    # S4	    (201,301)	34.867155	37.01149	35.656481	4914
    # lon4	    (1,	301)	-89.085857	-87.716895	-88.320707	0
    # lat4	    (1,	301)	24.371618	26.146342	25.573801	0
    # t4	    (1,	301)	737254.5333	737272.0137	737263.1334	0
    # T1l	    (201,240)	4.971843	29.91509	12.041973	0
    # S1l	    (201,240)	32.886428	41.790276	35.479336	0
    # T2l	    (201,238)	4.70553	    30.756818	12.094563	0
    # S2l	    (201,238)	29.426921	36.761195	35.464176	0
    # T3l	    (201,223)	4.753081	27.064557	10.053855	223
    # S3l	    (201,223)	34.903059	36.599977	35.288714	223
    # T4l	    (201,301)	5.125582	31.045829	13.877145	4914
    # S4l	    (201,301)	34.92949	37.019112	35.665196	4914
    # t1l	    (1,	240)	736863.0151	736883.0033	736873.0092	0
    # t2l	    (1,	238)	736892.9703	736914.0272	736903.4988	0
    # t3l	    (1,	223)	737142.9568	737180.9581	737161.9574	0
    # t4l	    (1,	301)	737254.5333	737272.0137	737263.2735	0


    def haversine(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def calculate_distances(latitudes, longitudes):
        """ Calculate the cumulative distance between successive lat/long pairs. """
        n = len(latitudes)
        distances = np.zeros(n)
        for i in range(1, n):
            distances[i] = distances[i-1] + haversine(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
        return distances

    def datenums_to_datetimes(matlab_datenums):
        """
        Convert an array of MATLAB datenum values to Python datetime objects.

        Parameters:
        matlab_datenums (np.array): Array of MATLAB datenum values

        Returns:
        list: List of Python datetime objects
        """
        python_datetimes = [datenum_to_datetime(datenum) for datenum in matlab_datenums]
        return python_datetimes

    def plot_field(data, distances, depths, variable_name, title):
        """
        Plot the temperature or salinity field over distance.
        
        Parameters:
        data (np.array): 2D array of temperature or salinity data
        distances (np.array): 1D array of distances corresponding to the data
        variable_name (str): Name of the variable ('Temperature' or 'Salinity')
        title (str): Title of the plot
        """
        if variable_name == 'Temperature':
            vmin = 0
            vmax = 40
            step = 5
            cmap = ccm.thermal
        elif variable_name == 'Salinity':
            vmin = 34
            vmax = 37
            step = 1
            cmap = ccm.haline
        elif variable_name == "T Differences":
            vmin = -4
            vmax = 4
            step = 0.2
            cmap = 'coolwarm'
        elif variable_name == "S Differences":
            vmin = -1
            vmax = 1
            step = 0.1
            cmap = 'PiYG'
        else:
            raise ValueError(f"Invalid variable name: {variable_name}")
        
        num_levels = int((vmax - vmin) / step + 1)
        cmap = plt.get_cmap(cmap, num_levels)
        
        plt.figure(figsize=(12, 6))
        plt.contour(distances, depths, data, levels=np.arange(vmin, np.ceil(vmax)+1, step), colors='black', linewidths=0.1)
        plt.pcolormesh(distances, depths, data, shading='nearest', cmap=cmap, vmin = vmin, vmax = vmax)
        plt.colorbar(label=f'{variable_name} [{variable_name[0]}]', extend='both')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (index)')
        plt.title(title)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.show()

    # Extract locations and distance
    latitudes_T1 = gl_data['lat1'][0]
    longitudes_T1 = gl_data['lon1'][0]
    latitudes_T2 = gl_data['lat2'][0]
    longitudes_T2 = gl_data['lon2'][0]
    latitudes_T3 = gl_data['lat3'][0]
    longitudes_T3 = gl_data['lon3'][0]
    latitudes_T4 = gl_data['lat4'][0]
    longitudes_T4 = gl_data['lon4'][0]
    d1 = calculate_distances(latitudes_T1, longitudes_T1)
    d2 = calculate_distances(latitudes_T2, longitudes_T2)
    d3 = calculate_distances(latitudes_T3, longitudes_T3)
    d4 = calculate_distances(latitudes_T4, longitudes_T4)

    #times
    tt1 = gl_data['t1'][0]
    tt2 = gl_data['t2'][0]
    tt3 = gl_data['t3'][0]
    tt4 = gl_data['t4'][0]
    t1 = datenums_to_datetimes(tt1)
    t2 = datenums_to_datetimes(tt2)
    t3 = datenums_to_datetimes(tt3)
    t4 = datenums_to_datetimes(tt4)

    # Extract temperature and salinity
    T1 = gl_data['T1']
    S1 = gl_data['S1']
    T2 = gl_data['T2']
    S2 = gl_data['S2']
    T3 = gl_data['T3']
    S3 = gl_data['S3']
    T4 = gl_data['T4']
    S4 = gl_data['S4']
    
    if 'sss1' in data:
        # with open(dataset_pickle_file, 'rb') as file:
        #     data = pickle.load(file)
        
        sss1 = data['sss1']
        sss2 = data['sss2']
        sss3 = data['sss3']
        sss4 = data['sss4']
        sst1 = data['sst1']
        sst2 = data['sst2']
        sst3 = data['sst3']
        sst4 = data['sst4']
        aviso1 = data['aviso1']
        aviso2 = data['aviso2']
        aviso3 = data['aviso3']
        aviso4 = data['aviso4']
        
    else:    
        # Extract aviso, sst, sss
        sss1, sst1, aviso1 = load_satellite_data(t1, latitudes_T1, longitudes_T1)
        sss2, sst2, aviso2 = load_satellite_data(t2, latitudes_T2, longitudes_T2)
        sss3, sst3, aviso3 = load_satellite_data(t3, latitudes_T3, longitudes_T3)
        sss4, sst4, aviso4 = load_satellite_data(t4, latitudes_T4, longitudes_T4)
        
        data['sss1'], data['sst1'], data['aviso1'] = sss1, sst1, aviso1
        data['sss2'], data['sst2'], data['aviso2'] = sss2, sst2, aviso2
        data['sss3'], data['sst3'], data['aviso3'] = sss3, sst3, aviso3
        data['sss4'], data['sst4'], data['aviso4'] = sss4, sst4, aviso4
        
        with open(dataset_pickle_file, 'wb') as file:
            data = {
                'min_depth' : min_depth,
                'max_depth': max_depth,
                'epochs': epochs,
                'patience': patience,
                'n_components': n_components,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'dropout_prob': dropout_prob,
                'layers_config': layers_config,
                'input_params': input_params,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
                'full_dataset': full_dataset,
                'sss1': sss1,
                'sss2': sss2,
                'sss3': sss3,
                'sss4': sss4,
                'sst1': sst1,
                'sst2': sst2,
                'sst3': sst3,
                'sst4': sst4,
                'aviso1': aviso1,
                'aviso2': aviso2,
                'aviso3': aviso3,
                'aviso4': aviso4
            }
            pickle.dump(data, file)

    # Prepare the inputs
    gld_tensor1 = prepare_inputs(tt1, latitudes_T1, longitudes_T1, sss1, sst1, aviso1, input_params)
    gld_tensor2 = prepare_inputs(tt2, latitudes_T2, longitudes_T2, sss2, sst2, aviso2, input_params)
    gld_tensor3 = prepare_inputs(tt3, latitudes_T3, longitudes_T3, sss3, sst3, aviso3, input_params)
    gld_tensor4 = prepare_inputs(tt4, latitudes_T4, longitudes_T4, sss4, sst4, aviso4, input_params)

    # depth vector
    gld_depths = np.arange(0, 201*5, 5)

    pred_max_depth = 1004

    def get_glider_predictions(model, loader, tensor, device, max_depth=pred_max_depth, min_depth=0):
        tensor = tensor.to(device)

        # Get predictions
        trained_model.eval()
        with torch.no_grad():
            gld_predictions_pcs = trained_model(tensor)
            gld_predictions_pcs_cpu = gld_predictions_pcs.cpu().numpy()
            gld_predictions = val_dataset.dataset.inverse_transform(gld_predictions_pcs_cpu)
        
        #crop at max depth
        T_predictions = gld_predictions[0][min_depth : max_depth+1, :]
        S_predictions = gld_predictions[1][min_depth : max_depth+1, :]
        return T_predictions, S_predictions

    T_pred1, S_pred1 = get_glider_predictions(trained_model, val_loader, gld_tensor1, device)
    T_pred2, S_pred2 = get_glider_predictions(trained_model, val_loader, gld_tensor2, device)
    T_pred3, S_pred3 = get_glider_predictions(trained_model, val_loader, gld_tensor3, device)
    T_pred4, S_pred4 = get_glider_predictions(trained_model, val_loader, gld_tensor4, device)

    pred_depths = np.arange(0, pred_max_depth+1, 1)

    def bin_data(data, bin_size):
        """
        Bin data vertically to a given bin size.

        Args:
        - data (np.array): Data to be binned.
        - bin_size (int): Size of each bin.

        Returns:
        - np.array: Binned data.
        """
        n_rows = data.shape[0] // bin_size
        binned_data = np.mean(data[:n_rows * bin_size].reshape(n_rows, bin_size, -1), axis=1)
        return binned_data

    # Define the bin size
    bin_size = 5

    # Bin the predicted data
    T_pred1_binned = bin_data(T_pred1, bin_size)
    T_pred2_binned = bin_data(T_pred2, bin_size)
    T_pred3_binned = bin_data(T_pred3, bin_size)
    T_pred4_binned = bin_data(T_pred4, bin_size)

    S_pred1_binned = bin_data(S_pred1, bin_size)
    S_pred2_binned = bin_data(S_pred2, bin_size)
    S_pred3_binned = bin_data(S_pred3, bin_size)
    S_pred4_binned = bin_data(S_pred4, bin_size)

    # Calculate the differences
    T_diff1 = T_pred1_binned - T1
    T_diff2 = T_pred2_binned - T2
    T_diff3 = T_pred3_binned - T3
    T_diff4 = T_pred4_binned - T4
    S_diff1 = S_pred1_binned - S1
    S_diff2 = S_pred2_binned - S2
    S_diff3 = S_pred3_binned - S3
    S_diff4 = S_pred4_binned - S4

    def plot_field_subplot(data, distances, depths, variable_name, title, subplot_pos, fig):
        """
        Plot a field as a subplot.
        
        Parameters:
        data (np.array): 2D array of temperature or salinity data
        distances (np.array): 1D array of distances corresponding to the data
        depths (np.array): 1D array of depths corresponding to the data
        variable_name (str): Name of the variable ('Temperature', 'Salinity', etc.)
        title (str): Title of the subplot
        subplot_pos (int): Position of the subplot in the figure
        fig (matplotlib.figure.Figure): Figure object to plot on
        """
        ax = fig.add_subplot(subplot_pos)

        if variable_name == 'Temperature':
            vmin = 5
            vmax = 30
            step = 5
            cmap = ccm.thermal
            unit = '°C'
        elif variable_name == 'Salinity':
            vmin = 35
            vmax = 37
            step = 0.25
            cmap = ccm.haline
            unit = 'PSU'
        elif variable_name == "T Difference":
            vmin = -4
            vmax = 4
            step = 0.5
            cmap = 'bwr'
            unit = '°C'
        elif variable_name == "S Difference":
            vmin = -1
            vmax = 1
            step = 0.125
            cmap = 'PiYG'
            unit = 'PSU'
        else:
            raise ValueError(f"Invalid variable name: {variable_name}")
        
        num_levels = int((vmax - vmin) / step + 1)
        cmap = plt.get_cmap(cmap, num_levels)
        
        # rows = subplot_pos//100
        # cols = (subplot_pos%100)//10
        # id = subplot_pos%10
        # isFirstColumn = id%rows == 1
        # isLastRow = idcols == 1
        
        contour = ax.contour(distances, depths, data, levels=np.arange(vmin + step/2, np.ceil(vmax+step)+1, step), colors='black', linewidths=0.2)
        pcm = ax.pcolormesh(distances, depths, data, shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        # if isFirstColumn:
        ax.set_ylabel('Depth [m]')
        # else:
        fig.colorbar(pcm, ax=ax, label=f'{variable_name} [{unit}]', extend='both')
        # if isLastRow:
        ax.set_xlabel('Distance (km)')
        ax.grid(color='gray', linestyle='--', linewidth=0.7)
        ax.set_title(title)
        ax.invert_yaxis()

    # Create a figure for the combined plots
    fig = plt.figure(figsize=(18, 18))  # Adjust size as needed
    plot_field_subplot(T1, d1, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred1, d1, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff1, d1, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S1, d1, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred1, d1, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff1, d1, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(f"Poseidon Crossing #1 \n{t1[0].strftime('%Y-%m-%d')} to {t1[-1].strftime('%Y-%m-%d')}", fontsize=18, fontweight="bold")
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()

    #2
    fig = plt.figure(figsize=(18, 18))  # Adjust size as needed
    plot_field_subplot(T2, d2, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred2, d2, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff2, d2, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S2, d2, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred2, d2, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff2, d2, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(f"Poseidon Crossing #2 \n{t2[0].strftime('%Y-%m-%d')} to {t2[-1].strftime('%Y-%m-%d')}", fontsize=18, fontweight="bold")
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()

    #3
    fig = plt.figure(figsize=(18, 18))  # Adjust size as needed
    plot_field_subplot(T3, d3, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred3, d3, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff3, d3, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S3, d3, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred3, d3, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff3, d3, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(f"Campeche Crossing #1 and #2 \n{t3[0].strftime('%Y-%m-%d')} to {t3[-1].strftime('%Y-%m-%d')}", fontsize=18, fontweight="bold")
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()

    #4
    fig = plt.figure(figsize=(18, 18))  # Adjust size as needed
    plot_field_subplot(T4, d4, gld_depths, "Temperature", "Glider T", 321, fig)
    plot_field_subplot(T_pred4, d4, pred_depths, "Temperature", "Synthetic T", 323, fig)
    plot_field_subplot(T_diff4, d4, gld_depths, "T Difference", "T Difference", 325, fig)
    plot_field_subplot(S4, d4, gld_depths, "Salinity", "Glider S", 322, fig)
    plot_field_subplot(S_pred4, d4, pred_depths, "Salinity", "Synthetic S", 324, fig)
    plot_field_subplot(S_diff4, d4, gld_depths, "S Difference", "S Difference", 326, fig)
    plt.suptitle(f"Intense LCE \n{t4[0].strftime('%Y-%m-%d')} to {t4[-1].strftime('%Y-%m-%d')}", fontsize=18, fontweight="bold")
    plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
    plt.show()
    

    def calculate_correlation(observation, prediction):
        """
        Calculate the Pearson correlation coefficient between two 2D matrices,
        ignoring positions with NaNs in the observation matrix.

        Args:
        - observation (np.array): 2D array of observed data with NaNs for missing values.
        - prediction (np.array): 2D array of predicted data.

        Returns:
        - float: Pearson correlation coefficient, or NaN if it cannot be calculated.
        """
        # Flatten the arrays to 1D
        obs_flat = observation.flatten()
        pred_flat = prediction.flatten()

        # Create a mask for non-NaN values
        valid_mask = ~np.isnan(obs_flat)

        # Filter both arrays to include only the valid (non-NaN) values
        valid_obs = obs_flat[valid_mask]
        valid_pred = pred_flat[valid_mask]

        # Calculate the Pearson correlation coefficient on the non-NaN values
        if valid_obs.size == 0:
            return np.nan  # Return NaN if no valid observations
        correlation, _ = pearsonr(valid_obs, valid_pred)

        return correlation
    
    def average_depth(targets, depths):
        return np.nansum(targets.T * depths) / np.nansum(targets)
    
    def histogram_available_depths(targets):
        # counts all the available depths from all profiles
        return np.sum(-1*(np.isnan(targets)-1), axis=1)
    
    def equivalent_average_statistic(predictions, targets, count, depths, function):
        '''
        Adjusts the calculation of an average statistic (e.g., RMSE or bias) to account for the 
        depth binning of the primary dataset and uses the histogram of valid measurements to weight 
        these statistics.
        
        :param predictions: 2D array of predictions (profiles x depth) with 1m depth resolution.
        :param targets: 2D array of targets (profiles x depth) with 1m depth resolution.
        :param histogram: Array of counts of valid measurements at each 5m depth bin.
        :param depth_bins: The depth bins corresponding to the histogram (e.g., every 5 meters).
        :param function: The statistical function to use (e.g., rmse or bias).
        :return: The weighted average statistic across the depth bins.
        '''
        # Initialize an array to hold the statistic for each 5m bin
        stats_per_bin = np.zeros(len(depths))
        step = depths[1] - depths[0]
        
        # Iterate over each 5m bin
        for i in range(len(depths) - 1):
            # Indices of 1m data that fall into the current 5m bin
            indices = np.arange(depths[i], depths[i] + step, 1)
            
            bin_predictions = predictions[indices, :]
            bin_targets = targets[indices, :]
            stats_per_bin[i] = function(bin_predictions, bin_targets)

        # Calculate the weighted average statistic
        # Use histogram as weights, ensuring alignment in length
        weighted_stat = np.nansum(stats_per_bin * count) / np.sum(count)
        
        corr_stat = calculate_correlation(targets, predictions)
        
        return weighted_stat, corr_stat
    
    avg_d1 = int(np.round(average_depth(T1, gld_depths))) + 1
    avg_d2 = int(np.round(average_depth(T2, gld_depths))) + 1
    avg_d3 = int(np.round(average_depth(T3, gld_depths))) + 1
    avg_d4 = int(np.round(average_depth(T4, gld_depths))) + 1
    
    h1 = histogram_available_depths(T1)
    h2 = histogram_available_depths(T2)
    h3 = histogram_available_depths(T3) 
    h4 = histogram_available_depths(T4)
    
    eq_rmse_T1, eq_corr_T1 = equivalent_average_statistic(pred_T, original_profiles[:,0,:], h1, gld_depths, rmse)
    eq_bias_T1, eq_corr_T1 = equivalent_average_statistic(pred_T, original_profiles[:,0,:], h1, gld_depths, bias)
    eq_rmse_S1, eq_corr_S1 = equivalent_average_statistic(pred_S, original_profiles[:,1,:], h1, gld_depths, rmse)
    eq_bias_S1, eq_corr_S1 = equivalent_average_statistic(pred_S, original_profiles[:,1,:], h1, gld_depths, bias)
    eq_rmse_T2, eq_corr_T2 = equivalent_average_statistic(pred_T, original_profiles[:,0,:], h2, gld_depths, rmse)
    eq_bias_T2, eq_corr_T2 = equivalent_average_statistic(pred_T, original_profiles[:,0,:], h2, gld_depths, bias)
    eq_rmse_S2, eq_corr_S2 = equivalent_average_statistic(pred_S, original_profiles[:,1,:], h2, gld_depths, rmse)
    eq_bias_S2, eq_corr_S2 = equivalent_average_statistic(pred_S, original_profiles[:,1,:], h2, gld_depths, bias)
    eq_rmse_T3, eq_corr_T3 = equivalent_average_statistic(pred_T, original_profiles[:,0,:], h3, gld_depths, rmse)
    eq_bias_T3, eq_corr_T3 = equivalent_average_statistic(pred_T, original_profiles[:,0,:], h3, gld_depths, bias)
    eq_rmse_S3, eq_corr_S3 = equivalent_average_statistic(pred_S, original_profiles[:,1,:], h3, gld_depths, rmse)
    eq_bias_S3, eq_corr_S3 = equivalent_average_statistic(pred_S, original_profiles[:,1,:], h3, gld_depths, bias)
    eq_rmse_T4, eq_corr_T4 = equivalent_average_statistic(pred_T, original_profiles[:,0,:], h4, gld_depths, rmse)
    eq_bias_T4, eq_corr_T4 = equivalent_average_statistic(pred_T, original_profiles[:,0,:], h4, gld_depths, bias)
    eq_rmse_S4, eq_corr_S4 = equivalent_average_statistic(pred_S, original_profiles[:,1,:], h4, gld_depths, rmse)
    eq_bias_S4, eq_corr_S4 = equivalent_average_statistic(pred_S, original_profiles[:,1,:], h4, gld_depths, bias)    
    
    correlation_T1 = calculate_correlation(T1, T_pred1_binned)
    correlation_S1 = calculate_correlation(S1, S_pred1_binned)
    correlation_T2 = calculate_correlation(T2, T_pred2_binned)
    correlation_S2 = calculate_correlation(S2, S_pred2_binned)
    correlation_T3 = calculate_correlation(T3, T_pred3_binned)
    correlation_S3 = calculate_correlation(S3, S_pred3_binned)
    correlation_T4 = calculate_correlation(T4, T_pred4_binned)
    correlation_S4 = calculate_correlation(S4, S_pred4_binned)
    
    print("Crossing & T RMSE & T Bias & T R^2 & S RMSE & S Bias & S R^2")
    print(f"Poseidon #1 & {rmse(T_pred1_binned, T1):.3f} & {bias(T_pred1_binned, T1):.3f} & {correlation_T1:.3f} & {rmse(S_pred1_binned, S1):.3f} & {bias(S_pred1_binned, S1):.3f} & {correlation_S1:.3f}\\\\")
    print(f"Poseidon #2 & {rmse(T_pred2_binned, T2):.3f} & {bias(T_pred2_binned, T2):.3f} & {correlation_T2:.3f} & {rmse(S_pred2_binned, S2):.3f} & {bias(S_pred2_binned, S2):.3f} & {correlation_S2:.3f}\\\\")
    print(f"Campeche #1  & {rmse(T_pred3_binned, T3):.3f} & {bias(T_pred3_binned, T3):.3f} & {correlation_T3:.3f} & {rmse(S_pred3_binned, S3):.3f} & {bias(S_pred3_binned, S3):.3f} & {correlation_S3:.3f} \\\\")
    print(f"Intense LCE  & {rmse(T_pred4_binned, T4):.3f} & {bias(T_pred4_binned, T4):.3f} & {correlation_T4:.3f} & {rmse(S_pred4_binned, S4):.3f} & {bias(S_pred4_binned, S4):.3f} & {correlation_S4:.3f} \\\\")          
    
    lat_all = full_dataset.LAT
    lon_all = full_dataset.LON
    
    # Initialize lists for latitudes and longitudes of each dataset
    lat_train, lon_train = lat_all[train_indices], lon_all[train_indices]
    lat_val, lon_val = lat_all[val_indices], lon_all[val_indices]
    lat_test, lon_test = lat_all[test_indices], lon_all[test_indices]

    # Create a plot with cartopy
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale('50m'), color='black')
    ax.coastlines(resolution='50m')

    # Plot points for each dataset in one go
    ax.scatter(lon_train, lat_train, s=3, color='k', alpha=0.7, label='ARGO - Train set', transform=ccrs.Geodetic())
    ax.scatter(lon_test, lat_test, s=3, color='b', alpha=0.7, label='ARGO - Validation set', transform=ccrs.Geodetic())
    ax.scatter(lon_val, lat_val, s=30, color='r', marker='x', alpha=0.5, label='ARGO - Test set', transform=ccrs.Geodetic())
    ax.scatter(lon_train, lat_train, s=3, color='k', alpha=0.04, transform=ccrs.Geodetic())
    ax.scatter(lon_test, lat_test, s=3, color='b', alpha=0.04, transform=ccrs.Geodetic())    # Set x and y ticks, 
    ax.plot(longitudes_T1, latitudes_T1, color='c', linewidth=2, transform=ccrs.Geodetic(), label='Glider tracks')
    ax.plot(longitudes_T2, latitudes_T2, color='c', linewidth=2, transform=ccrs.Geodetic())
    ax.plot(longitudes_T3, latitudes_T3, color='c', linewidth=2, transform=ccrs.Geodetic())
    ax.plot(longitudes_T4, latitudes_T4, color='c', linewidth=2, transform=ccrs.Geodetic())
    ax.set_xticks(np.arange(-99, -79, 2))
    ax.set_yticks(np.arange(18, 34, 2))
    #add grid
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    # Add a legend
    plt.legend(loc='lower right',fontsize=14)

    plt.title("Data availability", fontsize=22, fontweight="bold")
    plt.show()
    
    # get ssh data
    aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
    bbox = (18, 32, -99, -81)
    # t1_date = datenum_to_datetime(np.median(gl_data['t1'][0]))
    # t2_date = datenum_to_datetime(np.median(gl_data['t2'][0]))
    # t3_date = datenum_to_datetime(np.median(gl_data['t3'][0]))
    # t4_date = datenum_to_datetime(np.median(gl_data['t4'][0]))
    t1_date = datenum_to_datetime(gl_data['t1'][0].mean())
    t2_date = datenum_to_datetime(gl_data['t2'][0].mean())
    t3_date = datenum_to_datetime(gl_data['t3'][0].mean())
    t4_date = datenum_to_datetime(gl_data['t4'][0].mean())
    # t1_date = datetime.combine(t1_date.date(), datetime.min.time())
    # t2_date = datetime.combine(t2_date.date(), datetime.min.time())
    # t3_date = datetime.combine(t3_date.date(), datetime.min.time())
    # t4_date = datetime.combine(t4_date.date(), datetime.min.time())
    aviso1_adt, aviso_lats, aviso_lons = get_aviso_by_date(aviso_folder, t1_date, bbox)
    X, Y = np.meshgrid(aviso_lons.values, aviso_lats.values)
    aviso2_adt, _, _ = get_aviso_by_date(aviso_folder, t2_date, bbox)
    aviso3_adt, _, _ = get_aviso_by_date(aviso_folder, t3_date, bbox)
    aviso4_adt, _, _ = get_aviso_by_date(aviso_folder, t4_date, bbox)
    
    # Create individual plots for glider tracks with the same colorbar scale
    fig = plt.figure(figsize=(12, 12))

    # Subplot 1
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND.with_scale('50m'), color='black', zorder=0)
    ax1.coastlines(resolution='50m')
    cf1 = ax1.contourf(X, Y, aviso1_adt.adt.values, cmap='jet', levels=50, extend='both')
    ax1.plot(longitudes_T1, latitudes_T1, color='#FF69B4', linewidth=2, transform=ccrs.Geodetic(), label='Poseidon crossing #1')
    ax1.set_xticks(np.arange(-99, -79, 2))
    ax1.set_yticks(np.arange(18, 34, 2))
    ax1.set_extent([-99, -81, 18, 32])
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # Add colorbar for subplot 1
    cbar1 = plt.colorbar(cf1, ax=ax1, fraction=0.036, pad=0.04)
    # Format the colorbar ticks to two decimal places
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Add colorbar title
    cbar1.set_label('ADT (m)', fontsize=10)
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar1.set_label('ADT (m)', fontsize=8)
    cbar1.ax.tick_params(labelsize=8)  # Make the colorbar ticks smaller
    ax1.title.set_text(t1_date.strftime('%Y-%m-%d'))
    
    # Subplot 2
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND.with_scale('50m'), color='black', zorder=0)
    ax2.coastlines(resolution='50m')
    cf2 = ax2.contourf(X, Y, aviso2_adt.adt.values, cmap='jet', levels=50, extend='both')
    ax2.plot(longitudes_T2, latitudes_T2, color='#FF69B4', linewidth=2, transform=ccrs.Geodetic(), label='Poseidon crossing #2')
    ax2.set_xticks(np.arange(-99, -79, 2))
    ax2.set_yticks(np.arange(18, 34, 2))
    ax2.set_extent([-99, -81, 18, 32])
    ax2.grid(color='gray', linestyle='--', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # Add colorbar for subplot 2
    cbar2 = plt.colorbar(cf2, ax=ax2, fraction=0.036, pad=0.04)
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar2.set_label('ADT (m)', fontsize=8)
    cbar2.ax.tick_params(labelsize=8)
    ax2.title.set_text(t2_date.strftime('%Y-%m-%d'))

    # Subplot 3
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3.add_feature(cfeature.LAND.with_scale('50m'), color='black', zorder=0)
    ax3.coastlines(resolution='50m')
    cf3 = ax3.contourf(X, Y, aviso3_adt.adt.values, cmap='jet', levels=50, extend='both')
    ax3.plot(longitudes_T3, latitudes_T3, color='#FF69B4', linewidth=2, transform=ccrs.Geodetic(), label='Campeche')
    ax3.set_xticks(np.arange(-99, -79, 2))
    ax3.set_yticks(np.arange(18, 34, 2))
    ax3.set_extent([-99, -81, 18, 32])
    ax3.grid(color='gray', linestyle='--', linewidth=0.5)
    ax3.tick_params(axis='both', which='major', labelsize=10)

    # Add colorbar for subplot 3
    cbar3 = plt.colorbar(cf3, ax=ax3, fraction=0.036, pad=0.04)
    cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar3.set_label('ADT (m)', fontsize=8)
    cbar3.ax.tick_params(labelsize=8)
    ax3.title.set_text(t3_date.strftime('%Y-%m-%d'))

    # Subplot 4
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())
    ax4.add_feature(cfeature.LAND.with_scale('50m'), color='black', zorder=0)
    ax4.coastlines(resolution='50m')
    cf4 = ax4.contourf(X, Y, aviso4_adt.adt.values, cmap='jet', levels=50, extend='both')
    ax4.plot(longitudes_T4, latitudes_T4, color='#FF69B4', linewidth=2, transform=ccrs.Geodetic(), label='Intense LCE')
    ax4.set_xticks(np.arange(-99, -79, 2))
    ax4.set_yticks(np.arange(18, 34, 2))
    ax4.set_extent([-99, -81, 18, 32])
    ax4.grid(color='gray', linestyle='--', linewidth=0.5)
    ax4.tick_params(axis='both', which='major', labelsize=10)

    # Add colorbar for subplot 4
    cbar4 = plt.colorbar(cf4, ax=ax4, fraction=0.036, pad=0.04)
    cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar4.set_label('ADT (m)', fontsize=8)
    cbar4.ax.tick_params(labelsize=8)
    ax4.title.set_text(t4_date.strftime('%Y-%m-%d'))

    # Add a title for the entire figure
    plt.suptitle("Gliders", fontsize=22, fontweight="bold")

    # Display the plot
    plt.show()

    # # Meunier data plots



    # # Perform linear regression
    # slope, intercept, r_value, p_value, std_err = linregress(full_dataset.SH1950, full_dataset.AVISO_ADT)

    # # Print the results
    # print(f"Slope: {slope}")
    # print(f"Intercept: {intercept}")
    # print(f"Coefficient of determination (R²): {r_value**2}")
    # print(f"P-value: {p_value}")
    # print(f"Standard error of the regression estimate: {std_err}")

    # # Normalize ADT values for comparison
    # ADT_normalized = full_dataset.SH1950*slope + intercept

    # # Plot histograms
    # plt.figure(figsize=(6, 6))

    # # Histogram of ADT
    # plt.hist(ADT_normalized, bins=100, alpha=0.5, density=True, label='SH1950', edgecolor='k')

    # # Generate a KDE plot for SSH
    # sns.kdeplot(full_dataset.AVISO_ADT, color="r", label='ADT')

    # plt.xlabel('SSH [m]')
    # plt.ylabel('Frequency')
    # plt.title('SSH Distribution')
    # plt.legend(loc='upper right')

    # plt.show()
    
    # # create a plot of the location and SH of the profiles:
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.add_feature(cfeature.LAND.with_scale('50m'), color='black')
    # # ax.add_feature(cfeature.OCEAN.with_scale('110m'))  # Adds ocean feature, might include basic bathymetry
    # ax.coastlines(resolution='50m')

    # # Plot points for each dataset in one go
    # scatter = ax.scatter(full_dataset.LON, full_dataset.LAT, c = ADT_normalized, transform=ccrs.Geodetic(), s=2, cmap='jet')
    # ax.set_xticks(np.arange(-99, -79, 2))
    # ax.set_yticks(np.arange(18, 34, 2))
    # plt.xticks(fontsize=11)  # Adjust the font size as needed
    # plt.yticks(fontsize=11)  # Adjust the font size as needed
    # ax.grid(color='gray', linestyle='--', linewidth=0.51)
    # cbar = plt.colorbar(scatter, label='SH1950', shrink=0.78)
    # cbar.set_label('SH1950 [m]', fontsize=16)
    # cbar.ax.tick_params(labelsize=16)
    # plt.xlabel('Longitude', fontsize=16)
    # plt.ylabel('Latitude', fontsize=16)
    # plt.title('ARGO locations')
    # plt.show()
    
    # # T-S diagram
        
    # tempL=np.linspace(np.min(full_dataset.TEMP)-1,np.max(full_dataset.TEMP)+1,156)

    # salL=np.linspace(np.min(full_dataset.SAL)-1,np.max(full_dataset.SAL)+1,156)

    # Tg, Sg = np.meshgrid(tempL,salL)
    # sigma_theta = gsw.sigma0(Sg, Tg)
    # cnt = np.linspace(sigma_theta.min(), sigma_theta.max(),156)
    
    # # Normalize the ADT values for color mapping
    # norm = mcolors.Normalize(vmin=ADT_normalized.min(), vmax=ADT_normalized.max())
    # cmap = plt.cm.jet  # Choose a colormap

    # # Create the T-S plot
    # fig, ax = plt.subplots(figsize=(10, 8))
    # cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', zorder=1)
    
    # # Plot each line
    # for i in range(full_dataset.TEMP.shape[1]):  # Assuming TEMP and SAL have the same second dimension
    #     # TEMP[:, i] and SAL[:, i] form the x and y coordinates of the ith line
    #     color = cmap(norm(ADT_normalized[i]))  # Map the ADT value to a color
    #     ax.plot(full_dataset.SAL[:, i], full_dataset.TEMP[:, i], color=color, linewidth=0.5)

    # for i in range(pred_T.shape[1]):  # Assuming TEMP and SAL have the same second dimension
    #     # TEMP[:, i] and SAL[:, i] form the x and y coordinates of the ith line
    #     # color = cmap(norm(ADT_normalized[i]))  # Map the ADT value to a color
    #     ax.plot(pred_S[:, i], pred_T[:, i], color='pink', linewidth=0.2)
    
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # This line is necessary for ScalarMappable to work with colorbar
    
    # # Mark the cores of the water masses with circles and labels
    # cores = {
    #     "SAAIW ": (34.9, 6.5),
    #     "GCW ": (36.4, 22.3),
    #     "NASUW ": (36.8, 22)
    # }

    # for label, (salinity, temperature) in cores.items():
    #     ax.plot(salinity, temperature, 'o', markersize=7, color='black')
    #     ax.text(salinity, temperature, label, fontsize=11, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')

    # # cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.036, pad=0.04)
    # # cbar.set_label('SH1950', fontsize=12)
    # # cbar.ax.tick_params(labelsize=11)
    # # set x lims to 34.5 to 37.5
    # ax.set_xlim(34.5, 37.5)
    # cl=plt.clabel(cs,fontsize=10,inline=False,fmt='%.1f',colors='k')
    # plt.xlabel('Salinity [PSU]')
    # plt.ylabel('Temperature [°C]')
    # plt.title('T-S Diagram')
    
    # # ax.set_xticks(np.arange(34.5, 37.5, 0.5))

    # plt.show()
    
    # fig = plt.figure(figsize=(6, 6))
    # plt.plot(ADT_normalized, full_dataset.AVISO_ADT, '.', markersize=0.6)
    # # add a trend line
    # plt.plot([ADT_normalized.min(), ADT_normalized.max()], [ADT_normalized.min(), ADT_normalized.max()], 'k')
    # plt.xlabel('SH1950 [m]')
    # plt.ylabel('SSH [m]')
    # plt.title('SH1950 vs SSH')
    # plt.show()
    
    # # compare T and S profile against PCA reconstruction
    # prof_number = 300
    # prof_number = np.atleast_1d(prof_number)
    # depths = np.arange(0, 501, 1)
    # pca_prof = full_dataset.get_profiles(prof_number,True)
    # pca_T = pca_prof[depths,0,:]
    # pca_S = pca_prof[depths,1,:]
    # ori_prof = full_dataset.get_profiles(prof_number,False)
    # ori_T = ori_prof[depths,0,:]
    # ori_S = ori_prof[depths,1,:]
    
    
    # fig = plt.figure(figsize=(10, 8))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.plot(pca_T, depths, label='15 PCS recon', color='r', linewidth=2)
    # ax1.plot(ori_T, depths, label='Original', color='k', linewidth=1)
    # ax1.set_xlabel('Temperature [°C]')
    # ax1.set_ylabel('Depth [m]')
    # ax1.set_title('Temperature')
    # ax1.grid()
    # #invert y axis
    # ax1.invert_yaxis()
    # ax1.legend(fontsize=12)
    
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.plot(pca_S, depths, label='15 PCS recon', color='c', linewidth=2)
    # ax2.plot(ori_S, depths, label='Original', color='k', linewidth=1)
    # ax2.set_xlabel('Salinity [PSU])')
    # # ax2.set_ylabel('Depth [m]')
    # ax2.set_title('Salinity')
    # ax2.grid()
    # #invert y axis
    # ax2.invert_yaxis()
    # ax2.legend(fontsize=12)
    
    # calculate average bias and rmse for depth ranges
    print("Depth range \t NeSPReSO T RMSE \t GEM T RMSE \t MLR T RMSE \t ISOP T RMSE \t NeSPReSO T Bias \t GEM T Bias \t MLR T Bias \t ISOP T Bias \t NeSPReSO S RMSE \t GEM S RMSE \t MLR S RMSE \t ISOP S RMSE \t NeSPReSO S Bias \t GEM S Bias \t MLR S Bias \t ISOP S Bias \t NeSPReSO T R^2 \t GEM T R^2 \t MLR T R^2 \t NeSPReSO S R^2 \t GEM S R^2 \t MLR S R^2")
    intervals = [(min_depth, 20), (20, 100), (100, 200), (200, 500), (500, 1000), (1000, max_depth), (0, 1000), (min_depth, max_depth)]    
    
    for i in range(len(intervals)):
        min_d, max_d = intervals[i]
        i_isop_dpt = np.where((isop_depths >= min_d) & (isop_depths <= max_d))[0]
        calc_depths = isop_depths[i_isop_dpt].astype(int)
        # NN
            # gem_temp_errors = (gem_temp.T - original_profiles[:, 0, :]) ** 2
            # gem_sal_errors = (gem_sal.T - original_profiles[:, 1, :]) ** 2

            # nn_temp_errors = (pred_T[:, :] - original_profiles[:, 0, :]) ** 2
            # nn_sal_errors = (pred_S[:, :] - original_profiles[:, 1, :]) ** 2
        ori_t = original_profiles[calc_depths, 0, :]
        ori_s = original_profiles[calc_depths, 1, :]
        nn_t = pred_T[calc_depths, :]
        nn_s = pred_S[calc_depths, :]
        gem_t = gem_temp[:,calc_depths].T
        gem_s = gem_sal[:,calc_depths].T
        mlr_t = temp_MLR_profiles[calc_depths,:]
        mlr_s = sal_MLR_profiles[calc_depths,:]
        
        nn_t_rmse = rmse(nn_t, ori_t)
        gem_t_rmse = rmse(gem_t, ori_t)
        nn_t_bias = bias(nn_t, ori_t)
        gem_t_bias = bias(gem_t, ori_t)
        nn_s_rmse = rmse(nn_s, ori_s)
        gem_s_rmse = rmse(gem_s, ori_s)
        nn_s_bias = bias(nn_s, ori_s)
        gem_s_bias = bias(gem_s, ori_s)
        
        mlr_t_rmse = rmse(mlr_t, ori_t)
        mlr_t_bias = bias(mlr_t, ori_t)
        mlr_s_rmse = rmse(mlr_s, ori_s)
        mlr_s_bias = bias(mlr_s, ori_s)
        
        isop_avg_t_rmse = np.mean(ist.rmse.values[i_isop_dpt])
        isop_avg_t_bias = np.mean(ist.bias.values[i_isop_dpt])
        isop_avg_s_rmse = np.mean(iss.rmse.values[i_isop_dpt])
        isop_avg_s_bias = np.mean(iss.bias.values[i_isop_dpt])
        
        nn_T_corr = calculate_correlation(nn_t, ori_t)
        gem_T_corr = calculate_correlation(gem_t, ori_t)
        nn_S_corr = calculate_correlation(nn_s, ori_s)
        gem_S_corr = calculate_correlation(gem_s, ori_s)
        mlr_T_corr = calculate_correlation(mlr_t, ori_t)
        mlr_S_corr = calculate_correlation(mlr_s, ori_s)
                
        print(f"[{min_d}-{max_d}] \t {nn_t_rmse:.3f} \t {gem_t_rmse:.3f} \t {mlr_t_rmse:.3f} \t {isop_avg_t_rmse:.3f} \t {nn_t_bias:.3f} \t {gem_t_bias:.3f} \t {mlr_t_bias:.3f} \t {isop_avg_t_bias:.3f} \t {nn_s_rmse:.3f} \t {gem_s_rmse:.3f} \t {mlr_s_rmse:.3f} \t {isop_avg_s_rmse:.3f} \t {nn_s_bias:.3f} \t {gem_s_bias:.3f} \t {mlr_s_bias:.3f} \t {isop_avg_s_bias:.3f} \t {nn_T_corr:.3f} \t {gem_T_corr:.3f} \t {mlr_T_corr:.3f} \t {nn_S_corr:.3f} \t {gem_S_corr:.3f} \t {mlr_S_corr:.3f}")
        # print(f"[{min_d}-{max_d}] \t {nn_t_rmse:.3f} \t {gem_t_rmse:.3f} \t {isop_avg_t_rmse:.3f} \t {nn_t_bias:.3f} \t {gem_t_bias:.3f} \t {isop_avg_t_bias:.3f} \t {nn_s_rmse:.3f} \t {gem_s_rmse:.3f} \t {isop_avg_s_rmse:.3f} \t {nn_s_bias:.3f} \t {gem_s_bias:.3f} \t {isop_avg_s_bias:.3f} \t {nn_T_corr:.3f} \t {gem_T_corr:.3f} \t {nn_S_corr:.3f} \t {gem_S_corr:.3f}")
        # print("\hline")
        
        
    # #create a netcdf file with the validation dataset
    # sst_val = get_inputs(val_loader, device)[:,-2]
    # lat_val = val_dataset.dataset.LAT[val_indices]
    # lon_val = val_dataset.dataset.LON[val_indices]
    # date_val = datenums_to_datetimes(val_dataset.dataset.TIME[val_indices])
    # T_profiles_val = original_profiles[:,0,:]
    # S_profiles_val = original_profiles[:,1,:]
    # depth = np.arange(0,1801)

    # sst_val.shape, lat_val.shape, lon_val.shape, type(date_val), type(date_val[0]), len(date_val), T_profiles_val.shape, S_profiles_val.shape, depth.shape

    # from netCDF4 import Dataset
    # # import numpy as np
    # from datetime import datetime, timedelta

    # def create_netcdf(filename, sst_val, lat_val, lon_val, date_val, T_profiles_val, S_profiles_val, depth):
    #     with Dataset(filename, 'w', format='NETCDF4') as nc:
    #         # Dimensions
    #         nc.createDimension('time', len(date_val))
    #         nc.createDimension('lat', len(lat_val))
    #         nc.createDimension('lon', len(lon_val))
    #         nc.createDimension('depth', len(depth))
            
    #         # Variables
    #         times = nc.createVariable('time', 'f8', ('time',))
    #         lats = nc.createVariable('lat', 'f4', ('lat',))
    #         lons = nc.createVariable('lon', 'f4', ('lon',))
    #         depths = nc.createVariable('depth', 'f4', ('depth',))
    #         sst = nc.createVariable('sst', 'f4', ('time',))
    #         T_profiles = nc.createVariable('T_profiles', 'f4', ('depth', 'time'))
    #         S_profiles = nc.createVariable('S_profiles', 'f4', ('depth', 'time'))
            
    #         # Convert datetime to numeric time values
    #         ref_date = datetime(1900, 1, 1)
    #         numeric_dates = [(d - ref_date).total_seconds() for d in date_val]
    #         times[:] = numeric_dates
            
    #         # Assign data
    #         lats[:] = lat_val
    #         lons[:] = lon_val
    #         depths[:] = depth
    #         sst[:] = sst_val
    #         T_profiles[:, :] = T_profiles_val
    #         S_profiles[:, :] = S_profiles_val
            
    #         # Add attributes
    #         nc.description = 'Dataset used for acquiring statistics for ISOP, GEM and NeSPReSO methods. Contains SST, latitude, longitude, date, temperature profiles, salinity profiles, and depth.'
    #         sst.units = 'Celsius'
    #         lats.units = 'degrees'
    #         lons.units = 'degrees'
    #         depths.units = 'meter'
    #         times.units = 'seconds since 1900-01-01 00:00:00'
    #         T_profiles.units = 'Celsius'
    #         S_profiles.units = 'PSU'
            
    #         print(f"NetCDF file '{filename}' created successfully.")
            
    # # making a histogream of missing dates
    # full_dataset.data['TIME']
    # full_dataset.TIME
    # dates = datenums_to_datetimes(np.sort(full_dataset.data['TIME'][np.isin(full_dataset.data['TIME'], full_dataset.TIME, invert=True)]))
    # # Extracting year and month for each date
    # date_counts = Counter([(date.year, date.month) for date in dates])

    # # Sorting the dates for plotting
    # sorted_date_counts = dict(sorted(date_counts.items()))

    # # Creating labels and values for the histogram
    # labels = [f"{year}-{month:02}" for year, month in sorted_date_counts.keys()]
    # values = list(sorted_date_counts.values())

    # # Plotting the histogram
    # plt.figure(figsize=(22, 14))
    # plt.bar(labels, values)
    # plt.xlabel('Year-Month')
    # plt.ylabel('Frequency')
    # plt.title('Monthly Histogram of Dates')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
    
    # filename = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/Test_dataset.nc"
    # # Creating the NetCDF file
    # create_netcdf(filename, sst_val, lat_val, lon_val, date_val, T_profiles_val, S_profiles_val, depth)

    # xr.open_dataset(filename).depth
    
    # # %%
    # from scipy.spatial import cKDTree

    # # lon_min = -88
    # # lon_max = -82
    # lon_min = np.floor(np.min(full_dataset.LON))
    # lon_max =  np.ceil(np.max(full_dataset.LON))
    # lat_min = np.floor(np.min(full_dataset.LAT))
    # lat_max =  np.ceil(np.max(full_dataset.LAT))

    # # Define grid spacing
    # grid_spacing = 0.1  # degrees

    # # Create the grid within the bounding box
    # lats_grid = np.arange(lat_min, lat_max + grid_spacing, grid_spacing)
    # lons_grid = np.arange(lon_min, lon_max + grid_spacing, grid_spacing)

    # # Use meshgrid to create a grid of coordinates
    # lats_mesh, lons_mesh = np.meshgrid(lats_grid, lons_grid)

    # # Flatten the meshgrid arrays to obtain the full list of coordinates
    # grid_points = np.vstack([lats_mesh.ravel(), lons_mesh.ravel()]).T

    # # Build a KD-Tree with the original LAT and LON data
    # data_points = np.vstack([full_dataset.LAT, full_dataset.LON]).T
    # tree = cKDTree(data_points)

    # # Query the tree for each grid point to find the distance to the nearest data point
    # distances, _ = tree.query(grid_points, distance_upper_bound=0.5)

    # # Filter the grid points where the distance is infinity (no points within 0.2 degrees)
    # filtered_grid_points = grid_points[distances != np.inf]

    # # Plot original data points and the filtered grid points
    # plt.figure(figsize=(10, 8))
    # plt.scatter(filtered_grid_points[:, 1], filtered_grid_points[:, 0], color='red', label='Filtered Grid Points', s=0.2)
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Points within 0.5 degrees of original data')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # %%
#     def calculate_sound_speed_NPL(T, S, Z, Phi=45):
#     """
#     Calculate sound speed (in m/s) using the NPL equation.
#     T: Temperature in degrees Celsius
#     S: Salinity in PSU
#     Z: Depth in meters
#     Phi: Latitude in degrees (default 45)
#     """
#     c = (1402.5 + 5 * T - 5.44e-2 * T**2 + 2.1e-4 * T**3 
#          + 1.33 * S - 1.23e-2 * S * T + 8.7e-5 * S * T**2 
#          + 1.56e-2 * Z + 2.55e-7 * Z**2 - 7.3e-12 * Z**3 
#          + 1.2e-6 * Z * (Phi - 45) - 9.5e-13 * T * Z**3 
#          + 3e-7 * T**2 * Z + 1.43e-5 * S * Z)
#     return c

# # Recalculate sound speed at each depth using the NPL equation
# sound_speed_profile_NPL = np.array([calculate_sound_speed_NPL(T, S, z) for T, S, z in zip(temperature_profile, salinity_profile, depths)])

# # Finding the Sonic Layer Depth (SLD) using the NPL equation
# max_sound_speed_index_NPL = np.argmax(sound_speed_profile_NPL)
# SLD_NPL = depths[max_sound_speed_index_NPL]
# # Conversion factor from meters to feet
# meters_to_feet = 3.28084

# # Conversion factor for the gradient from per feet to per 100 meters
# conversion_factor = meters_to_feet / 100

# # Calculating the Below Layer Gradient (BLG) using the NPL equation
# gradient_NPL = np.gradient(sound_speed_profile_NPL, depths_feet)
# # Average gradient below MLD in m/s per 100 feet using the NPL equation
# BLG_NPL = np.mean(gradient_NPL[MLD_index:]) * conversion_factor

# ## Eddy experiment - nature run stuff

# #compare ssh distributions:

#     from matplotlib.ticker import PercentFormatter
#     from scipy.io import loadmat

#     def aggregate_from_mat(folder_path, *variable_names):
#         aggregated_data = {var_name: [] for var_name in variable_names}

#         # Loop through all files in the directory
#         for filename in os.listdir(folder_path):
#             if filename.endswith('.mat'):
#                 file_path = os.path.join(folder_path, filename)
#                 mat_data = loadmat(file_path)
                
#                 # Check if each variable exists in the .mat file and aggregate
#                 for var_name in variable_names:
#                     if var_name in mat_data:
#                         var_data = mat_data[var_name]
#                         aggregated_data[var_name].append(np.expand_dims(var_data, axis=-1))
#                     else:
#                         print(f"'{var_name}' not found in {filename}")
            
#         # Combine all variable data into single numpy arrays along the new axis
#         for var_name in variable_names:
#             if aggregated_data[var_name]:
#                 aggregated_data[var_name] = np.concatenate(aggregated_data[var_name], axis=-1)
#             else:
#                 print(f"No '{var_name}' data found in any .mat files.")
        
#         return aggregated_data

#     # Example usage:
#     folder_path = '/unity/g2/jmiranda/SubsurfaceFields/Data/NatureRun/'
#     ssh_nature_run = aggregate_from_mat(folder_path, 'ssh10')['ssh10'].flatten()

#     a = full_dataset.AVISO_ADT
#     n, bins, _ = plt.hist(a, weights=np.ones(len(a))/len(a), bins=100, color='blue', label='Training AVISO SSH')
#     plt.hist(ssh_nature_run, weights=np.ones(len(ssh_nature_run))/len(ssh_nature_run), bins=bins, color='red', label='Nature run SSH')
#     plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

#     # Set custom x-ticks every 0.1 from -0.4 to 0.9
#     plt.xticks(np.arange(-0.4, 1.0, 0.1), fontsize=11)
#     plt.yticks(fontsize=11)
#     plt.legend(fontsize=11)



#     # Compare T/S diagrams for ssh ranges
#     ssh_nature_run = aggregate_from_mat(folder_path, 'ssh10')['ssh10']
#     T_nature_run = aggregate_from_mat(folder_path, 'temp10')['temp10']
#     S_nature_run = aggregate_from_mat(folder_path, 'sal10')['sal10']

#     import matplotlib.colors as mcolors

#     def plot_ts_profiles(datasets, dataset_labels, sigma_theta, Sg, Tg, cores, cmap_name='viridis'):
#         """
#         Plots T-S profiles from multiple datasets on the same plot.
        
#         Parameters:
#         - datasets: List of tuples [(TEMP1, SAL1), (TEMP2, SAL2), ...]
#                     Each tuple contains temperature and salinity data.
#         - dataset_labels: List of labels corresponding to each dataset.
#         - sigma_theta: 2D array of sigma_theta values for contour plotting.
#         - Sg: 2D array of salinity grid values for contour plotting.
#         - Tg: 2D array of temperature grid values for contour plotting.
#         - cores: Dictionary containing core water mass points to be marked on the plot.
#                 Example: {"SAAIW": (34.9, 6.5), "GCW": (36.4, 22.3), "NASUW": (36.8, 22)}
#         - cmap_name: Name of the color map to use for distinguishing datasets (default: 'viridis').
        
#         Returns:
#         - None
#         """
        
#         # Initialize the plot
#         fig, ax = plt.subplots(figsize=(10, 8))
        
#         # Plot sigma_theta contours
#         cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', zorder=1)
        
#         # Create a color map
#         cmap = plt.get_cmap(cmap_name)
#         colors = cmap(np.linspace(0, 1, len(datasets)))
        
#         # Plot T-S profiles for each dataset
#         for idx, (TEMP, SAL) in enumerate(datasets):
#             label = dataset_labels[idx]
#             color = colors[idx]
            
#             # Ensure TEMP and SAL are 2D arrays for plotting
#             if TEMP.ndim == 1:
#                 TEMP = TEMP[:, np.newaxis]
#             if SAL.ndim == 1:
#                 SAL = SAL[:, np.newaxis]
            
#             for i in range(TEMP.shape[1]):  # Plot each profile in the dataset
#                 ax.plot(SAL[:, i], TEMP[:, i], color=color, linewidth=0.5, label=label if i == 0 else "")
        
#         # Plot core water masses
#         for label, (salinity, temperature) in cores.items():
#             ax.plot(salinity, temperature, 'o', markersize=7, color='black')
#             ax.text(salinity, temperature, label, fontsize=11, verticalalignment='bottom', horizontalalignment='right', fontweight='bold')

#         # Configure the plot
#         ax.set_xlim(34.5, 37.5)
#         plt.clabel(cs, fontsize=10, inline=False, fmt='%.1f', colors='k')
#         plt.xlabel('Salinity [PSU]')
#         plt.ylabel('Temperature [°C]')
#         plt.title('T-S Diagram')
#         plt.legend(fontsize=10)
#         plt.show()
        
#     def index_for_range(data, min_val, max_val):
#         return np.where((data >= min_val) & (data <= max_val))[0]

#     # # Filter data based on SSH ranges
#     # ssh_nature_run = ssh_nature_run.flatten()  # Assuming SSH values need to be compared
#     # # Remove NaN values and corresponding indices from b, T_nature_run, and S_nature_run
#     # valid_indices = ~np.isnan(ssh_nature_run)
#     # ssh_nature_run = ssh_nature_run[valid_indices]
#     # T_nature_run = T_nature_run[valid_indices]
#     # S_nature_run = S_nature_run[valid_indices]
#     # ssh_05to_01 = index_for_range(ssh_nature_run, -0.05, -0.01)
#     # ssh_01to01 = index_for_range(ssh_nature_run, -0.01, 0.01)
#     # ssh_01to10 = index_for_range(ssh_nature_run, 0.01, 0.1)
#     # ssh_10to30 = index_for_range(ssh_nature_run, 0.1, 0.3)

#     # # Build datasets with correct dimensions
#     # datasets = [
#     #     (T_nature_run[ssh_05to_01], S_nature_run[ssh_05to_01]),
#     #     (T_nature_run[ssh_01to01], S_nature_run[ssh_01to01]),
#     #     (T_nature_run[ssh_01to10], S_nature_run[ssh_01to10]),
#     #     (T_nature_run[ssh_10to30], S_nature_run[ssh_10to30])
#     # ]

#     # dataset_labels = ['SSH -0.05 to -0.01', 'SSH -0.01 to 0.01', 'SSH 0.01 to 0.1', 'SSH 0.1 to 0.3']

#     # # Plotting the T-S profiles
#     # plot_ts_profiles(datasets, dataset_labels, sigma_theta, Sg, Tg, cores, cmap_name='viridis')
    
    ## Reviews
    
    # Make a bar plot showing how many profiles are in the training, validation and test datasets per month

    
    def count_profiles_per_month(dataset, indices):
        dates = [datetime.fromordinal(int(d)) for d in dataset.TIME[indices]]
        df = pd.DataFrame({'date': dates})
        return df.groupby(df['date'].dt.month).size().reindex(range(1, 13), fill_value=0)

    train_counts = count_profiles_per_month(train_dataset.dataset, train_indices)
    val_counts = count_profiles_per_month(val_dataset.dataset, val_indices)
    test_counts = count_profiles_per_month(test_dataset.dataset, test_indices)

    # Combine all dates and get unique months
    all_months = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))
    
    # Combine all counts into a single DataFrame
    df = pd.DataFrame({
        'Train': train_counts,
        'Validation': val_counts,
        'Test': test_counts
    })

    # Calculate the total number of profiles for each month
    df_total = df.sum(axis=1)
    # Calculate the percentage for each dataset
    df_percentage = df.div(df_total, axis=0) * 100

    # Update the index to display month abbreviations
    df_percentage.index = [calendar.month_abbr[i] for i in df_percentage.index]

    # Plot
    ax = df_percentage.plot(kind='bar', stacked=True, figsize=(15, 6), width=0.8)
    plt.title('Profiles per Month')
    plt.xlabel('Month')
    plt.ylabel('%')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncols=3)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add total number labels on top of each bar
    for i, total in enumerate(df_total):
        ax.text(i, 0, f'Total:\n{total:,.0f}', ha='center', va='bottom', )

    # Add percentage labels on each bar segment
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='center')

    # Set y-axis to show percentages from 0 to 100
    plt.ylim(0, 100)  # Increase to 105 to accommodate total labels?

    plt.tight_layout()
    plt.show()