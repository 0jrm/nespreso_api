import torch
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA
import mat73


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
                sst_date, sst_lats, sst_lons = get_sst_ghrsst_by_date(sst_folder, c_date, bbox)
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
 
